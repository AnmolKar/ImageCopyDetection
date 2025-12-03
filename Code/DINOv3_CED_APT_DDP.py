import os
import sys
import json
import warnings
import inspect
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch import amp
import torch.backends.cudnn as cudnn

from transformers import AutoModel, AutoProcessor
from huggingface_hub import login
from dotenv import load_dotenv

import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as V

# ======================================================================
# Import your loaders + eval utilities
# ======================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
UTILITIES_DIR = SCRIPT_DIR / "Utilities"

if str(UTILITIES_DIR) not in sys.path:
    sys.path.append(str(UTILITIES_DIR))

from disc21_loader import (  # type: ignore
    Disc21DataConfig,
    build_transforms,
    get_train_dataset,
    get_reference_dataset,
    get_query_dataset,
    get_pair_dataset,
    create_dataloader,
    load_groundtruth,
)

from ndec_loader import (  # type: ignore
    NdecDataConfig,
    build_default_loaders as build_ndec_loaders,
    load_groundtruth as load_ndec_groundtruth,
)

from eval_utils import (
    compute_descriptors_for_loader,
    evaluate_retrieval,
    compute_descriptors_for_loader_tta,
    cosine_similarity,
    compute_muap_and_rp90,
    benchmark_inference,
)

# ----------------------------------------------------------------------
# Global device placeholder (set inside main() after DDP init)
# ----------------------------------------------------------------------
device: torch.device


# ======================================================================
# Distributed setup helpers
# ======================================================================
def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize torch.distributed using environment variables set by torchrun.

    Returns:
        rank: global rank
        world_size: total number of processes
        local_rank: index of GPU on this node to use in this process
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Fallback: single-process, single-GPU (no true DDP)
        rank, world_size, local_rank = 0, 1, 0
        # torch.distributed uses env:// rendezvous, so provide sane defaults
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ======================================================================
# Configs
# ======================================================================
@dataclass
class ExperimentConfig:
    disc21_root: Path = WORKSPACE_ROOT / "DISC21"
    ndec_root: Path = WORKSPACE_ROOT / "NDEC"
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    img_size_train: int = 224
    img_size_eval: int = 224
    batch_size_train: int = 32
    batch_size_eval: int = 32
    batch_size_pairs: int = 32
    num_workers: int = 4  # still used by Disc21DataConfig / NdecDataConfig
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    temperature_ntxent: float = 0.2
    temperature_kl: float = 0.1
    lambda_kl: float = 1.0
    lambda_local: float = 1.0
    lambda_bce: float = 1.0
    lambda_asl_mtr: float = 1.0

    # Epochs
    num_epochs_disc21: int = 30
    num_epochs_ndec: int = 20

    # Early stopping (DISC21)
    early_stopping_patience_disc21: int = 4
    early_stopping_min_delta_disc21: float = 5e-5
    min_epochs_disc21: int = 15

    # Early stopping (NDEC)
    early_stopping_patience_ndec: int = 2
    early_stopping_min_delta_ndec: float = 5e-5
    min_epochs_ndec: int = 5

    checkpoint_path: Path = Path("artifacts/checkpoints/ced_apt_model_ddp.pt")
    encoding_cache_root: Path = Path("artifacts/sinov3_ced_encoding")
    encoding_chunk_size: int = 1024
    eval_query_chunk_size: int = 256
    eval_ref_chunk_size: int = 8192
    eval_global_pairs_per_query: int = 512


# ======================================================================
# Backbone, aggregator, classifier, CEDModel (+ APT)
# ======================================================================
class DinoV3Backbone(nn.Module):
    """Wrapper that exposes token-level DINOv3 activations."""

    def __init__(self, model_name: str):
        super().__init__()
        dtype = torch.float32  # keep backbone full precision for stability

        self.processor = self._load_processor(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(device)
        self._supports_attn_impl = hasattr(self.model, "set_attn_implementation")
        current_impl = getattr(self.model, "_attn_implementation", None)
        self._attn_impl_is_eager = current_impl == "eager"
        self.patch_size = getattr(self.model.config, "patch_size", 16)
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.hidden_size = self.model.config.hidden_size

    @staticmethod
    def _load_processor(model_name: str):
        """
        Try AutoImageProcessor if available; otherwise fall back to AutoProcessor.
        This avoids ImportError on older transformers versions.
        """
        try:
            from transformers import AutoImageProcessor  # type: ignore
            return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # NOTE: no @torch.inference_mode here – that caused the runtime error.
    def preprocess(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use no_grad just for the CPU-side numpy + processor work
        with torch.no_grad():
            images_np = [img.detach().cpu().permute(1, 2, 0).numpy() for img in images]
            encoded = self.processor(images=images_np, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    def _ensure_eager_attn(self):
        if self._supports_attn_impl and not self._attn_impl_is_eager:
            try:
                self.model.set_attn_implementation("eager")
                self._attn_impl_is_eager = True
            except Exception as exc:
                warnings.warn(
                    f"DinoV3Backbone: unable to switch attention implementation to 'eager': {exc}"
                )

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        if output_attentions:
            self._ensure_eager_attn()
        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
        )
        last_hidden = outputs.last_hidden_state  # [B, 1 + R + N, D]
        cls_token = last_hidden[:, 0, :]
        reg_tokens = last_hidden[:, 1 : 1 + self.num_register_tokens, :]
        patch_tokens_flat = last_hidden[:, 1 + self.num_register_tokens :, :]

        B, _, H, W = pixel_values.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        patch_tokens = patch_tokens_flat.view(B, h_p, w_p, -1)

        attn_grid = None
        if output_attentions and outputs.attentions is not None:
            attn_last = outputs.attentions[-1]
            cls_attn = attn_last[:, :, 0, :]  # [B, heads, num_tokens]
            start_idx = 1 + self.num_register_tokens
            end_idx = start_idx + patch_tokens_flat.size(1)
            cls_to_patches = cls_attn[:, :, start_idx:end_idx]
            attn_weights = cls_to_patches.mean(dim=1)  # [B, N_patches]
            attn_grid = attn_weights.view(B, h_p, w_p)

        return {
            "cls": cls_token,
            "patch_tokens": patch_tokens,
            "patch_tokens_flat": patch_tokens_flat,
            "reg_tokens": reg_tokens,
            "attn_cls_to_patches": attn_grid,
            "attentions": outputs.attentions if output_attentions else None,
        }

    def get_features_from_images(self, images: torch.Tensor, output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        encoded = self.preprocess(images)
        return self.forward(encoded["pixel_values"], output_attentions=output_attentions)


class CEDFeatureAggregator(nn.Module):
    """Fuse CLS tokens with pooled local descriptors to mimic CED embeddings."""

    def __init__(self, dim: int, gem_p: float = 3.0, use_proj: bool = True):
        super().__init__()
        self.dim = dim
        self.gem_p = gem_p
        self.use_proj = use_proj

        if use_proj:
            self.proj_cls = nn.Linear(dim, dim)
            self.proj_loc = nn.Linear(dim, dim)
        else:
            self.proj_cls = nn.Identity()
            self.proj_loc = nn.Identity()

    @staticmethod
    def gem(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
        # Force stable fp32 math and avoid huge values in pow
        x = x.float()
        x = x.clamp(min=eps)
        x = x.pow(p)
        x = x.mean(dim=1)
        return x.pow(1.0 / p)

    def compute_local_embedding(
        self,
        patch_tokens_flat: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weighted_tokens = patch_tokens_flat.float()
        if attn_weights is not None:
            weights = attn_weights.view(attn_weights.size(0), -1).float()
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            weights = weights.unsqueeze(-1)
            weighted_tokens = weighted_tokens * (weights + 1e-6)

        local = self.gem(weighted_tokens, p=self.gem_p)
        local = self.proj_loc(local)
        return F.normalize(local, dim=-1)

    def forward(
        self,
        cls: torch.Tensor,
        patch_tokens_flat: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
        return_local: bool = False,
    ) -> torch.Tensor:
        cls_global = self.proj_cls(cls.float())
        cls_global = F.normalize(cls_global, dim=-1)
        local = self.compute_local_embedding(patch_tokens_flat, attn_weights=attn_weights)
        descriptor = torch.cat([cls_global, local], dim=-1)
        descriptor = F.normalize(descriptor, dim=-1)

        if return_local:
            return descriptor, local
        return descriptor


class TransformerBlock(nn.Module):
    """Minimal Transformer encoder block for token refinement."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class CopyEditClassifier(nn.Module):
    """Cross-attention head that predicts copy-edit likelihoods for query/reference pairs."""

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, q_tokens: torch.Tensor, r_tokens: torch.Tensor) -> torch.Tensor:
        q_norm = q_tokens
        r_norm = r_tokens
        cross_out, _ = self.cross_attn(q_norm, r_norm, r_norm)
        x = q_tokens + cross_out
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits


# ======================================================================
# Adaptive Patch Transformer (APT)
# ======================================================================
class AdaptivePatchTransformer(nn.Module):
    """
    Lightweight, DINO-compatible variant of Adaptive Patch Transformer (APT).

    It does content-aware patch mixing on ViT patch tokens:
      - low-variance regions are replaced by a shared 2x2 block embedding
      - high-variance regions keep their original token

    This approximates adaptive patch sizes without changing the DINO patch
    embedding layer or sequence length.
    """

    def __init__(self, dim: int, gate_scale: float = 3.0, use_attn: bool = True):
        super().__init__()
        # How strongly we push towards block-averaging in smooth regions
        self.gate_scale = nn.Parameter(torch.tensor(gate_scale, dtype=torch.float32))
        self.use_attn = use_attn

        # Small refinement MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        patch_tokens_flat: torch.Tensor,        # [B, N, D]
        attn_grid: Optional[torch.Tensor] = None,  # [B, H, W] or None
    ) -> torch.Tensor:
        B, N, D = patch_tokens_flat.shape

        # Infer spatial grid from attention map if available; otherwise assume square
        if attn_grid is not None:
            H, W = attn_grid.shape[1], attn_grid.shape[2]
        else:
            H = W = int(N ** 0.5)
            assert H * W == N, (
                "APT expects patch tokens to form a square grid if attn_grid is not provided."
            )

        x = patch_tokens_flat.view(B, H, W, D)  # [B, H, W, D]

        # Token-level "complexity" via embedding variance
        var_map = x.var(dim=-1, unbiased=False)  # [B, H, W]

        if self.use_attn and attn_grid is not None:
            # Emphasize tokens that the backbone already thinks are important
            var_map = var_map * (attn_grid.float() + 1e-6)

        # Normalise per image
        mean = var_map.mean(dim=(1, 2), keepdim=True)
        std = var_map.std(dim=(1, 2), keepdim=True) + 1e-6
        norm_score = (var_map - mean) / std

        # Gate in (0, 1): high for complex regions, low for smooth regions
        gate = torch.sigmoid(self.gate_scale * norm_score)  # [B, H, W]
        gate = gate.unsqueeze(-1)  # [B, H, W, 1]

        # 2x2 block average to mimic "larger patches" in smooth regions
        assert H % 2 == 0 and W % 2 == 0, "APT block mixing assumes an even patch grid (e.g., 14x14)."

        # [B, H//2, 2, W//2, 2, D]
        x_blocks = x.view(B, H // 2, 2, W // 2, 2, D)
        # block_avg: [B, H//2, 1, W//2, 1, D]
        block_avg = x_blocks.mean(dim=(2, 4), keepdim=True)
        # Broadcast back to [B, H, W, D]
        block_avg = block_avg.expand(-1, -1, 2, -1, 2, -1).reshape(B, H, W, D)

        # Content-aware mixing: complex regions ≈ original, smooth regions ≈ block-avg
        mixed = gate * x + (1.0 - gate) * block_avg  # [B, H, W, D]
        mixed = mixed.view(B, N, D)

        # Small residual refinement
        mixed = mixed + self.mlp(mixed)
        return mixed


class CEDModel(nn.Module):
    """High-level wrapper orchestrating backbone encoding and pair scoring."""

    def __init__(self, backbone: DinoV3Backbone, dim: int, use_apt: bool = True):
        super().__init__()
        self.backbone = backbone

        # --------- Adaptive Patch Transformer (APT) -----------
        self.use_apt = use_apt
        if use_apt:
            self.apt = AdaptivePatchTransformer(dim=dim)
            self.apt = self.apt.to(device=device, dtype=torch.float32)
        else:
            self.apt = None
        # ------------------------------------------------------

        self.aggregator = CEDFeatureAggregator(dim=dim, gem_p=3.0, use_proj=True)
        self.aggregator = self.aggregator.to(device=device, dtype=torch.float32)
        self.classifier = CopyEditClassifier(dim=dim)
        self.classifier = self.classifier.to(device=device, dtype=torch.float32)

    def encode_images(
        self,
        images: torch.Tensor,
        return_local: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Run backbone + aggregator in full precision to avoid fp16 overflow/NANs
        with amp.autocast(device_type="cuda", enabled=False):
            feats = self.backbone.get_features_from_images(images, output_attentions=True)

            # --------- APT on patch tokens --------------------
            if self.apt is not None:
                feats["patch_tokens_flat"] = self.apt(
                    feats["patch_tokens_flat"],
                    attn_grid=feats.get("attn_cls_to_patches", None),
                )
            # ---------------------------------------------------

            agg_out = self.aggregator(
                cls=feats["cls"],
                patch_tokens_flat=feats["patch_tokens_flat"],
                attn_weights=feats.get("attn_cls_to_patches"),
                return_local=return_local,
            )
        if return_local:
            descriptor, local = agg_out
            feats["local_descriptor"] = local
        else:
            descriptor = agg_out
        return descriptor, feats

    def score_pair(
        self,
        q_images: torch.Tensor,
        r_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Also keep classification backbone path stable
        with amp.autocast(device_type="cuda", enabled=False):
            # request attentions so APT can use them too
            q_feats = self.backbone.get_features_from_images(q_images, output_attentions=True)
            r_feats = self.backbone.get_features_from_images(r_images, output_attentions=True)

            # --------- APT on patch tokens --------------------
            if self.apt is not None:
                q_feats["patch_tokens_flat"] = self.apt(
                    q_feats["patch_tokens_flat"],
                    attn_grid=q_feats.get("attn_cls_to_patches", None),
                )
                r_feats["patch_tokens_flat"] = self.apt(
                    r_feats["patch_tokens_flat"],
                    attn_grid=r_feats.get("attn_cls_to_patches", None),
                )
            # ---------------------------------------------------

            q_tokens = q_feats["patch_tokens_flat"]
            r_tokens = r_feats["patch_tokens_flat"]
            logits = self.classifier(q_tokens, r_tokens).squeeze(-1)
        return logits, q_feats, r_feats


# ======================================================================
# Losses, augmentations (with FP32 + NaN guards)
# ======================================================================
copy_edit_aug = V.Compose(
    [
        V.RandomResizedCrop(224, scale=(0.6, 1.0)),  # img_size will be reset in main()
        V.RandomHorizontalFlip(),
        V.ColorJitter(0.2, 0.2, 0.2, 0.1),
        V.RandomGrayscale(p=0.1),
    ]
)


def make_positive_negative_pairs(batch_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = batch_imgs.size(0)
    imgs_cpu = batch_imgs.detach().cpu()
    x_anchor = batch_imgs.to(device, non_blocking=True)
    x_pos = torch.stack([copy_edit_aug(img) for img in imgs_cpu]).to(device, non_blocking=True)
    perm = torch.randperm(B)
    x_neg = torch.stack([copy_edit_aug(imgs_cpu[i]) for i in perm]).to(device, non_blocking=True)
    return x_anchor, x_pos, x_neg


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent (SimCLR) loss with safe float32 math.
    """
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)

    reps = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = reps @ reps.t() / temperature  # [2B, 2B]

    diag_mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(diag_mask, -1e4)

    batch_size = z1.size(0)
    targets = torch.arange(batch_size, 2 * batch_size, device=sim.device)
    targets = torch.cat([targets, torch.arange(0, batch_size, device=sim.device)], dim=0)

    loss = F.cross_entropy(sim, targets)
    return loss


def similarity_kl_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Symmetric KL between similarity distributions, with stable log_softmax.
    """
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)

    sim1 = z1 @ z1.t() / temperature
    sim2 = z2 @ z2.t() / temperature

    log_p1 = F.log_softmax(sim1, dim=-1)
    log_p2 = F.log_softmax(sim2, dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()

    kl1 = F.kl_div(log_p1, p2, reduction="batchmean", log_target=False)
    kl2 = F.kl_div(log_p2, p1, reduction="batchmean", log_target=False)

    return 0.5 * (kl1 + kl2)


def multi_similarity_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Multi-similarity loss in float32.
    """
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device)

    emb = F.normalize(embeddings.float(), dim=-1)
    sims = emb @ emb.t()
    loss = torch.zeros(1, device=embeddings.device, dtype=torch.float32)
    valid = 0

    for i in range(emb.size(0)):
        pos_mask = (labels == labels[i]).clone()
        pos_mask[i] = False
        neg_mask = labels != labels[i]

        pos_sims = sims[i][pos_mask]
        neg_sims = sims[i][neg_mask]

        if pos_sims.numel() == 0 or neg_sims.numel() == 0:
            continue

        pos_term = (1.0 / alpha) * torch.log1p(
            torch.sum(torch.exp(torch.clamp(-alpha * (pos_sims - (1 - margin)), max=50.0)))
        )
        neg_term = (1.0 / beta) * torch.log1p(
            torch.sum(torch.exp(torch.clamp(beta * (neg_sims - margin), max=50.0)))
        )

        loss = loss + pos_term + neg_term
        valid += 1

    if valid == 0:
        return torch.tensor(0.0, device=embeddings.device)

    return loss / valid


def asl_loss(
    v_former: torch.Tensor,
    v_latter: torch.Tensor,
    lambda_mtr: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    ASL loss: norm ratio + metric term (NT-Xent).
    """
    vf = v_former.float()
    vl = v_latter.float()

    norm_f = vf.norm(p=2, dim=-1)
    norm_l = vl.norm(p=2, dim=-1)
    ratio = (norm_l + eps) / (norm_f + eps)

    ratio = torch.clamp(ratio, 0.1, 10.0)
    loss_ratio = torch.exp(1.0 - ratio).mean()

    loss_metric = nt_xent_loss(vf, vl, temperature=0.2)
    return loss_ratio + lambda_mtr * loss_metric


bce_loss = nn.BCEWithLogitsLoss()


# ======================================================================
# Checkpoint helpers (with training_state)
# ======================================================================
def save_checkpoint(
    model: CEDModel,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int,
    training_state: Dict,
):
    if rank != 0:
        return
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": training_state,
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"[Rank 0] Saved checkpoint to {path}")


def load_checkpoint_if_available(
    model: CEDModel,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    rank: int,
    training_state: Dict,
) -> Dict:
    """
    If checkpoint exists, load model, optimizer, and training_state.
    NOTE: if you trained previously *without* APT, you may need to delete
    the old checkpoint because state_dict keys have changed.
    """
    if not os.path.exists(path):
        if rank == 0:
            print(f"[Checkpoint] No existing checkpoint at {path}, starting fresh.")
        return training_state

    map_location = device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)  # strict=False for safety
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    loaded_state = ckpt.get("state", {})
    training_state.update(loaded_state)

    if rank == 0:
        print(f"[Checkpoint] Loaded checkpoint from {path}")
        print(
            f"[Checkpoint] Resuming from DISC21 epoch {training_state['epoch_disc21']} "
            f"and NDEC epoch {training_state['epoch_ndec']}"
        )
    return training_state


# ======================================================================
# Ground-truth helpers for DISC21 + NDEC
# ======================================================================
def load_disc21_groundtruth_map(split: str, root: Path) -> Dict[str, List[str]]:
    df = load_groundtruth(split=split, root=root)
    gt: Dict[str, List[str]] = {}
    for row in df.itertuples():
        qid = str(getattr(row, "query_id"))
        rid = str(getattr(row, "reference_id"))
        if rid and rid != "nan":
            gt.setdefault(qid, []).append(rid)
    return gt


def load_ndec_groundtruth_map(
    root: Path,
    csv_name: str = "public_ground_truth_h5.csv",
    drop_missing: bool = True,
) -> Dict[str, List[str]]:
    df = load_ndec_groundtruth(root=root, csv_name=csv_name, drop_missing=drop_missing)
    gt: Dict[str, List[str]] = {}
    for row in df.itertuples():
        qid = str(getattr(row, "query_id"))
        rid = str(getattr(row, "reference_id"))
        gt.setdefault(qid, []).append(rid)
    return gt


# ======================================================================
# Main training + evaluation (DDP) with resume
# ======================================================================
def main():
    global device

    # ------------------ basic setup ------------------
    script_dir = Path(__file__).resolve().parent

    env_path = script_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    cudnn.benchmark = True

    # AMP scaler: handle both legacy (device) and newer (device_type) signatures
    scaler_kwargs = {"enabled": True}
    scaler_params = inspect.signature(amp.GradScaler).parameters
    if "device_type" in scaler_params:
        scaler_kwargs["device_type"] = "cuda"
    elif "device" in scaler_params:
        scaler_kwargs["device"] = "cuda"
    scaler = amp.GradScaler(**scaler_kwargs)

    cfg = ExperimentConfig()
    if rank == 0:
        print("Config:", cfg)
        print(f"World size: {world_size} | Rank: {rank} | Local rank: {local_rank}")
        sys.stdout.flush()

    # Make checkpoint path relative to script dir
    checkpoint_path = str((script_dir / cfg.checkpoint_path).resolve())

    # Update augment crop size based on config
    global copy_edit_aug
    copy_edit_aug = V.Compose(
        [
            V.RandomResizedCrop(cfg.img_size_train, scale=(0.6, 1.0)),
            V.RandomHorizontalFlip(),
            V.ColorJitter(0.2, 0.2, 0.2, 0.1),
            V.RandomGrayscale(p=0.1),
        ]
    )

    # ------------------ datasets + loaders ------------------
    disc_cfg = Disc21DataConfig(
        root=cfg.disc21_root,
        img_size_train=cfg.img_size_train,
        img_size_eval=cfg.img_size_eval,
        batch_size_train=cfg.batch_size_train,
        batch_size_eval=cfg.batch_size_eval,
        num_workers=cfg.num_workers,
    )

    train_tfms, eval_tfms = build_transforms(
        img_size_train=cfg.img_size_train,
        img_size_eval=cfg.img_size_eval,
    )

    # DISC21 training dataset
    train_ds = get_train_dataset(root=disc_cfg.root, transform=train_tfms)

    # DDP: distributed sampler for training
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    # Dynamically choose num_workers for loaders
    cpu_count = os.cpu_count() or 8
    num_workers = min(16, max(4, cpu_count // max(world_size, 1)))

    if rank == 0:
        print(f"[DataLoader] Using num_workers={num_workers}, prefetch_factor=2, persistent_workers=True")
        sys.stdout.flush()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # DISC21 eval datasets (used only on rank 0)
    if rank == 0:
        ref_ds = get_reference_dataset(root=disc_cfg.root, transform=eval_tfms)
        dev_queries_ds = get_query_dataset("dev", root=disc_cfg.root, transform=eval_tfms)
        test_queries_ds = get_query_dataset("test", root=disc_cfg.root, transform=eval_tfms)

        ref_loader = DataLoader(
            ref_ds,
            batch_size=cfg.batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        dev_query_loader = DataLoader(
            dev_queries_ds,
            batch_size=cfg.batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        test_query_loader = DataLoader(
            test_queries_ds,
            batch_size=cfg.batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        ref_loader = dev_query_loader = test_query_loader = None  # type: ignore

    # ---------- NDEC config + loaders ----------
    ndec_cfg = NdecDataConfig()
    if hasattr(ndec_cfg, "root"):
        ndec_cfg.root = cfg.ndec_root
    if hasattr(ndec_cfg, "img_size_train"):
        ndec_cfg.img_size_train = cfg.img_size_train
    if hasattr(ndec_cfg, "img_size_eval"):
        ndec_cfg.img_size_eval = cfg.img_size_eval
    if hasattr(ndec_cfg, "batch_size_pairs"):
        ndec_cfg.batch_size_pairs = cfg.batch_size_pairs
    if hasattr(ndec_cfg, "batch_size_eval"):
        ndec_cfg.batch_size_eval = cfg.batch_size_eval
    if hasattr(ndec_cfg, "num_workers"):
        ndec_cfg.num_workers = cfg.num_workers

    ndec_query_loader, ndec_ref_loader, ndec_pos_pair_loader, ndec_neg_pair_loader = build_ndec_loaders(
        ndec_cfg
    )

    ndec_neg_dataset = ndec_neg_pair_loader.dataset
    ndec_neg_sampler = DistributedSampler(
        ndec_neg_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    ndec_neg_train_loader = DataLoader(
        ndec_neg_dataset,
        batch_size=ndec_neg_pair_loader.batch_size,
        sampler=ndec_neg_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    if rank == 0:
        print(f"Train images: {len(train_ds):,}")
        print(f"NDEC negative pairs batches (per rank): {len(ndec_neg_train_loader):,}")
        sys.stdout.flush()

    # ------------------ model + optimizer (wrapped in DDP) ------------------
    backbone = DinoV3Backbone(model_name=cfg.model_name)
    ced_model = CEDModel(backbone=backbone, dim=backbone.hidden_size, use_apt=True).to(device)

    ddp_model = DDP(
        ced_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": ddp_model.module.backbone.parameters(), "lr": cfg.lr_backbone},
            {
                "params": list(ddp_model.module.aggregator.parameters())
                + list(ddp_model.module.classifier.parameters())
                + (list(ddp_model.module.apt.parameters()) if ddp_model.module.apt is not None else []),
                "lr": cfg.lr_head,
            },
        ],
        weight_decay=cfg.weight_decay,
    )

    # ------------------ training state (for resume) ------------------
    training_state = {
        "epoch_disc21": 0,
        "epoch_ndec": 0,
        "best_disc21_loss": float("inf"),
        "best_ndec_loss": float("inf"),
        "disc21_bad_epochs": 0,
        "ndec_bad_epochs": 0,
        "disc21_losses": [],
        "ndec_losses": [],
    }

    training_state = load_checkpoint_if_available(
        ddp_model.module, optimizer, checkpoint_path, rank, training_state
    )

    start_epoch_disc21 = training_state["epoch_disc21"]
    start_epoch_ndec = training_state["epoch_ndec"]
    best_disc21_loss = training_state["best_disc21_loss"]
    best_ndec_loss = training_state["best_ndec_loss"]
    disc21_bad_epochs = training_state["disc21_bad_epochs"]
    ndec_bad_epochs = training_state["ndec_bad_epochs"]
    disc21_losses: List[float] = training_state["disc21_losses"]
    ndec_losses: List[float] = training_state["ndec_losses"]

    # ------------------ training flags ------------------
    should_train_disc21 = True
    should_train_ndec = True
    did_train = False

    if start_epoch_disc21 >= cfg.num_epochs_disc21:
        should_train_disc21 = False
        if rank == 0:
            print(f"[DISC21] Already completed {start_epoch_disc21} epochs; skipping DISC21 training.")
    if start_epoch_ndec >= cfg.num_epochs_ndec:
        should_train_ndec = False
        if rank == 0:
            print(f"[NDEC] Already completed {start_epoch_ndec} epochs; skipping NDEC training.")

    # ------------------ DISC21 training loop (DDP) ------------------
    if should_train_disc21:
        did_train = True

        if rank == 0 and start_epoch_disc21 > 0:
            print(f"[DISC21] Resuming from epoch {start_epoch_disc21 + 1}")
            sys.stdout.flush()

        for epoch in range(start_epoch_disc21, cfg.num_epochs_disc21):
            ddp_model.train()
            train_sampler.set_epoch(epoch)
            current_epoch = epoch + 1

            if rank == 0:
                progress = tqdm(train_loader, desc=f"[DISC21][Epoch {current_epoch}/{cfg.num_epochs_disc21}]")
            else:
                progress = train_loader

            epoch_loss = 0.0
            epoch_correct = 0.0
            epoch_total = 0.0

            for imgs, _ in progress:
                imgs = imgs.to(device, non_blocking=True)
                x_anchor, x_pos, x_neg = make_positive_negative_pairs(imgs)

                optimizer.zero_grad(set_to_none=True)

                # forward + loss with AMP autocast
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    # encode anchor & positive (internally forced to fp32)
                    v_anchor, feats_anchor = ddp_model.module.encode_images(x_anchor, return_local=True)
                    v_pos, feats_pos = ddp_model.module.encode_images(x_pos, return_local=True)

                    # OPTIONAL: temporary sanity check
                    if not torch.isfinite(v_anchor).all() or not torch.isfinite(v_pos).all():
                        if rank == 0:
                            print(
                                "[SANITY] Backbone/APT output has non-finite values. "
                                f"v_anchor finite={torch.isfinite(v_anchor).all().item()}, "
                                f"v_pos finite={torch.isfinite(v_pos).all().item()}"
                            )
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    loss_contrast = nt_xent_loss(v_anchor, v_pos, temperature=cfg.temperature_ntxent)
                    loss_kl = similarity_kl_loss(v_anchor, v_pos, temperature=cfg.temperature_kl)

                    local_anchor = feats_anchor.get("local_descriptor")
                    local_pos = feats_pos.get("local_descriptor")
                    local_embeddings = torch.cat([local_anchor, local_pos], dim=0)
                    batch_ids = torch.arange(local_anchor.size(0), device=local_anchor.device)
                    local_labels = torch.cat([batch_ids, batch_ids], dim=0)
                    loss_local = multi_similarity_loss(local_embeddings, local_labels)

                    anchor_tokens = feats_anchor["patch_tokens_flat"].detach()
                    pos_tokens = feats_pos["patch_tokens_flat"].detach()

                    with amp.autocast(device_type="cuda", enabled=False):
                        with torch.no_grad():
                            neg_feats = ddp_model.module.backbone.get_features_from_images(x_neg, output_attentions=True)
                            # Apply APT to negative tokens to keep classifier consistent
                            if ddp_model.module.apt is not None:
                                neg_feats["patch_tokens_flat"] = ddp_model.module.apt(
                                    neg_feats["patch_tokens_flat"],
                                    attn_grid=neg_feats.get("attn_cls_to_patches", None),
                                )
                        neg_tokens = neg_feats["patch_tokens_flat"].detach()
                        logits_pos = ddp_model.module.classifier(anchor_tokens, pos_tokens).squeeze(-1)
                        logits_neg = ddp_model.module.classifier(anchor_tokens, neg_tokens).squeeze(-1)

                    labels_pos = torch.ones_like(logits_pos)
                    labels_neg = torch.zeros_like(logits_neg)
                    logits = torch.cat([logits_pos, logits_neg], dim=0)
                    labels = torch.cat([labels_pos, labels_neg], dim=0)

                    loss_bce = bce_loss(logits.float(), labels.float())

                    loss = (
                        loss_contrast
                        + cfg.lambda_kl * loss_kl
                        + cfg.lambda_local * loss_local
                        + cfg.lambda_bce * loss_bce
                    )

                # NaN / inf guard with debug
                if not torch.isfinite(loss):
                    if rank == 0:
                        def finite(t):
                            return bool(torch.isfinite(t).all().item())

                        print("[WARN][DISC21] Non-finite loss encountered; skipping batch.")
                        print(
                            "  components finite?:",
                            f"contrast={finite(loss_contrast)},",
                            f"kl={finite(loss_kl)},",
                            f"local={finite(loss_local)},",
                            f"bce={finite(loss_bce)}",
                        )
                        try:
                            print("  loss_contrast:", float(loss_contrast.detach().cpu()))
                            print("  loss_kl      :", float(loss_kl.detach().cpu()))
                            print("  loss_local   :", float(loss_local.detach().cpu()))
                            print("  loss_bce     :", float(loss_bce.detach().cpu()))
                        except Exception as e:
                            print("  [DEBUG] error printing loss components:", e)

                        for name, t in [
                            ("v_anchor", v_anchor),
                            ("v_pos", v_pos),
                            ("local_anchor", local_anchor),
                            ("local_pos", local_pos),
                            ("logits_pos", logits_pos),
                            ("logits_neg", logits_neg),
                        ]:
                            if not torch.isfinite(t).all():
                                finite_vals = t[torch.isfinite(t)]
                                if finite_vals.numel() > 0:
                                    t_min = float(finite_vals.min().detach().cpu())
                                    t_max = float(finite_vals.max().detach().cpu())
                                else:
                                    t_min = float("nan")
                                    t_max = float("nan")
                                print(f"  [DEBUG] {name} has non-finite values. min={t_min}, max={t_max}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Backward with AMP
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += float(loss.detach().cpu())

                with torch.no_grad():
                    preds = (torch.sigmoid(logits.float()) >= 0.5).float()
                    correct = (preds == labels).sum().item()
                    total = labels.numel()
                    epoch_correct += correct
                    epoch_total += total

                if rank == 0:
                    batch_acc = correct / max(total, 1)
                    progress.set_postfix(
                        {
                            "loss": f"{float(loss.detach().cpu()):.4f}",
                            "contrast": f"{loss_contrast.item():.4f}",
                            "local": f"{loss_local.item():.4f}",
                            "bce_acc": f"{batch_acc:.3f}",
                        }
                    )

            # Average loss across processes
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss_avg = epoch_loss_tensor.item() / world_size / max(len(train_loader), 1)

            correct_tensor = torch.tensor(epoch_correct, device=device)
            total_tensor = torch.tensor(epoch_total, device=device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            epoch_acc_global = (
                (correct_tensor.item() / max(total_tensor.item(), 1.0)) if total_tensor.item() > 0 else 0.0
            )

            stop_now = False
            if rank == 0:
                disc21_losses.append(epoch_loss_avg)
                print(
                    f"[DISC21][Epoch {current_epoch}] avg loss across ranks = {epoch_loss_avg:.4f}, "
                    f"BCE accuracy = {epoch_acc_global:.4f}"
                )
                sys.stdout.flush()

                improved = epoch_loss_avg + cfg.early_stopping_min_delta_disc21 < best_disc21_loss
                if improved:
                    best_disc21_loss = epoch_loss_avg
                    disc21_bad_epochs = 0
                else:
                    disc21_bad_epochs += 1

                training_state["epoch_disc21"] = current_epoch
                training_state["best_disc21_loss"] = best_disc21_loss
                training_state["disc21_bad_epochs"] = disc21_bad_epochs
                training_state["disc21_losses"] = disc21_losses

                save_checkpoint(ddp_model.module, optimizer, checkpoint_path, rank=0, training_state=training_state)

                can_stop_disc21 = current_epoch >= cfg.min_epochs_disc21
                if not improved and disc21_bad_epochs >= cfg.early_stopping_patience_disc21 and can_stop_disc21:
                    print(
                        f"[DISC21] Early stopping after {cfg.early_stopping_patience_disc21} bad epoch(s)."
                    )
                    stop_now = True
                elif not improved and disc21_bad_epochs >= cfg.early_stopping_patience_disc21:
                    remaining = cfg.min_epochs_disc21 - current_epoch
                    print(
                        f"[DISC21] Patience exhausted but continuing {remaining} more epoch(s) "
                        f"to satisfy min_epochs_disc21."
                    )

            stop_tensor = torch.tensor(1 if stop_now else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break

    # ------------------ NDEC ASL fine-tuning (DDP on all ranks) ------------------
    if should_train_ndec:
        did_train = True

        if rank == 0:
            if start_epoch_ndec > 0:
                print(f"[NDEC] Resuming ASL fine-tuning from epoch {start_epoch_ndec + 1}.")
            else:
                print("[NDEC] Starting ASL fine-tuning with DDP on all ranks.")
            sys.stdout.flush()

        for epoch in range(start_epoch_ndec, cfg.num_epochs_ndec):
            ddp_model.train()
            ndec_neg_sampler.set_epoch(epoch)
            current_epoch = epoch + 1

            if rank == 0:
                progress = tqdm(
                    ndec_neg_train_loader,
                    desc=f"[NDEC][Epoch {current_epoch}/{cfg.num_epochs_ndec}]",
                )
            else:
                progress = ndec_neg_train_loader

            total_loss = 0.0

            for img_a, img_b, _, _ in progress:
                img_a = img_a.to(device, non_blocking=True)
                img_b = img_b.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    v_former, _ = ddp_model.module.encode_images(img_a)
                    v_latter, _ = ddp_model.module.encode_images(img_b)

                    loss_asl = asl_loss(v_former=v_former, v_latter=v_latter, lambda_mtr=cfg.lambda_asl_mtr)

                if not torch.isfinite(loss_asl):
                    if rank == 0:
                        print("[WARN][NDEC] Non-finite ASL loss encountered; skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss_asl).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += float(loss_asl.detach().cpu())

                if rank == 0:
                    progress.set_postfix({"asl_loss": f"{float(loss_asl.detach().cpu()):.4f}"})

            loss_tensor = torch.tensor(total_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_asl_global = loss_tensor.item() / world_size / max(len(ndec_neg_train_loader), 1)

            stop_now_ndec = False
            if rank == 0:
                ndec_losses.append(avg_asl_global)
                print(f"[NDEC][Epoch {current_epoch}] avg ASL loss (global) = {avg_asl_global:.4f}")
                sys.stdout.flush()

                improved = avg_asl_global + cfg.early_stopping_min_delta_ndec < best_ndec_loss
                if improved:
                    best_ndec_loss = avg_asl_global
                    ndec_bad_epochs = 0
                else:
                    ndec_bad_epochs += 1

                training_state["epoch_ndec"] = current_epoch
                training_state["best_ndec_loss"] = best_ndec_loss
                training_state["ndec_bad_epochs"] = ndec_bad_epochs
                training_state["ndec_losses"] = ndec_losses

                save_checkpoint(ddp_model.module, optimizer, checkpoint_path, rank=0, training_state=training_state)

                can_stop_ndec = current_epoch >= cfg.min_epochs_ndec
                if not improved and ndec_bad_epochs >= cfg.early_stopping_patience_ndec and can_stop_ndec:
                    print(
                        f"[NDEC] Early stopping after {cfg.early_stopping_patience_ndec} bad epoch(s)."
                    )
                    stop_now_ndec = True
                elif not improved and ndec_bad_epochs >= cfg.early_stopping_patience_ndec:
                    remaining = cfg.min_epochs_ndec - current_epoch
                    print(
                        f"[NDEC] Patience exhausted but continuing {remaining} more epoch(s) "
                        f"to satisfy min_epochs_ndec."
                    )

            stop_tensor_ndec = torch.tensor(1 if stop_now_ndec else 0, device=device)
            dist.broadcast(stop_tensor_ndec, src=0)
            if stop_tensor_ndec.item() == 1:
                break

    # ------------------ Dump training curves to JSON (rank 0) ------------------
    if rank == 0:
        log_dir = (script_dir / "artifacts/logs_apt").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        curves_path = log_dir / "training_curves_ced_apt.json"
        with curves_path.open("w") as f:
            json.dump(
                {
                    "disc21_losses": disc21_losses,
                    "ndec_losses": ndec_losses,
                },
                f,
                indent=2,
            )
        print(f"[Logs] Wrote training curves to {curves_path}")

    # ------------------ Evaluation using eval_utils (rank 0 only) ------------------
    if rank == 0:
        print("\n[Eval] Starting DISC21 + NDEC evaluation with eval_utils metrics (µAP, R@P90).")
        sys.stdout.flush()

        eval_model = ddp_model.module  # unwrap CEDModel

        encoding_cache_root = (script_dir / cfg.encoding_cache_root).resolve()
        encoding_cache_root.mkdir(parents=True, exist_ok=True)

        def encode_with_cache(loader: DataLoader, relative_artifact: str):
            artifact_path = (script_dir / relative_artifact).resolve()
            return compute_descriptors_for_loader(
                loader,
                eval_model,
                str(artifact_path),
                checkpoint_root=str(encoding_cache_root),
                resume=True,
                chunk_size=cfg.encoding_chunk_size,
            )

        disc_ref_vecs, disc_ref_ids = encode_with_cache(ref_loader, "artifacts/disc21_ref")

        for split_name, q_loader in [("dev", dev_query_loader), ("test", test_query_loader)]:
            print(f"\n[Eval][DISC21-{split_name}] Encoding queries + computing retrieval metrics...")
            sys.stdout.flush()

            q_vecs, q_ids = encode_with_cache(q_loader, f"artifacts/disc21_{split_name}_query")
            gt_map = load_disc21_groundtruth_map(split=split_name, root=disc_cfg.root)

            metrics = evaluate_retrieval(
                query_vecs=q_vecs,
                query_ids=q_ids,
                ref_vecs=disc_ref_vecs,
                ref_ids=disc_ref_ids,
                gt_map=gt_map,
                topk_list=[1, 5, 10, 20],
                device=device,
                query_chunk_size=cfg.eval_query_chunk_size,
                ref_chunk_size=cfg.eval_ref_chunk_size,
                max_global_pairs_per_query=cfg.eval_global_pairs_per_query,
            )
            print(f"[Eval][DISC21-{split_name}] metrics:", metrics)
            sys.stdout.flush()

        print("\n[Eval][NDEC] Encoding references + queries...")
        sys.stdout.flush()

        ndec_ref_vecs, ndec_ref_ids = encode_with_cache(ndec_ref_loader, "artifacts/ndec_ref")
        ndec_query_vecs, ndec_query_ids = encode_with_cache(ndec_query_loader, "artifacts/ndec_query")
        ndec_gt_map = load_ndec_groundtruth_map(cfg.ndec_root, drop_missing=True)

        ndec_metrics = evaluate_retrieval(
            query_vecs=ndec_query_vecs,
            query_ids=ndec_query_ids,
            ref_vecs=ndec_ref_vecs,
            ref_ids=ndec_ref_ids,
            gt_map=ndec_gt_map,
            topk_list=[1, 5, 10, 20],
            device=device,
            query_chunk_size=cfg.eval_query_chunk_size,
            ref_chunk_size=cfg.eval_ref_chunk_size,
            max_global_pairs_per_query=cfg.eval_global_pairs_per_query,
        )
        print("[Eval][NDEC] metrics:", ndec_metrics)
        sys.stdout.flush()

    if did_train and rank == 0:
        print(f"\n[Training complete] Final checkpoint at {checkpoint_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
