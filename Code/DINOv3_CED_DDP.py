import os
import sys
import json
import warnings
import inspect
import gc
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

from ced_evaluation import (
    compute_descriptors_for_loader,
    evaluate_retrieval,
    cosine_similarity,
    compute_muap_and_rp90,
    benchmark_inference,
    ced_two_stage_eval,
    build_ref_index_map,
    normalize_id,
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

    # ==== CHANGED FOR CEDETECTOR ====
    # Train at 224, eval at 384 (paper: 224 → 384)
    img_size_train: int = 224
    img_size_eval: int = 384

    # Paper uses batch_size = 64, lr = 2e-5 for Adam.
    # You can lower batch_size if you hit OOM.
    batch_size_train: int = 32
    batch_size_eval: int = 32
    batch_size_pairs: int = 32

    num_workers: int = 4  # still used by Disc21DataConfig / NdecDataConfig

    # ==== CED AUGMENTATION CONFIG ====
    # Enable comprehensive CED augmentation pipeline (35 transforms from AugLy + AlbumentationsX)
    use_ced_augmentations: bool = True
    ced_min_ops: int = 2  # Minimum augmentation operations per image
    ced_max_ops: int = 6  # Maximum augmentation operations per image
    ced_aug_seed: Optional[int] = None  # Random seed for augmentations (None = random)

    # ==== CED EVALUATION CONFIG ====
    # Enable two-stage evaluation pipeline (patch extraction + classifier)
    use_ced_two_stage_eval: bool = True
    ced_k_candidates_per_patch: int = 10  # Top-k candidates per patch (paper uses 10)

    # Paper: Adam lr = 2e-5. We use same LR for backbone+heads.
    lr_backbone: float = 2e-5
    lr_head: float = 2e-5

    weight_decay: float = 1e-4

    # ==== CHANGED FOR CEDETECTOR ====
    # NT-Xent temperature τ = 0.025
    temperature_ntxent: float = 0.025

    # Kozachenko–Leonenko entropy weight λ = 0.5
    lambda_kl: float = 0.5

    # Multi-similarity loss hyperparams (α, β, γ=margin) = (2, 50, 1)
    lambda_local: float = 1.0
    lambda_bce: float = 1.0
    lambda_asl_mtr: float = 1.0

    # Epochs (paper uses 30 on ISC)
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

    checkpoint_path: Path = Path("artifacts/checkpoints/ced_model_ddp.pt")
    encoding_cache_root: Path = Path("artifacts/sinov3_ced_encoding")
    encoding_chunk_size: int = 1024
    eval_query_chunk_size: int = 256
    eval_ref_chunk_size: int = 8192

    # top-N candidate pairs per query for global metrics
    eval_global_pairs_per_query: int = 512



# ======================================================================
# Backbone, aggregator, classifier, CEDModel
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

    # ==== NEW: return cls_global, local, descriptor ====
    def forward_components(
        self,
        cls: torch.Tensor,
        patch_tokens_flat: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Global (CLS) embedding
        cls_global = self.proj_cls(cls.float())
        cls_global = F.normalize(cls_global, dim=-1)

        # Local pooled embedding
        local = self.compute_local_embedding(patch_tokens_flat, attn_weights=attn_weights)

        # Concatenate global + local (paper uses both for matching)
        descriptor = torch.cat([cls_global, local], dim=-1)
        descriptor = F.normalize(descriptor, dim=-1)
        return cls_global, local, descriptor

    def forward(
        self,
        cls: torch.Tensor,
        patch_tokens_flat: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
        return_local: bool = False,
    ) -> torch.Tensor:
        cls_global, local, descriptor = self.forward_components(
            cls=cls,
            patch_tokens_flat=patch_tokens_flat,
            attn_weights=attn_weights,
        )
        if return_local:
            # kept for backwards compatibility if you ever use it directly
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


class CEDModel(nn.Module):
    """High-level wrapper orchestrating backbone encoding and pair scoring."""

    def __init__(self, backbone: DinoV3Backbone, dim: int):
        super().__init__()
        self.backbone = backbone
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

            # ==== CHANGED: use forward_components to get cls, local, descriptor ====
            cls_global, local, descriptor = self.aggregator.forward_components(
                cls=feats["cls"],
                patch_tokens_flat=feats["patch_tokens_flat"],
                attn_weights=feats.get("attn_cls_to_patches"),
            )

        # stash for losses
        feats["cls_global"] = cls_global                   # z in the paper
        if return_local:
            feats["local_descriptor"] = local              # u in the paper

        return descriptor, feats

    def score_pair(
        self,
        q_images: torch.Tensor,
        r_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Classification path can still use patch tokens directly
        with amp.autocast(device_type="cuda", enabled=False):
            q_feats = self.backbone.get_features_from_images(q_images)
            r_feats = self.backbone.get_features_from_images(r_images)
            q_tokens = q_feats["patch_tokens_flat"]
            r_tokens = r_feats["patch_tokens_flat"]
            logits = self.classifier(q_tokens, r_tokens).squeeze(-1)
        return logits, q_feats, r_feats


# ======================================================================
# Losses (with FP32 + NaN guards)
# ======================================================================

# Note: On-the-fly augmentation for positive/negative pairs is now handled
# by a separate augmentation transform that will be applied in make_positive_negative_pairs.
# The comprehensive CED augmentation pipeline (35 transforms) is applied during
# dataset loading for the anchor images. For positive/negative pairs during training,
# we apply additional on-the-fly copy-edit style augmentations.

def make_positive_negative_pairs(
    batch_imgs: torch.Tensor,
    copy_edit_aug: Optional[V.Compose] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create positive and negative pairs from a batch of images.

    Args:
        batch_imgs: Input batch [B, C, H, W]
        copy_edit_aug: Optional torchvision transform for on-the-fly augmentation

    Returns:
        Tuple of (anchor, positive, negative) tensors
    """
    B = batch_imgs.size(0)
    imgs_cpu = batch_imgs.detach().cpu()
    x_anchor = batch_imgs.to(device, non_blocking=True)

    # Create positive pairs (same image, different augmentation)
    if copy_edit_aug is not None:
        x_pos = torch.stack([copy_edit_aug(img) for img in imgs_cpu]).to(device, non_blocking=True)
    else:
        # If no augmentation provided, use anchor as positive (identity)
        x_pos = x_anchor.clone()

    # Create negative pairs (different images, augmented)
    perm = torch.randperm(B)
    if copy_edit_aug is not None:
        x_neg = torch.stack([copy_edit_aug(imgs_cpu[i]) for i in perm]).to(device, non_blocking=True)
    else:
        # If no augmentation provided, use permuted anchors
        x_neg = x_anchor[perm]

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


def kozachenko_leonenko_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Kozachenko–Leonenko differential entropy estimator term (Equation 5 in the paper).
    We want each descriptor's nearest neighbour (excluding itself) to be not too close,
    encouraging more uniform coverage on the hypersphere.

    z: [B, D] global descriptors (CLS projections), already L2-normalized.
    """
    z = F.normalize(z.float(), dim=-1)
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    # Pairwise Euclidean distances
    dists = torch.cdist(z, z, p=2)  # [B, B]

    # Ignore self-distances by setting diagonal to +inf
    inf_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    dists = dists.masked_fill(inf_mask, float("inf"))

    # Nearest neighbour distance for each sample
    min_dists, _ = dists.min(dim=1)  # [B]

    # log of nearest-neighbour distance, averaged over batch
    min_dists = min_dists.clamp_min(eps)
    return min_dists.log().mean()


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
    margin: float = 1.0,
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
    """
    if not os.path.exists(path):
        if rank == 0:
            print(f"[Checkpoint] No existing checkpoint at {path}, starting fresh.")
        return training_state

    map_location = device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
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

    # Create on-the-fly augmentation for positive/negative pair generation during training
    # This is a lightweight torchvision transform applied to tensors, separate from the
    # comprehensive CED augmentation pipeline (35 transforms) applied during data loading
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

    # Build transforms with optional CED augmentation pipeline
    if rank == 0:
        aug_mode = "CED augmentation pipeline (35 transforms)" if cfg.use_ced_augmentations else "simple transforms"
        print(f"[Augmentation] Using {aug_mode}")
        if cfg.use_ced_augmentations:
            print(f"[Augmentation] Random ops per image: [{cfg.ced_min_ops}, {cfg.ced_max_ops}]")
        sys.stdout.flush()

    train_tfms, eval_tfms = build_transforms(
        img_size_train=cfg.img_size_train,
        img_size_eval=cfg.img_size_eval,
        use_ced_augmentations=cfg.use_ced_augmentations,
        ced_min_ops=cfg.ced_min_ops,
        ced_max_ops=cfg.ced_max_ops,
        seed=cfg.ced_aug_seed,
    )

    # DISC21 training dataset
    train_ds = get_train_dataset(root=disc_cfg.root, transform=train_tfms)

    # DDP: distributed sampler for training
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    # Choose num_workers for loaders. Use config as a safe upper bound to
    # avoid excessive per-process CPU/memory use that can trigger OOMs.
    cpu_count = os.cpu_count() or 8
    # Respect the value in config (default small) but don't exceed available CPU quota
    num_workers = min(cfg.num_workers, max(2, cpu_count // max(world_size, 1)))

    if rank == 0:
        print(f"[DataLoader] Using num_workers={num_workers}, prefetch_factor=2, persistent_workers=True")
        sys.stdout.flush()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    # ------------------ DISC21 eval datasets (ALL RANKS) ------------------
    # Instantiate eval datasets on every rank and shard with DistributedSampler.
    ref_ds = get_reference_dataset(root=disc_cfg.root, transform=eval_tfms)
    dev_queries_ds = get_query_dataset("dev", root=disc_cfg.root, transform=eval_tfms)
    test_queries_ds = get_query_dataset("test", root=disc_cfg.root, transform=eval_tfms)

    ref_sampler = DistributedSampler(
        ref_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    dev_q_sampler = DistributedSampler(
        dev_queries_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    test_q_sampler = DistributedSampler(
        test_queries_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    ref_loader = DataLoader(
        ref_ds,
        batch_size=cfg.batch_size_eval,
        sampler=ref_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    dev_query_loader = DataLoader(
        dev_queries_ds,
        batch_size=cfg.batch_size_eval,
        sampler=dev_q_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    test_query_loader = DataLoader(
        test_queries_ds,
        batch_size=cfg.batch_size_eval,
        sampler=test_q_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

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
    ced_model = CEDModel(backbone=backbone, dim=backbone.hidden_size).to(device)

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
                + list(ddp_model.module.classifier.parameters()),
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
                x_anchor, x_pos, x_neg = make_positive_negative_pairs(imgs, copy_edit_aug=copy_edit_aug)

                optimizer.zero_grad(set_to_none=True)

                # forward + loss with AMP autocast
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    # encode anchor & positive
                    # descriptor is global+local; feats_* carries cls_global + local_descriptor
                    v_anchor, feats_anchor = ddp_model.module.encode_images(x_anchor, return_local=True)
                    v_pos, feats_pos = ddp_model.module.encode_images(x_pos, return_local=True)

                    # ==== NEW: use CLS-only projections z for contrastive + KL ====
                    z_anchor = feats_anchor["cls_global"]   # [B, D]
                    z_pos    = feats_pos["cls_global"]      # [B, D]

                    # Concatenate both views for KL (2B descriptors)
                    z_all = torch.cat([z_anchor, z_pos], dim=0)

                    # L_SimCLR (NT-Xent, Eq. 3 & 4)
                    loss_simclr = nt_xent_loss(z_anchor, z_pos, temperature=cfg.temperature_ntxent)

                    # L_KL (Kozachenko–Leonenko entropy term, Eq. 5)
                    loss_kl = kozachenko_leonenko_loss(z_all)

                    # L_contrast = L_SimCLR + λ L_KL  (Eq. 6)
                    loss_contrast = loss_simclr + cfg.lambda_kl * loss_kl

                    # ==== Local multi-similarity loss (L_MSL, Eq. 7) ====
                    local_anchor = feats_anchor.get("local_descriptor")
                    local_pos    = feats_pos.get("local_descriptor")
                    local_embeddings = torch.cat([local_anchor, local_pos], dim=0)
                    batch_ids = torch.arange(local_anchor.size(0), device=local_anchor.device)
                    local_labels = torch.cat([batch_ids, batch_ids], dim=0)
                    loss_local = multi_similarity_loss(local_embeddings, local_labels)

                    # ==== BCE on classifier (L_BCE, Eq. 9) ====
                    anchor_tokens = feats_anchor["patch_tokens_flat"].detach()
                    pos_tokens    = feats_pos["patch_tokens_flat"].detach()

                    with amp.autocast(device_type="cuda", enabled=False):
                        with torch.no_grad():
                            neg_feats = ddp_model.module.backbone.get_features_from_images(x_neg)
                        neg_tokens = neg_feats["patch_tokens_flat"].detach()
                        logits_pos = ddp_model.module.classifier(anchor_tokens, pos_tokens).squeeze(-1)
                        logits_neg = ddp_model.module.classifier(anchor_tokens, neg_tokens).squeeze(-1)

                    labels_pos = torch.ones_like(logits_pos)
                    labels_neg = torch.zeros_like(logits_neg)
                    logits = torch.cat([logits_pos, logits_neg], dim=0)
                    labels = torch.cat([labels_pos, labels_neg], dim=0)

                    loss_bce = bce_loss(logits.float(), labels.float())

                    # ==== Total loss (Eq. 8): L = L_contrast + L_MSL + L_BCE ====
                    loss = loss_contrast + cfg.lambda_local * loss_local + cfg.lambda_bce * loss_bce

                # NaN / inf guard with debug
                if not torch.isfinite(loss):
                    if rank == 0:
                        def finite(t):
                            return bool(torch.isfinite(t).all().item())

                        print("[WARN][DISC21] Non-finite loss encountered; skipping batch.")
                        print(
                            "  components finite?:",
                            f"simclr={finite(loss_simclr)},",
                            f"kl={finite(loss_kl)},",
                            f"local={finite(loss_local)},",
                            f"bce={finite(loss_bce)}",
                        )
                        try:
                            print("  loss_simclr :", float(loss_simclr.detach().cpu()))
                            print("  loss_kl     :", float(loss_kl.detach().cpu()))
                            print("  loss_local  :", float(loss_local.detach().cpu()))
                            print("  loss_bce    :", float(loss_bce.detach().cpu()))
                        except Exception as e:
                            print("  [DEBUG] error printing loss components:", e)
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
                            "simclr": f"{loss_simclr.item():.4f}",
                            "kl": f"{loss_kl.item():.4f}",
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
        log_dir = (script_dir / "artifacts/logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        curves_path = log_dir / "training_curves.json"
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

    # ------------------ Evaluation (Distributed) ------------------
    eval_model = ddp_model.module  # unwrap CEDModel
    encoding_cache_root = (script_dir / cfg.encoding_cache_root).resolve()
    encoding_cache_root.mkdir(parents=True, exist_ok=True)

    def encode_and_save_rank(loader: DataLoader, relative_artifact_name: str):
        """Encode loader shard on this rank and persist to a rank-specific folder."""

        rank_root = encoding_cache_root / relative_artifact_name / f"rank_{rank}"
        rank_root.mkdir(parents=True, exist_ok=True)
        save_prefix = rank_root / f"rank_{rank}"

        return compute_descriptors_for_loader(
            loader,
            eval_model,
            save_path_prefix=str(save_prefix),
            checkpoint_root=str(rank_root),
            resume=True,
            chunk_size=cfg.encoding_chunk_size,
        )

    if rank == 0:
        print("\n[Eval] Starting Distributed Encoding...")
        sys.stdout.flush()

    encode_and_save_rank(ref_loader, "disc21_ref")
    encode_and_save_rank(dev_query_loader, "disc21_dev_query")
    encode_and_save_rank(test_query_loader, "disc21_test_query")
    encode_and_save_rank(ndec_ref_loader, "ndec_ref")
    encode_and_save_rank(ndec_query_loader, "ndec_query")

    if rank == 0:
        print("[Eval] Waiting for all ranks to finish encoding...")
        sys.stdout.flush()
    dist.barrier()

    if rank == 0:
        print("[Eval] All ranks finished. Merging results and computing metrics...")
        sys.stdout.flush()

        def load_and_merge_ranks(relative_artifact_name: str):
            merged_vecs: List[torch.Tensor] = []
            merged_ids: List[str] = []

            base_path = encoding_cache_root / relative_artifact_name

            for r in range(world_size):
                rank_folder = base_path / f"rank_{r}" / f"rank_{r}"
                vec_path = rank_folder / "final_embeddings.pt"
                id_path = rank_folder / "final_ids.npy"

                if not vec_path.exists() or not id_path.exists():
                    print(f"[WARN] Missing output for rank {r} at {vec_path}")
                    continue

                merged_vecs.append(torch.load(vec_path, map_location="cpu"))
                merged_ids.extend(np.load(id_path, allow_pickle=True).tolist())

            if not merged_vecs:
                return torch.empty(0), []

            final_vecs = torch.cat(merged_vecs, dim=0)

            unique_map: Dict[str, int] = {}
            unique_order: List[int] = []
            for idx, sample_id in enumerate(merged_ids):
                sid = str(sample_id)
                if sid in unique_map:
                    continue
                unique_map[sid] = idx
                unique_order.append(idx)

            final_vecs = final_vecs[unique_order]
            final_ids = [merged_ids[i] for i in unique_order]

            print(
                f"       Merged {relative_artifact_name}: {len(merged_ids)} -> {len(final_ids)} unique items."
            )
            return final_vecs.to(device), final_ids

        disc_ref_vecs, disc_ref_ids = load_and_merge_ranks("disc21_ref")
        disc_ref_vecs = F.normalize(disc_ref_vecs.float(), dim=-1)

        for split_name in ["dev", "test"]:
            print(f"\n[Eval][DISC21-{split_name}] Computing metrics...")
            gt_map = load_disc21_groundtruth_map(split=split_name, root=disc_cfg.root)

            if cfg.use_ced_two_stage_eval:
                # Use CED two-stage evaluation (patch + classifier)
                print(f"[Eval][DISC21-{split_name}] Running CED two-stage pipeline...")
                print(f"  k_candidates per patch: {cfg.ced_k_candidates_per_patch}")

                # Get the appropriate query loader
                if split_name == "dev":
                    query_loader_for_eval = dev_query_loader
                else:
                    query_loader_for_eval = test_query_loader

                ced_metrics = ced_two_stage_eval(
                    model=eval_model,
                    query_loader=query_loader_for_eval,
                    ref_vecs=disc_ref_vecs,
                    ref_ids=disc_ref_ids,
                    ref_ds=ref_ds,
                    gt_map=gt_map,
                    k_candidates=cfg.ced_k_candidates_per_patch,
                    device=device,
                    verbose=True,
                )
                print(f"[Eval][DISC21-{split_name}] CED two-stage metrics:", ced_metrics)

                # Optionally also compute descriptor-only baseline for comparison
                print(f"[Eval][DISC21-{split_name}] Computing descriptor-only baseline for comparison...")
                q_vecs, q_ids = load_and_merge_ranks(f"disc21_{split_name}_query")
                baseline_metrics = evaluate_retrieval(
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
                print(f"[Eval][DISC21-{split_name}] Descriptor-only baseline:", baseline_metrics)
                del q_vecs, q_ids, baseline_metrics

            else:
                # Use descriptor-only evaluation (old method)
                q_vecs, q_ids = load_and_merge_ranks(f"disc21_{split_name}_query")

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
                print(f"[Eval][DISC21-{split_name}] Descriptor-only metrics:", metrics)
                del q_vecs, q_ids, metrics

            gc.collect()

        print("\n[Eval][NDEC] Computing metrics...")
        ndec_ref_vecs, ndec_ref_ids = load_and_merge_ranks("ndec_ref")
        ndec_ref_vecs = F.normalize(ndec_ref_vecs.float(), dim=-1)
        ndec_gt_map = load_ndec_groundtruth_map(cfg.ndec_root, drop_missing=True)

        if cfg.use_ced_two_stage_eval:
            # Use CED two-stage evaluation for NDEC
            print(f"[Eval][NDEC] Running CED two-stage pipeline...")
            print(f"  k_candidates per patch: {cfg.ced_k_candidates_per_patch}")

            # Get NDEC ref dataset from loader
            ndec_ref_ds = ndec_ref_loader.dataset

            ced_ndec_metrics = ced_two_stage_eval(
                model=eval_model,
                query_loader=ndec_query_loader,
                ref_vecs=ndec_ref_vecs,
                ref_ids=ndec_ref_ids,
                ref_ds=ndec_ref_ds,
                gt_map=ndec_gt_map,
                k_candidates=cfg.ced_k_candidates_per_patch,
                device=device,
                verbose=True,
            )
            print("[Eval][NDEC] CED two-stage metrics:", ced_ndec_metrics)

            # Optionally compute descriptor-only baseline for comparison
            print(f"[Eval][NDEC] Computing descriptor-only baseline for comparison...")
            ndec_query_vecs, ndec_query_ids = load_and_merge_ranks("ndec_query")
            ndec_baseline_metrics = evaluate_retrieval(
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
            print("[Eval][NDEC] Descriptor-only baseline:", ndec_baseline_metrics)

        else:
            # Use descriptor-only evaluation (old method)
            ndec_query_vecs, ndec_query_ids = load_and_merge_ranks("ndec_query")

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
            print("[Eval][NDEC] Descriptor-only metrics:", ndec_metrics)

        sys.stdout.flush()

    if did_train and rank == 0:
        print(f"\n[Training complete] Final checkpoint at {checkpoint_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
