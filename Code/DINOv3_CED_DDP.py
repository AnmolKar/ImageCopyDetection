import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoModel, AutoProcessor
from huggingface_hub import login
from dotenv import load_dotenv

import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as V

# ======================================================================
# Import your loaders + eval utilities
# ======================================================================
code_dir = Path("/home/jowatson/Deep Learning/Code").resolve()
if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))

utilities_dir = code_dir / "Utilities"
if str(utilities_dir) not in sys.path:
    sys.path.append(str(utilities_dir))

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
    disc21_root: Path = Path("/home/jowatson/Deep Learning/DISC21")
    ndec_root: Path = Path("/home/jowatson/Deep Learning/NDEC")
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    img_size_train: int = 224
    img_size_eval: int = 224
    batch_size_train: int = 64
    batch_size_eval: int = 64
    batch_size_pairs: int = 64
    num_workers: int = 4
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    temperature_ntxent: float = 0.2
    temperature_kl: float = 0.1
    lambda_kl: float = 1.0
    lambda_local: float = 1.0
    lambda_bce: float = 1.0
    lambda_asl_mtr: float = 1.0
    num_epochs_disc21: int = 30
    num_epochs_ndec: int = 20
    early_stopping_patience_disc21: int = 3
    early_stopping_min_delta_disc21: float = 1e-4
    early_stopping_patience_ndec: int = 2
    early_stopping_min_delta_ndec: float = 1e-4
    checkpoint_path: Path = Path("artifacts/checkpoints/ced_model_ddp.pt")


@dataclass
class NdecConfig:
    root: Path = Path("/home/jowatson/Deep Learning/NDEC")
    img_size_train: int = 224
    img_size_eval: int = 224
    batch_size_pairs: int = 64
    batch_size_eval: int = 64
    num_workers: int = 4


# ======================================================================
# Backbone, aggregator, classifier, CEDModel
# ======================================================================
class DinoV3Backbone(nn.Module):
    """Wrapper that exposes token-level DINOv3 activations."""

    def __init__(self, model_name: str):
        super().__init__()
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = self._load_processor(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype, trust_remote_code=True)
        self.model.to(device)
        self._supports_attn_impl = hasattr(self.model, "set_attn_implementation")
        self._attn_impl_is_eager = False
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

    @torch.inference_mode()
    def preprocess(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        images_np = [img.detach().cpu().permute(1, 2, 0).numpy() for img in images]
        encoded = self.processor(images=images_np, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        if output_attentions and self._supports_attn_impl and not self._attn_impl_is_eager:
            try:
                self.model.set_attn_implementation("eager")
                self._attn_impl_is_eager = True
            except Exception as exc:
                warnings.warn(
                    f"Unable to switch attention implementation to 'eager': {exc}. "
                    "Attention maps may be unavailable.",
                    stacklevel=1,
                )

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
        x = x.clamp(min=eps).pow(p)
        x = x.mean(dim=1)
        return x.pow(1.0 / p)

    def compute_local_embedding(
        self,
        patch_tokens_flat: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weighted_tokens = patch_tokens_flat
        if attn_weights is not None:
            weights = attn_weights.view(attn_weights.size(0), -1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            weights = weights.unsqueeze(-1)
            weighted_tokens = patch_tokens_flat * (weights + 1e-6)

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
        cls_global = self.proj_cls(cls)
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


class CEDModel(nn.Module):
    """High-level wrapper orchestrating backbone encoding and pair scoring."""

    def __init__(self, backbone: DinoV3Backbone, dim: int):
        super().__init__()
        self.backbone = backbone
        model_dtype = next(self.backbone.parameters()).dtype
        self.aggregator = CEDFeatureAggregator(dim=dim, gem_p=3.0, use_proj=True)
        self.aggregator = self.aggregator.to(device=device, dtype=model_dtype)
        self.classifier = CopyEditClassifier(dim=dim)
        self.classifier = self.classifier.to(device=device, dtype=model_dtype)

    def encode_images(
        self,
        images: torch.Tensor,
        return_local: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.backbone.get_features_from_images(images, output_attentions=True)
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
        q_feats = self.backbone.get_features_from_images(q_images)
        r_feats = self.backbone.get_features_from_images(r_images)
        q_tokens = q_feats["patch_tokens_flat"]
        r_tokens = r_feats["patch_tokens_flat"]
        logits = self.classifier(q_tokens, r_tokens).squeeze(-1)
        return logits, q_feats, r_feats


# ======================================================================
# Losses, augmentations
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
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    reps = torch.cat([z1, z2], dim=0)
    sim = reps @ reps.t() / temperature
    mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    if not torch.is_floating_point(sim):
        raise TypeError("Similarity matrix must be floating point for NT-Xent loss")
    fill_value = torch.finfo(sim.dtype).min
    sim.masked_fill_(mask, float(fill_value))
    batch_size = z1.size(0)
    targets = torch.arange(batch_size, 2 * batch_size, device=sim.device)
    targets = torch.cat([targets, torch.arange(0, batch_size, device=sim.device)])
    loss = F.cross_entropy(sim, targets)
    return loss


def similarity_kl_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim1 = z1 @ z1.t() / temperature
    sim2 = z2 @ z2.t() / temperature
    mask = torch.eye(sim1.size(0), dtype=torch.bool, device=sim1.device)
    if not torch.is_floating_point(sim1):
        raise TypeError("Similarity matrix must be floating point for KL loss")
    fill_value = torch.finfo(sim1.dtype).min
    sim1.masked_fill_(mask, float(fill_value))
    sim2.masked_fill_(mask, float(fill_value))
    p1 = F.softmax(sim1, dim=-1)
    p2 = F.softmax(sim2, dim=-1)
    kl1 = F.kl_div(p1.log(), p2, reduction="batchmean")
    kl2 = F.kl_div(p2.log(), p1, reduction="batchmean")
    return 0.5 * (kl1 + kl2)


def multi_similarity_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 50.0,
    margin: float = 0.5,
) -> torch.Tensor:
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device)
    sims = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).t()
    loss = torch.zeros(1, device=embeddings.device)
    valid = 0
    for i in range(embeddings.size(0)):
        pos_mask = (labels == labels[i]).clone()
        pos_mask[i] = False
        neg_mask = labels != labels[i]
        pos_sims = sims[i][pos_mask]
        neg_sims = sims[i][neg_mask]
        if pos_sims.numel() == 0 or neg_sims.numel() == 0:
            continue
        pos_term = (1.0 / alpha) * torch.log1p(
            torch.sum(torch.exp(-alpha * (pos_sims - (1 - margin))))
        )
        neg_term = (1.0 / beta) * torch.log1p(
            torch.sum(torch.exp(beta * (neg_sims - margin)))
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
    norm_f = v_former.norm(p=2, dim=-1)
    norm_l = v_latter.norm(p=2, dim=-1)
    ratio = (norm_l + eps) / (norm_f + eps)
    loss_ratio = torch.exp(1.0 - ratio).mean()
    loss_metric = nt_xent_loss(v_former, v_latter)
    return loss_ratio + lambda_mtr * loss_metric


bce_loss = nn.BCEWithLogitsLoss()


# ======================================================================
# Checkpoint helpers
# ======================================================================
def save_checkpoint(model: CEDModel, optimizer: torch.optim.Optimizer, path: str, rank: int):
    if rank != 0:
        return
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"[Rank 0] Saved checkpoint to {path}")


# ======================================================================
# Ground-truth helpers for DISC21 + NDEC (for eval_utils.evaluate_retrieval)
# ======================================================================
def load_disc21_groundtruth_map(split: str, root: Path) -> Dict[str, List[str]]:
    """
    Build query_id -> [reference_id, ...] map from DISC21 CSV.
    """
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
# Main training + evaluation (DDP)
# ======================================================================
def main():
    global device

    # ------------------ basic setup ------------------
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    cfg = ExperimentConfig()
    ndec_cfg = NdecConfig()
    if rank == 0:
        print("Config:", cfg)
        print(f"World size: {world_size} | Rank: {rank} | Local rank: {local_rank}")
        sys.stdout.flush()

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

    ndec_data_cfg = NdecDataConfig(
        root=ndec_cfg.root,
        img_size_train=ndec_cfg.img_size_train,
        img_size_eval=ndec_cfg.img_size_eval,
        batch_size_pairs=ndec_cfg.batch_size_pairs,
        batch_size_eval=ndec_cfg.batch_size_eval,
        num_workers=ndec_cfg.num_workers,
    )

    train_tfms, eval_tfms = build_transforms(
        img_size_train=cfg.img_size_train,
        img_size_eval=cfg.img_size_eval,
    )

    # DISC21 training dataset
    train_ds = get_train_dataset(root=disc_cfg.root, transform=train_tfms)

    # DDP: distributed sampler for training
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_train,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
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
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        dev_query_loader = DataLoader(
            dev_queries_ds,
            batch_size=cfg.batch_size_eval,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        test_query_loader = DataLoader(
            test_queries_ds,
            batch_size=cfg.batch_size_eval,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    else:
        ref_loader = dev_query_loader = test_query_loader = None  # type: ignore

    # NDEC loaders (query, ref, pos, neg)
    ndec_query_loader, ndec_ref_loader, ndec_pos_pair_loader, ndec_neg_pair_loader = build_ndec_loaders(ndec_data_cfg)

    if rank == 0:
        print(f"Train images: {len(train_ds):,}")
        print(f"NDEC negative pairs batches: {len(ndec_neg_pair_loader):,}")
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

    # ------------------ training flags ------------------
    should_train_disc21 = True
    should_train_ndec = True  # NDEC ASL stage runs on rank 0 only
    did_train = False

    # ------------------ DISC21 training loop (DDP) ------------------
    if should_train_disc21:
        did_train = True
        best_disc21_loss = float("inf")
        disc21_bad_epochs = 0

        for epoch in range(cfg.num_epochs_disc21):
            ddp_model.train()
            train_sampler.set_epoch(epoch)

            if rank == 0:
                progress = tqdm(train_loader, desc=f"[DISC21][Epoch {epoch+1}/{cfg.num_epochs_disc21}]")
            else:
                progress = train_loader

            epoch_loss = 0.0

            for imgs, _ in progress:
                imgs = imgs.to(device, non_blocking=True)
                x_anchor, x_pos, x_neg = make_positive_negative_pairs(imgs)

                optimizer.zero_grad(set_to_none=True)

                # encode anchor & positive
                v_anchor, feats_anchor = ddp_model.module.encode_images(x_anchor, return_local=True)
                v_pos, feats_pos = ddp_model.module.encode_images(x_pos, return_local=True)

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

                with torch.no_grad():
                    neg_feats = ddp_model.module.backbone.get_features_from_images(x_neg)
                neg_tokens = neg_feats["patch_tokens_flat"].detach()

                logits_pos = ddp_model.module.classifier(anchor_tokens, pos_tokens).squeeze(-1)
                logits_neg = ddp_model.module.classifier(anchor_tokens, neg_tokens).squeeze(-1)

                labels_pos = torch.ones_like(logits_pos)
                labels_neg = torch.zeros_like(logits_neg)
                logits = torch.cat([logits_pos, logits_neg], dim=0)
                labels = torch.cat([labels_pos, labels_neg], dim=0)
                loss_bce = bce_loss(logits, labels)

                loss = (
                    loss_contrast
                    + cfg.lambda_kl * loss_kl
                    + cfg.lambda_local * loss_local
                    + cfg.lambda_bce * loss_bce
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

                if rank == 0:
                    progress.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "contrast": f"{loss_contrast.item():.4f}",
                            "local": f"{loss_local.item():.4f}",
                        }
                    )

            # Average loss across processes
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss_avg = epoch_loss_tensor.item() / world_size / len(train_loader)

            if rank == 0:
                print(f"[DISC21][Epoch {epoch+1}] avg loss across ranks = {epoch_loss_avg:.4f}")
                sys.stdout.flush()

                if epoch_loss_avg + cfg.early_stopping_min_delta_disc21 < best_disc21_loss:
                    best_disc21_loss = epoch_loss_avg
                    disc21_bad_epochs = 0
                    save_checkpoint(ddp_model.module, optimizer, str(cfg.checkpoint_path), rank=0)
                else:
                    disc21_bad_epochs += 1
                    if disc21_bad_epochs >= cfg.early_stopping_patience_disc21:
                        print(
                            f"[DISC21] Early stopping after {cfg.early_stopping_patience_disc21} bad epoch(s)."
                        )
                        break

    # ------------------ NDEC ASL fine-tuning (rank 0 only) ------------------
    if should_train_ndec and rank == 0:
        did_train = True
        best_ndec_loss = float("inf")
        ndec_bad_epochs = 0

        print("[NDEC] Starting ASL fine-tuning on rank 0 only.")
        sys.stdout.flush()

        for epoch in range(cfg.num_epochs_ndec):
            ddp_model.module.train()
            total_loss = 0.0

            progress = tqdm(ndec_neg_pair_loader, desc=f"[NDEC][Epoch {epoch+1}/{cfg.num_epochs_ndec}]")

            for img_a, img_b, _, _ in progress:
                img_a = img_a.to(device, non_blocking=True)
                img_b = img_b.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                v_former, _ = ddp_model.module.encode_images(img_a)
                v_latter, _ = ddp_model.module.encode_images(img_b)

                loss_asl = asl_loss(v_former=v_former, v_latter=v_latter, lambda_mtr=cfg.lambda_asl_mtr)

                loss_asl.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss_asl.item()
                progress.set_postfix({"asl_loss": f"{loss_asl.item():.4f}"})

            avg_asl = total_loss / len(ndec_neg_pair_loader)
            print(f"[NDEC][Epoch {epoch+1}] avg ASL loss = {avg_asl:.4f}")
            sys.stdout.flush()

            if avg_asl + cfg.early_stopping_min_delta_ndec < best_ndec_loss:
                best_ndec_loss = avg_asl
                ndec_bad_epochs = 0
                save_checkpoint(ddp_model.module, optimizer, str(cfg.checkpoint_path), rank=0)
            else:
                ndec_bad_epochs += 1
                if ndec_bad_epochs >= cfg.early_stopping_patience_ndec:
                    print(
                        f"[NDEC] Early stopping after {cfg.early_stopping_patience_ndec} bad epoch(s)."
                    )
                    break

    # ------------------ Evaluation using eval_utils (rank 0 only) ------------------
    if rank == 0:
        print("\n[Eval] Starting DISC21 + NDEC evaluation with eval_utils metrics (µAP, R@P90).")
        sys.stdout.flush()

        eval_model = ddp_model.module  # unwrap CEDModel

        # ---- DISC21: compute reference descriptors once ----
        disc_ref_vecs, disc_ref_ids = compute_descriptors_for_loader(
            ref_loader, eval_model, "artifacts/disc21_ref"
        )

        for split_name, q_loader in [("dev", dev_query_loader), ("test", test_query_loader)]:
            print(f"\n[Eval][DISC21-{split_name}] Encoding queries + computing retrieval metrics...")
            sys.stdout.flush()

            q_vecs, q_ids = compute_descriptors_for_loader(
                q_loader, eval_model, f"artifacts/disc21_{split_name}_query"
            )
            gt_map = load_disc21_groundtruth_map(split=split_name, root=disc_cfg.root)

            metrics = evaluate_retrieval(
                query_vecs=q_vecs,
                query_ids=q_ids,
                ref_vecs=disc_ref_vecs,
                ref_ids=disc_ref_ids,
                gt_map=gt_map,
                topk_list=[1, 5, 10, 20],
            )
            print(f"[Eval][DISC21-{split_name}] metrics:", metrics)
            sys.stdout.flush()

        # ---- NDEC retrieval metrics ----
        print("\n[Eval][NDEC] Encoding references + queries...")
        sys.stdout.flush()

        ndec_ref_vecs, ndec_ref_ids = compute_descriptors_for_loader(
            ndec_ref_loader, eval_model, "artifacts/ndec_ref"
        )
        ndec_query_vecs, ndec_query_ids = compute_descriptors_for_loader(
            ndec_query_loader, eval_model, "artifacts/ndec_query"
        )
        ndec_gt_map = load_ndec_groundtruth_map(ndec_cfg.root, drop_missing=True)

        ndec_metrics = evaluate_retrieval(
            query_vecs=ndec_query_vecs,
            query_ids=ndec_query_ids,
            ref_vecs=ndec_ref_vecs,
            ref_ids=ndec_ref_ids,
            gt_map=ndec_gt_map,
            topk_list=[1, 5, 10, 20],
        )
        print("[Eval][NDEC] metrics:", ndec_metrics)
        sys.stdout.flush()

    if did_train and rank == 0:
        print(f"\n[Training complete] Final checkpoint at {cfg.checkpoint_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
