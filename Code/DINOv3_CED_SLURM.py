import sys
import os
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoImageProcessor, AutoModel, AutoProcessor
from huggingface_hub import login
from dotenv import load_dotenv

import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as V

from disc21_loader import (
    Disc21DataConfig,
    build_transforms,
    get_train_dataset,
    get_reference_dataset,
    get_query_dataset,
    get_pair_dataset,
    create_dataloader,
    load_groundtruth,
)

from ndec_loader import (
    NdecDataConfig,
    build_default_loaders as build_ndec_loaders,
    load_groundtruth as load_ndec_groundtruth,
)


code_dir = Path("/home/jowatson/Deep Learning/Code").resolve()
if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))
    print(f"Added {code_dir} to sys.path")
else:
    print(f"Using existing sys.path entry: {code_dir}")

from ndec_loader import (
    NdecDataConfig,
    build_default_loaders as build_ndec_loaders,
    load_groundtruth as load_ndec_groundtruth,
 )

# %%
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

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
    checkpoint_path: Path = Path("artifacts/checkpoints/ced_model_final.pt")


@dataclass

class NdecConfig:
    root: Path = Path("/home/jowatson/Deep Learning/NDEC")
    img_size_train: int = 224
    img_size_eval: int = 224
    batch_size_pairs: int = 64
    batch_size_eval: int = 64
    num_workers: int = 4


cfg = ExperimentConfig()
print(cfg)

disc_cfg = Disc21DataConfig(
    root=cfg.disc21_root,
    img_size_train=cfg.img_size_train,
    img_size_eval=cfg.img_size_eval,
    batch_size_train=cfg.batch_size_train,
    batch_size_eval=cfg.batch_size_eval,
    num_workers=cfg.num_workers,
 )

ndec_cfg = NdecConfig(root=cfg.ndec_root)

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

train_ds = get_train_dataset(root=disc_cfg.root, transform=train_tfms)
ref_ds = get_reference_dataset(root=disc_cfg.root, transform=eval_tfms)

dev_queries_ds = get_query_dataset(
    "dev",
    root=disc_cfg.root,
    transform=eval_tfms,
 )

test_queries_ds = get_query_dataset(
    "test",
    root=disc_cfg.root,
    transform=eval_tfms,
 )

dev_pairs_ds = get_pair_dataset(
    "dev",
    root=disc_cfg.root,
    transform_query=eval_tfms,
    transform_reference=eval_tfms,
 )

test_pairs_ds = get_pair_dataset(
    "test",
    root=disc_cfg.root,
    transform_query=eval_tfms,
    transform_reference=eval_tfms,
 )

train_loader = create_dataloader(
    train_ds,
    batch_size=cfg.batch_size_train,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
 )

ref_loader = create_dataloader(
    ref_ds,
    batch_size=cfg.batch_size_eval,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
 )

dev_query_loader = create_dataloader(
    dev_queries_ds,
    batch_size=cfg.batch_size_eval,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
 )

test_query_loader = create_dataloader(
    test_queries_ds,
    batch_size=cfg.batch_size_eval,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
 )

dev_loader = create_dataloader(
    dev_pairs_ds,
    batch_size=cfg.batch_size_eval,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
 )

test_loader = create_dataloader(
    test_pairs_ds,
    batch_size=cfg.batch_size_eval,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
 )

ndec_query_loader, ndec_ref_loader, ndec_pos_pair_loader, ndec_neg_pair_loader = build_ndec_loaders(
    ndec_data_cfg
 )

print(
    f"Train images: {len(train_ds):,} | References: {len(ref_ds):,} | "
    f"Dev queries: {len(dev_queries_ds):,} | Test queries: {len(test_queries_ds):,} | "
    f"Dev pairs: {len(dev_pairs_ds):,} | Test pairs: {len(test_pairs_ds):,}"
 )

# %% [markdown]
# # DINOv3 Backbone Wrapper (ViT-B/16)
# 
# Wrap Hugging Face's ViT-B/16 DINOv3 checkpoint so downstream modules can request normalized CLS, register, and patch tokens without worrying about preprocessing logistics.

# %%
class DinoV3Backbone(nn.Module):

    """ Wrapper that exposes token-level DINOv3 activations."""
    def __init__(self, model_name: str = cfg.model_name):

        super().__init__()
        dtype = torch.float16 if device.type == "cuda" else torch.float32

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
        """Load image processor, falling back to generic processor when needed."""
        try:
            return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        except ValueError:
            return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    @torch.inference_mode()

    def preprocess(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert float tensors in [0,1] into pixel values expected by the HF model."""

        images_np = [img.detach().cpu().permute(1, 2, 0).numpy() for img in images]
        encoded = self.processor(images=images_np, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded



    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        """Run the ViT and return a dictionary of relevant token tensors."""

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

        """Preprocess raw images then run a forward pass in one call."""
        encoded = self.preprocess(images)
        return self.forward(encoded["pixel_values"], output_attentions=output_attentions)

# %%
backbone = DinoV3Backbone(model_name=cfg.model_name)



try:

    sample_imgs, sample_ids = next(iter(train_loader))
    sample_imgs = sample_imgs[:2].to(device)  # keep it light for smoke test

    with torch.no_grad():
        feats = backbone.get_features_from_images(sample_imgs)

    print(
        "CLS:", feats["cls"].shape,
        "| Patch grid:", feats["patch_tokens"].shape,
        "| Flat patches:", feats["patch_tokens_flat"].shape,
    )

except StopIteration:
    print("Train loader is empty—please verify DISC21 data paths.")

# %% [markdown]
# # CED-style Feature Aggregation (Global + Local)
# 
# Aggregate CLS and spatial patch descriptors into a compact CED vector via GeM pooling, emulating the original copy-edit detector's global/local fusion.

# %%
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
        """Generalized mean pooling over the sequence dimension."""

        x = x.clamp(min=eps).pow(p)
        x = x.mean(dim=1)

        return x.pow(1.0 / p)

    def compute_local_embedding(
        self, patch_tokens_flat: torch.Tensor, attn_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the normalized local descriptor after optional attention weighting."""
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
        """Return a concatenated, L2-normalized descriptor (global + local)."""

        cls_global = self.proj_cls(cls)
        cls_global = F.normalize(cls_global, dim=-1)
        local = self.compute_local_embedding(patch_tokens_flat, attn_weights=attn_weights)
        descriptor = torch.cat([cls_global, local], dim=-1)
        descriptor = F.normalize(descriptor, dim=-1)

        if return_local:
            return descriptor, local

        return descriptor

# %%
backbone_dtype = next(backbone.parameters()).dtype
aggregator = CEDFeatureAggregator(dim=backbone.hidden_size, gem_p=3.0, use_proj=True)
aggregator = aggregator.to(device=device, dtype=backbone_dtype)

if "feats" in locals():
    desc = aggregator(feats["cls"], feats["patch_tokens_flat"])
    print("CED descriptor shape:", desc.shape)

else:
    print("No cached backbone features available for aggregation test.")

# %% [markdown]
# # Copy-Edit Classifier Head (Cross-Attention)
# 
# Model pairwise copy-edit likelihood with a lightweight cross-attention module that lets query tokens attend to reference tokens before global pooling.

# %%
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

        """Return logits for each query/reference pair."""

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


# %%
classifier = CopyEditClassifier(dim=backbone.hidden_size)
classifier = classifier.to(device=device, dtype=backbone_dtype)

if "feats" in locals():
    q_tokens = feats["patch_tokens_flat"]
    r_tokens = feats["patch_tokens_flat"]

    logits = classifier(q_tokens, r_tokens)

    print("Classifier logits shape:", logits.shape)
else:

    print("No backbone features cached yet for classifier sanity check.")

# %% [markdown]
# ## CED Model Wrapper
# 
# Bundle the backbone, aggregator, and classifier so training/eval code can call concise helper methods (`encode_images`, `score_pair`).

# %%
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
        self, images: torch.Tensor, return_local: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.backbone.get_features_from_images(
            images, output_attentions=True
        )
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



    def score_pair(self, q_images: torch.Tensor, r_images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        q_feats = self.backbone.get_features_from_images(q_images)
        r_feats = self.backbone.get_features_from_images(r_images)
        q_tokens = q_feats["patch_tokens_flat"]
        r_tokens = r_feats["patch_tokens_flat"]
        logits = self.classifier(q_tokens, r_tokens).squeeze(-1)

        return logits, q_feats, r_feats

# %% [markdown]
# # Self-Supervised Training Loop (DISC21)
# 
# Create synthetic copy-edits, apply the combined NT-Xent + KL + local multi-sim objectives, and fine-tune the backbone/heads jointly before introducing the classifier BCE term.

# %%
copy_edit_aug = V.Compose(

    [
        V.RandomResizedCrop(cfg.img_size_train, scale=(0.6, 1.0)),
        V.RandomHorizontalFlip(),
        V.ColorJitter(0.2, 0.2, 0.2, 0.1),
        V.RandomGrayscale(p=0.1),
    ]
)

def make_positive_negative_pairs(batch_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create anchor/positive/negative triplets via stochastic augmentations."""

    B = batch_imgs.size(0)
    imgs_cpu = batch_imgs.detach().cpu()
    x_anchor = batch_imgs.to(device)
    x_pos = torch.stack([copy_edit_aug(img) for img in imgs_cpu]).to(device)
    perm = torch.randperm(B)
    x_neg = torch.stack([copy_edit_aug(imgs_cpu[i]) for i in perm]).to(device)

    return x_anchor, x_pos, x_neg



def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """SimCLR-style NT-Xent contrastive loss over normalized embeddings."""

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
    """Symmetric KL divergence between similarity distributions of two views."""
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
    embeddings: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0, beta: float = 50.0, margin: float = 0.5
) -> torch.Tensor:
    """Multi-similarity loss on local descriptors within a batch."""
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
        pos_term = (1.0 / alpha) * torch.log1p(torch.sum(torch.exp(-alpha * (pos_sims - (1 - margin)))))
        neg_term = (1.0 / beta) * torch.log1p(torch.sum(torch.exp(beta * (neg_sims - margin))))
        loss = loss + pos_term + neg_term
        valid += 1
    if valid == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return loss / valid



def asl_loss(
    v_former: torch.Tensor, v_latter: torch.Tensor, lambda_mtr: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    """Simplified ASL-style norm-ratio + metric alignment loss."""
    norm_f = v_former.norm(p=2, dim=-1)
    norm_l = v_latter.norm(p=2, dim=-1)
    ratio = (norm_l + eps) / (norm_f + eps)
    loss_ratio = torch.exp(1.0 - ratio).mean()
    loss_metric = nt_xent_loss(v_former, v_latter)
    return loss_ratio + lambda_mtr * loss_metric


bce_loss = nn.BCEWithLogitsLoss()

# %% [markdown]
# ## Loss Breakdown (CEDetector Alignment)
# 
# - **Global contrast + KL**: NT-Xent enforces view consistency while symmetric KL aligns similarity distributions, mirroring CEDetector's global objective.
# - **Local multi-sim**: GeM pooled locals use CLS attention weights so positives stay close and negatives push apart via multi-similarity mining.
# - **Classifier BCE (stop-grad)**: Copy-edit head receives frozen backbone tokens so it regularizes head weights without disturbing the encoder.

# %% [markdown]
# ## ASL Fine-Tuning (NDEC Hard Negatives)
# 
# Hard mismatched pairs from NDEC drive an ASL-style loss that (1) enforces norm asymmetry between former/latter patches and (2) keeps their descriptors metrically aligned, echoing the ASL paper's formulation.

# %%
ced_model = CEDModel(backbone=backbone, dim=backbone.hidden_size).to(device)

optimizer = torch.optim.AdamW(

    [
        {"params": ced_model.backbone.parameters(), "lr": cfg.lr_backbone},

        {
            "params": list(ced_model.aggregator.parameters())
            + list(ced_model.classifier.parameters()),
            "lr": cfg.lr_head,
        },
    ],

    weight_decay=cfg.weight_decay,
 )



should_train_disc21 = True  # flip to True to launch DISC21 self-supervised training
should_train_ndec = True    # flip to True to run the NDEC ASL fine-tuning stage

did_train = False


# %% [markdown]
# # (Optional) Save/Load Checkpoints & Precomputed Embeddings
#
# Utility helpers to persist model weights and cached descriptors for faster experimentation across sessions.


def save_checkpoint(model: CEDModel, optimizer: torch.optim.Optimizer, path: str):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.__dict__,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model: CEDModel, optimizer: torch.optim.Optimizer, path: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    print(f"Restored checkpoint from {path}")



if should_train_disc21:
    did_train = True

    best_disc21_loss = float("inf")
    disc21_bad_epochs = 0

    for epoch in range(cfg.num_epochs_disc21):

        ced_model.train()

        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"DISC21 Epoch {epoch + 1}/{cfg.num_epochs_disc21}")

        for imgs, _ in progress:

            x_anchor, x_pos, x_neg = make_positive_negative_pairs(imgs)

            optimizer.zero_grad()

            v_anchor, feats_anchor = ced_model.encode_images(x_anchor, return_local=True)
            v_pos, feats_pos = ced_model.encode_images(x_pos, return_local=True)

            loss_contrast = nt_xent_loss(
                v_anchor, v_pos, temperature=cfg.temperature_ntxent
            )
            loss_kl = similarity_kl_loss(
                v_anchor, v_pos, temperature=cfg.temperature_kl
            )

            local_anchor = feats_anchor.get("local_descriptor")
            local_pos = feats_pos.get("local_descriptor")
            local_embeddings = torch.cat([local_anchor, local_pos], dim=0)
            batch_ids = torch.arange(local_anchor.size(0), device=local_anchor.device)
            local_labels = torch.cat([batch_ids, batch_ids], dim=0)
            loss_local = multi_similarity_loss(
                local_embeddings, local_labels
            )

            anchor_tokens = feats_anchor["patch_tokens_flat"].detach()
            pos_tokens = feats_pos["patch_tokens_flat"].detach()

            with torch.no_grad():
                neg_feats = ced_model.backbone.get_features_from_images(x_neg)
            neg_tokens = neg_feats["patch_tokens_flat"].detach()

            logits_pos = ced_model.classifier(
                anchor_tokens,
                pos_tokens,
            ).squeeze(-1)
            logits_neg = ced_model.classifier(
                anchor_tokens,
                neg_tokens,
            ).squeeze(-1)

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

            torch.nn.utils.clip_grad_norm_(ced_model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "contrast": f"{loss_contrast.item():.4f}",
                "local": f"{loss_local.item():.4f}",
            })

        avg_loss = total_loss / len(train_loader)

        print(f"DISC21 Epoch {epoch + 1}: avg loss = {avg_loss:.4f}")

        if avg_loss + cfg.early_stopping_min_delta_disc21 < best_disc21_loss:
            best_disc21_loss = avg_loss
            disc21_bad_epochs = 0
        else:
            disc21_bad_epochs += 1
            if disc21_bad_epochs >= cfg.early_stopping_patience_disc21:
                print(
                    "Early stopping DISC21 training: no improvement for"
                    f" {cfg.early_stopping_patience_disc21} epoch(s)."
                )
                break



if should_train_ndec:
    did_train = True

    best_ndec_loss = float("inf")
    ndec_bad_epochs = 0

    for epoch in range(cfg.num_epochs_ndec):

        ced_model.train()
        total_loss = 0.0

        progress = tqdm(
            ndec_neg_pair_loader,
            desc=f"NDEC Epoch {epoch + 1}/{cfg.num_epochs_ndec}",
        )

        for img_a, img_b, _, _ in progress:

            img_a = img_a.to(device)
            img_b = img_b.to(device)

            optimizer.zero_grad()

            v_former, _ = ced_model.encode_images(img_a)
            v_latter, _ = ced_model.encode_images(img_b)

            loss_asl = asl_loss(
                v_former=v_former,
                v_latter=v_latter,
                lambda_mtr=cfg.lambda_asl_mtr,
            )

            loss_asl.backward()
            torch.nn.utils.clip_grad_norm_(ced_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_asl.item()

            progress.set_postfix({"asl_loss": f"{loss_asl.item():.4f}"})

        avg_asl = total_loss / len(ndec_neg_pair_loader)
        print(f"NDEC Epoch {epoch + 1}: avg ASL loss = {avg_asl:.4f}")

        if avg_asl + cfg.early_stopping_min_delta_ndec < best_ndec_loss:
            best_ndec_loss = avg_asl
            ndec_bad_epochs = 0
        else:
            ndec_bad_epochs += 1
            if ndec_bad_epochs >= cfg.early_stopping_patience_ndec:
                print(
                    "Early stopping NDEC training: no improvement for"
                    f" {cfg.early_stopping_patience_ndec} epoch(s)."
                )
                break

else:

    if not should_train_disc21:
        print("Training loops are disabled by default—enable the flags above to run them.")

if did_train:
    save_checkpoint(ced_model, optimizer, str(cfg.checkpoint_path))
    print(f"Saved final checkpoint to {cfg.checkpoint_path}")

# %% [markdown]
# # Evaluation on DISC21 Dev/Test (Retrieval)
# 
# Precompute reference embeddings, rank candidates via cosine similarity, and report retrieval metrics such as mAP and Recall@K. Use `run_disc21_retrieval_pipeline` to execute the full encode + score routine for either dev or test splits.

# %%
def compute_descriptors_for_loader(loader: DataLoader, model: CEDModel, save_path_prefix: str):
    """Encode a loader and optionally persist descriptors for fast reuse."""

    prefix_path = Path(save_path_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    all_vecs: List[torch.Tensor] = []
    all_ids: List[str] = []
    model.eval()

    with torch.no_grad():
        for imgs, sample_ids in tqdm(loader, desc=f"Encoding {prefix_path.stem}"):
            descriptors, _ = model.encode_images(imgs.to(device))
            all_vecs.append(descriptors.cpu())
            all_ids.extend([str(sid) for sid in sample_ids])

    all_vecs = torch.cat(all_vecs, dim=0)
    torch.save(all_vecs, f"{save_path_prefix}_embeddings.pt")
    np.save(f"{save_path_prefix}_ids.npy", np.array(all_ids))

    return all_vecs, all_ids


def load_ground_truth_map(split: str, root: Path) -> Dict[str, List[str]]:
    """Read DISC21 ground truth for a split via the helper and build a query->refs map."""

    df = load_groundtruth(split=split, root=root)
    gt: Dict[str, List[str]] = {}
    for row in df.itertuples():
        qid = str(getattr(row, "query_id"))
        rid = str(getattr(row, "reference_id"))
        if rid and rid != "nan":
            gt.setdefault(qid, []).append(rid)

    return gt


def cosine_similarity(queries: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    queries = F.normalize(queries, dim=-1)
    refs = F.normalize(refs, dim=-1)

    return queries @ refs.t()

def evaluate_retrieval(
    query_vecs: torch.Tensor,
    query_ids: List[str],
    ref_vecs: torch.Tensor,
    ref_ids: List[str],
    gt_map: Dict[str, List[str]],
    topk: int = 10,
 ) -> Dict[str, float]:
    """Compute Recall@K and mAP@K using cosine similarity rankings."""

    # Normalize IDs to compare: drop extensions
    query_ids_norm = [Path(qid).stem for qid in query_ids]
    ref_ids_norm = [Path(rid).stem for rid in ref_ids]

    # Build lookup using normalized IDs
    ref_ids_arr = np.array(ref_ids)
    ref_ids_norm_arr = np.array(ref_ids_norm)

    ref_vecs = ref_vecs.to(device)
    query_vecs = query_vecs.to(device)
    sims = cosine_similarity(query_vecs, ref_vecs)

    k = min(topk, ref_vecs.size(0))
    topk_idx = sims.topk(k, dim=1).indices.cpu()

    num_evaluable = 0
    recall_hits = 0
    map_sum = 0.0

    for q_idx, qid_norm in enumerate(query_ids_norm):
        gt_ids = gt_map.get(qid_norm, [])
        if not gt_ids:
            continue

        num_evaluable += 1
        retrieved_norm = ref_ids_norm_arr[topk_idx[q_idx].numpy()]
        gt_set = set(gt_ids)

        # Recall@K
        if any(rid in gt_set for rid in retrieved_norm[:k]):
            recall_hits += 1

        # mAP@K
        hits = 0
        ap = 0.0
        for rank, ref_id_norm in enumerate(retrieved_norm[:k], start=1):
            if ref_id_norm in gt_set:
                hits += 1
                ap += hits / rank
        if hits > 0:
            ap /= len(gt_ids)
        map_sum += ap

    if num_evaluable == 0:
        return {f"Recall@{k}": 0.0, f"mAP@{k}": 0.0}

    return {
        f"Recall@{k}": recall_hits / num_evaluable,
        f"mAP@{k}": map_sum / num_evaluable,
    }


def run_disc21_retrieval_pipeline(split: str = "dev", topk: int = 10):
    """Encode DISC21 references/queries and report retrieval metrics for a split."""

    split = split.lower()
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")

    query_loader = dev_query_loader if split == "dev" else test_query_loader
    ref_vecs, ref_ids = compute_descriptors_for_loader(
        ref_loader, ced_model, "artifacts/disc21_ref"
    )
    query_vecs, query_ids = compute_descriptors_for_loader(
        query_loader, ced_model, f"artifacts/disc21_{split}_query"
    )
    gt_map = load_ground_truth_map(split=split, root=disc_cfg.root)
    metrics = evaluate_retrieval(query_vecs, query_ids, ref_vecs, ref_ids, gt_map, topk=topk)
    print(f"DISC21 {split} metrics: {metrics}")
    return metrics

# %% [markdown]
# ## NDEC Copy-Edit Classifier Evaluation
# 
# Use NDEC positive and negative pairs to evaluate the copy-edit classifier head.
# Positive pairs = true matches from the NDEC ground-truth CSV.
# Negative pairs = explicit mismatches from the `negative_pair/` folders.

# %%
def evaluate_ndec_copy_edit_classifier(
    model: CEDModel,
    pos_loader: DataLoader,
    neg_loader: DataLoader,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate the copy-edit classifier head on NDEC using:
      - pos_loader: (query_img, ref_img, query_id, ref_id) with true matches
      - neg_loader: (img_a, img_b, name_a, name_b) with explicit mismatches

    Returns:
      {
        "accuracy": ...,
        "average_precision": ...
      }
    """
    model.eval()

    all_scores: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        # --- Positive pairs → label 1 ---
        for q_img, r_img, _, _ in tqdm(pos_loader, desc="Scoring NDEC positives"):
            q_img = q_img.to(device)
            r_img = r_img.to(device)

            logits, _, _ = model.score_pair(q_img, r_img)
            scores = torch.sigmoid(logits)

            all_scores.append(scores.cpu())
            all_labels.append(torch.ones_like(scores.cpu()))

        # --- Negative pairs → label 0 ---
        for img_a, img_b, _, _ in tqdm(neg_loader, desc="Scoring NDEC negatives"):
            img_a = img_a.to(device)
            img_b = img_b.to(device)

            logits, _, _ = model.score_pair(img_a, img_b)
            scores = torch.sigmoid(logits)

            all_scores.append(scores.cpu())
            all_labels.append(torch.zeros_like(scores.cpu()))

    if not all_scores:
        return {"accuracy": 0.0, "average_precision": 0.0}

    # Concatenate all batches
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    # Accuracy at fixed threshold
    preds = (scores >= threshold).astype(np.float32)
    accuracy = float((preds == labels).mean())

    # Simple AP computation (ranking by score)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    cum_pos = np.cumsum(sorted_labels)
    precision = cum_pos / (np.arange(len(sorted_labels)) + 1)

    # AP = average precision over the positions where label == 1
    pos_count = max(sorted_labels.sum(), 1.0)
    average_precision = float((precision * sorted_labels).sum() / pos_count)

    return {
        "accuracy": accuracy,
        "average_precision": average_precision,
    }


# Example usage (after training)
# ndec_cls_metrics = evaluate_ndec_copy_edit_classifier(
#     ced_model,
#     ndec_pos_pair_loader,
#     ndec_neg_pair_loader,
# )
# print("NDEC classifier metrics:", ndec_cls_metrics)

# %% [markdown]
# ## NDEC Retrieval Evaluation
# 
# Reuse the same descriptor helpers on the NDEC query/reference sets, then score with the shared retrieval evaluator to report mAP/Recall.

# %%
def load_ndec_groundtruth_map(
    root: Path, csv_name: str = "public_ground_truth_h5.csv", drop_missing: bool = False
) -> Dict[str, List[str]]:
    df = load_ndec_groundtruth(root=root, csv_name=csv_name, drop_missing=drop_missing)
    gt: Dict[str, List[str]] = {}
    for row in df.itertuples():
        qid = str(getattr(row, "query_id"))
        rid = str(getattr(row, "reference_id"))
        gt.setdefault(qid, []).append(rid)
    return gt



def run_ndec_retrieval_pipeline(topk: int = 10):
    """Compute NDEC retrieval metrics via the shared evaluator."""

    ref_vecs, ref_ids = compute_descriptors_for_loader(
        ndec_ref_loader, ced_model, "artifacts/ndec_ref"
)
    query_vecs, query_ids = compute_descriptors_for_loader(
        ndec_query_loader, ced_model, "artifacts/ndec_query"
)
    gt_map = load_ndec_groundtruth_map(ndec_cfg.root, drop_missing=True)
    metrics = evaluate_retrieval(
        query_vecs, query_ids, ref_vecs, ref_ids, gt_map, topk=topk
)
    print(f"NDEC metrics: {metrics}")
    return metrics



# Example usage (disabled by default)

# run_ndec_retrieval_pipeline(topk=10)

# %% [markdown]
# # (Optional) Save/Load Checkpoints & Precomputed Embeddings
# 
# Utility helpers to persist model weights and cached descriptors for faster experimentation across sessions.

# %%
def save_checkpoint(model: CEDModel, optimizer: torch.optim.Optimizer, path: str):
    ckpt = {

        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.__dict__,

    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"Saved checkpoint to {path}")





def load_checkpoint(model: CEDModel, optimizer: torch.optim.Optimizer, path: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    print(f"Restored checkpoint from {path}")


