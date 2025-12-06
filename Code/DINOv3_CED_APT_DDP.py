
import os
import sys
import json
import gc
import inspect
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from torch import amp

from huggingface_hub import login
from dotenv import load_dotenv

import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as V

# ======================================================================
# Import utilities
# ======================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
UTILITIES_DIR = SCRIPT_DIR / "Utilities"

if str(UTILITIES_DIR) not in sys.path:
    sys.path.append(str(UTILITIES_DIR))

# Model architecture
from ced_models import DinoV3Backbone, CEDModel

# APT integration
from apt_ced_wrapper import wrap_ced_with_apt

# Loss functions
from ced_losses import (
    nt_xent_loss,
    kozachenko_leonenko_loss,
    multi_similarity_loss,
    asl_loss,
    bce_loss,
)

# Distributed training
from ced_distributed import setup_distributed, cleanup_distributed

# Checkpoint management
from ced_checkpoints import CheckpointManager, TrainingState

# Evaluation
from ced_evaluation import (
    compute_descriptors_for_loader,
    evaluate_retrieval,
    ced_two_stage_eval,
    build_ref_index_map,
    normalize_id,
    compute_muap_and_rp90,
)

# Data loaders
from disc21_loader import (
    Disc21DataConfig,
    build_transforms,
    get_train_dataset,
    get_reference_dataset,
    get_query_dataset,
    load_groundtruth,
)

from ndec_loader import (
    NdecDataConfig,
    build_default_loaders as build_ndec_loaders,
    load_groundtruth as load_ndec_groundtruth,
)


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class ExperimentConfig:
    """Configuration for CEDetector training experiment."""
    disc21_root: Path = WORKSPACE_ROOT / "DISC21"
    ndec_root: Path = WORKSPACE_ROOT / "NDEC"
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    # Image sizes (Paper: 224 train, 384 eval)
    img_size_train: int = 224
    img_size_eval: int = 384

    # APT configuration
    use_apt: bool = True  # Enable Adaptive Patch Transformers
    apt_num_scales: int = 3  # Number of patch scales (16, 32, 64)
    apt_thresholds: List[float] = None  # Auto-set to [5.5, 4.0]
    apt_scorer_method: str = "entropy"  # 'entropy', 'laplacian', or 'upsampling'
    apt_analyze_only: bool = True  # Only analyze, don't apply (for compatibility)

    # Batch sizes (Paper: 64) - REDUCED TO FIT IN 24GB GPU MEMORY
    batch_size_train: int = 32  # Reduced from 64 to prevent OOM
    batch_size_eval: int = 32   # Reduced from 64 to prevent OOM
    batch_size_pairs: int = 32  # Reduced from 64 to prevent OOM

    num_workers: int = 8  # Increased for faster data loading

    # Gradient accumulation for effective larger batch
    gradient_accumulation_steps: int = 2  # Effective batch = 64 * 2 * 3 GPUs = 384

    # CED augmentation config
    use_ced_augmentations: bool = True
    ced_min_ops: int = 2
    ced_max_ops: int = 6
    ced_aug_seed: Optional[int] = None

    # CED evaluation config
    use_ced_two_stage_eval: bool = True
    ced_k_candidates_per_patch: int = 10  # Paper uses k=10
    skip_eval_during_training: bool = True  # Skip eval, only eval at end (FAST MODE)

    # Learning rates (Paper: 2e-5)
    lr_backbone: float = 2e-5
    lr_head: float = 2e-5
    weight_decay: float = 1e-4

    # Loss hyperparameters (exact paper values)
    temperature_ntxent: float = 0.025  # τ (Eq 3-4)
    lambda_kl: float = 0.5              # λ (Eq 5-6)
    lambda_local: float = 1.0           # L_MSL weight
    lambda_bce: float = 1.0             # L_BCE weight
    lambda_asl_mtr: float = 1.0         # ASL metric weight

    # Training epochs (Paper: 30 on ISC) - REDUCED FOR FAST TRAINING
    num_epochs_disc21: int = 10
    num_epochs_ndec: int = 4

    # Early stopping (DISC21) - REDUCED FOR FASTER TRAINING
    early_stopping_patience_disc21: int = 2
    early_stopping_min_delta_disc21: float = 1e-4
    min_epochs_disc21: int = 3

    # Early stopping (NDEC) - REDUCED FOR FASTER TRAINING
    early_stopping_patience_ndec: int = 1
    early_stopping_min_delta_ndec: float = 1e-4
    min_epochs_ndec: int = 2

    # Paths
    checkpoint_path: Path = Path("artifacts/checkpoints/ced_model_apt_ddp.pt")
    encoding_cache_root: Path = Path("artifacts/sinov3_ced_encoding")
    encoding_chunk_size: int = 2048  # Increased for faster encoding
    eval_query_chunk_size: int = 512  # Increased for faster eval
    eval_ref_chunk_size: int = 16384  # Increased for faster eval
    eval_global_pairs_per_query: int = 256  # Reduced to speed up eval


# ======================================================================
# Training utilities
# ======================================================================
def make_positive_negative_pairs(
    batch_imgs: torch.Tensor,
    copy_edit_aug: Optional[V.Compose] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create positive and negative pairs from a batch of images.

    Paper training: 50% positive pairs (x, x'), 50% negative pairs (x, x'')
    where x' is augmented from x, and x'' is augmented from a different image.

    Args:
        batch_imgs: Input batch [B, C, H, W]
        copy_edit_aug: Optional torchvision transform for on-the-fly augmentation

    Returns:
        Tuple of (anchor, positive, negative) tensors
    """
    B = batch_imgs.size(0)
    imgs_cpu = batch_imgs.detach().cpu()
    device = batch_imgs.device
    x_anchor = batch_imgs.to(device, non_blocking=True)

    # Create positive pairs (same image, different augmentation)
    if copy_edit_aug is not None:
        x_pos = torch.stack([copy_edit_aug(img) for img in imgs_cpu]).to(device, non_blocking=True)
    else:
        x_pos = x_anchor.clone()

    # Create negative pairs (different images, augmented)
    perm = torch.randperm(B)
    if copy_edit_aug is not None:
        x_neg = torch.stack([copy_edit_aug(imgs_cpu[i]) for i in perm]).to(device, non_blocking=True)
    else:
        x_neg = x_anchor[perm]

    return x_anchor, x_pos, x_neg


def load_disc21_groundtruth_map(split: str, root: Path) -> Dict[str, List[str]]:
    """Load DISC21 ground truth as a mapping dict."""
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
    """Load NDEC ground truth as a mapping dict."""
    df = load_ndec_groundtruth(root=root, csv_name=csv_name, drop_missing=drop_missing)
    gt: Dict[str, List[str]] = {}
    for row in df.itertuples():
        qid = str(getattr(row, "query_id"))
        rid = str(getattr(row, "reference_id"))
        gt.setdefault(qid, []).append(rid)
    return gt


# ======================================================================
# Main training loop
# ======================================================================
def main():
    # ------------------ Basic setup ------------------
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

    # Create on-the-fly augmentation for positive/negative pair generation
    copy_edit_aug = V.Compose(
        [
            V.RandomResizedCrop(cfg.img_size_train, scale=(0.6, 1.0)),
            V.RandomHorizontalFlip(),
            V.ColorJitter(0.2, 0.2, 0.2, 0.1),
            V.RandomGrayscale(p=0.1),
        ]
    )

    # ------------------ Datasets + loaders ------------------
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

    # Configure num_workers
    cpu_count = os.cpu_count() or 8
    num_workers = min(cfg.num_workers, max(2, cpu_count // max(world_size, 1)))

    if rank == 0:
        print(f"[DataLoader] Using num_workers={num_workers}, prefetch_factor=2, persistent_workers=False")
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
    ref_ds = get_reference_dataset(root=disc_cfg.root, transform=eval_tfms)
    dev_queries_ds = get_query_dataset("dev", root=disc_cfg.root, transform=eval_tfms)
    test_queries_ds = get_query_dataset("test", root=disc_cfg.root, transform=eval_tfms)

    ref_sampler = DistributedSampler(ref_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dev_q_sampler = DistributedSampler(dev_queries_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_q_sampler = DistributedSampler(test_queries_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    ref_loader = DataLoader(ref_ds, batch_size=cfg.batch_size_eval, sampler=ref_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2)
    dev_query_loader = DataLoader(dev_queries_ds, batch_size=cfg.batch_size_eval, sampler=dev_q_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2)
    test_query_loader = DataLoader(test_queries_ds, batch_size=cfg.batch_size_eval, sampler=test_q_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2)

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

    ndec_query_loader, ndec_ref_loader, ndec_pos_pair_loader, ndec_neg_pair_loader = build_ndec_loaders(ndec_cfg)

    ndec_neg_dataset = ndec_neg_pair_loader.dataset
    ndec_neg_sampler = DistributedSampler(ndec_neg_dataset, num_replicas=world_size, rank=rank, shuffle=True)
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

    # ------------------ Model + optimizer (wrapped in DDP) ------------------
    backbone = DinoV3Backbone(model_name=cfg.model_name)
    backbone.model.to(device)
    ced_model = CEDModel(backbone=backbone, dim=backbone.hidden_size).to(device)

    # Wrap with APT if enabled
    if cfg.use_apt:
        ced_model = wrap_ced_with_apt(
            ced_model,
            enable_apt=True,
            num_scales=cfg.apt_num_scales,
            thresholds=cfg.apt_thresholds,
            scorer_method=cfg.apt_scorer_method,
        )
        if rank == 0:
            print(f"[APT] Wrapped model with Adaptive Patch Transformers")
            print(f"[APT]   Num scales: {cfg.apt_num_scales}")
            print(f"[APT]   Thresholds: {cfg.apt_thresholds or [5.5, 4.0]}")
            print(f"[APT]   Scorer: {cfg.apt_scorer_method}")
            print(f"[APT]   Analysis mode: {cfg.apt_analyze_only}")
            sys.stdout.flush()

    from torch.nn.parallel import DistributedDataParallel as DDP
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

    # ------------------ Checkpoint management ------------------
    ckpt_manager = CheckpointManager(Path(checkpoint_path), rank)
    training_state = ckpt_manager.load(ddp_model.module, optimizer)
    if training_state is None:
        training_state = TrainingState()

    start_epoch_disc21 = training_state.epoch_disc21
    start_epoch_ndec = training_state.epoch_ndec
    best_disc21_loss = training_state.best_disc21_loss
    best_ndec_loss = training_state.best_ndec_loss
    disc21_bad_epochs = training_state.disc21_bad_epochs
    ndec_bad_epochs = training_state.ndec_bad_epochs
    disc21_losses: List[float] = training_state.disc21_losses
    ndec_losses: List[float] = training_state.ndec_losses
    disc21_accuracies: List[float] = training_state.disc21_accuracies

    # ------------------ Training flags ------------------
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
            accumulation_counter = 0

            for imgs, _ in progress:
                imgs = imgs.to(device, non_blocking=True)
                x_anchor, x_pos, x_neg = make_positive_negative_pairs(imgs, copy_edit_aug=copy_edit_aug)

                # Only zero gradients at the start of accumulation cycle
                if accumulation_counter == 0:
                    optimizer.zero_grad(set_to_none=True)

                # Forward + loss with AMP autocast
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    # Encode anchor & positive
                    v_anchor, feats_anchor = ddp_model.module.encode_images(x_anchor, return_local=True)
                    v_pos, feats_pos = ddp_model.module.encode_images(x_pos, return_local=True)

                    # Use CLS-only projections z for contrastive + KL (Paper Eq 6)
                    z_anchor = feats_anchor["cls_global"]
                    z_pos = feats_pos["cls_global"]
                    z_all = torch.cat([z_anchor, z_pos], dim=0)

                    # L_SimCLR (NT-Xent, Eq 3 & 4)
                    loss_simclr = nt_xent_loss(z_anchor, z_pos, temperature=cfg.temperature_ntxent)

                    # L_KL (Kozachenko–Leonenko entropy term, Eq 5)
                    loss_kl = kozachenko_leonenko_loss(z_all)

                    # L_contrast = L_SimCLR + λ L_KL  (Eq 6)
                    loss_contrast = loss_simclr + cfg.lambda_kl * loss_kl

                    # Local multi-similarity loss (L_MSL, Eq 7)
                    local_anchor = feats_anchor.get("local_descriptor")
                    local_pos = feats_pos.get("local_descriptor")
                    local_embeddings = torch.cat([local_anchor, local_pos], dim=0)
                    batch_ids = torch.arange(local_anchor.size(0), device=local_anchor.device)
                    local_labels = torch.cat([batch_ids, batch_ids], dim=0)
                    loss_local = multi_similarity_loss(local_embeddings, local_labels)

                    # BCE on classifier (L_BCE, Eq 9)
                    # Keep tokens in FP16 for memory efficiency
                    anchor_tokens = feats_anchor["patch_tokens_flat"].detach()
                    pos_tokens = feats_pos["patch_tokens_flat"].detach()

                    # Get negative features (keep in no_grad to save memory)
                    with torch.no_grad():
                        neg_feats = ddp_model.module.backbone.get_features_from_images(x_neg)
                    neg_tokens = neg_feats["patch_tokens_flat"].detach()

                    # Run classifier in FP16 to save memory
                    logits_pos = ddp_model.module.classifier(anchor_tokens.half(), pos_tokens.half()).squeeze(-1)
                    logits_neg = ddp_model.module.classifier(anchor_tokens.half(), neg_tokens.half()).squeeze(-1)

                    labels_pos = torch.ones_like(logits_pos)
                    labels_neg = torch.zeros_like(logits_neg)
                    logits = torch.cat([logits_pos, logits_neg], dim=0)
                    labels = torch.cat([labels_pos, labels_neg], dim=0)

                    loss_bce = bce_loss(logits.float(), labels.float())

                    # Total loss (Eq 8): L = L_contrast + L_MSL + L_BCE
                    loss = loss_contrast + cfg.lambda_local * loss_local + cfg.lambda_bce * loss_bce
                    # Scale loss by accumulation steps for proper gradient averaging
                    loss = loss / cfg.gradient_accumulation_steps

                # NaN / inf guard - synchronize across all ranks to prevent deadlock
                import torch.distributed as dist
                is_finite = torch.isfinite(loss)
                finite_tensor = torch.tensor(1 if is_finite else 0, device=device)
                dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)  # 0 if any rank has non-finite

                if finite_tensor.item() == 0:
                    if rank == 0:
                        print("[WARN][DISC21] Non-finite loss encountered on one or more ranks; skipping batch on all ranks.")
                    accumulation_counter = 0
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Backward with AMP (accumulate gradients)
                scaler.scale(loss).backward()

                accumulation_counter += 1

                # Only update weights after accumulating gradients
                if accumulation_counter >= cfg.gradient_accumulation_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    accumulation_counter = 0

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

                # Clear cached memory periodically to prevent fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Average loss across processes
            import torch.distributed as dist
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
                disc21_accuracies.append(epoch_acc_global)
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

                training_state.epoch_disc21 = current_epoch
                training_state.best_disc21_loss = best_disc21_loss
                training_state.disc21_bad_epochs = disc21_bad_epochs
                training_state.disc21_losses = disc21_losses
                training_state.disc21_accuracies = disc21_accuracies

                ckpt_manager.save(ddp_model.module, optimizer, training_state)

                can_stop_disc21 = current_epoch >= cfg.min_epochs_disc21
                if not improved and disc21_bad_epochs >= cfg.early_stopping_patience_disc21 and can_stop_disc21:
                    print(f"[DISC21] Early stopping after {cfg.early_stopping_patience_disc21} bad epoch(s).")
                    stop_now = True

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
                progress = tqdm(ndec_neg_train_loader, desc=f"[NDEC][Epoch {current_epoch}/{cfg.num_epochs_ndec}]")
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

                # NaN / inf guard - synchronize across all ranks to prevent deadlock
                import torch.distributed as dist
                is_finite = torch.isfinite(loss_asl)
                finite_tensor = torch.tensor(1 if is_finite else 0, device=device)
                dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)  # 0 if any rank has non-finite

                if finite_tensor.item() == 0:
                    if rank == 0:
                        print("[WARN][NDEC] Non-finite ASL loss encountered on one or more ranks; skipping batch on all ranks.")
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

            import torch.distributed as dist
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

                training_state.epoch_ndec = current_epoch
                training_state.best_ndec_loss = best_ndec_loss
                training_state.ndec_bad_epochs = ndec_bad_epochs
                training_state.ndec_losses = ndec_losses

                ckpt_manager.save(ddp_model.module, optimizer, training_state)

                can_stop_ndec = current_epoch >= cfg.min_epochs_ndec
                if not improved and ndec_bad_epochs >= cfg.early_stopping_patience_ndec and can_stop_ndec:
                    print(f"[NDEC] Early stopping after {cfg.early_stopping_patience_ndec} bad epoch(s).")
                    stop_now_ndec = True

            stop_tensor_ndec = torch.tensor(1 if stop_now_ndec else 0, device=device)
            dist.broadcast(stop_tensor_ndec, src=0)
            if stop_tensor_ndec.item() == 1:
                break

    # ------------------ Dump training curves to JSON (rank 0) ------------------
    if rank == 0:
        log_dir = (script_dir / "artifacts/logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        curves_path = log_dir / "training_curves_apt.json"

        training_metrics = {
            "disc21_losses": disc21_losses,
            "disc21_accuracies": disc21_accuracies,
            "ndec_losses": ndec_losses,
        }

        with curves_path.open("w") as f:
            json.dump(training_metrics, f, indent=2)
        print(f"[Logs] Wrote training curves to {curves_path}")
        print(f"[Logs]   DISC21 epochs: {len(disc21_losses)}, NDEC epochs: {len(ndec_losses)}")

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
    import torch.distributed as dist
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

            print(f"       Merged {relative_artifact_name}: {len(merged_ids)} -> {len(final_ids)} unique items.")
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

    # Print APT statistics if enabled
    if cfg.use_apt and rank == 0:
        print("\n" + "="*70)
        print("APT Token Reduction Analysis")
        print("="*70)
        eval_model.print_stats()

    cleanup_distributed()


if __name__ == "__main__":
    main()