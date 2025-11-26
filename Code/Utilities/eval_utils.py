# eval_utils.py
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# -----------------------------
# Descriptor computation (with optional TTA)
# -----------------------------

def compute_descriptors_for_loader(
    loader: DataLoader,
    model,
    save_path_prefix: str,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Encode a loader and optionally persist descriptors for fast reuse.

    Assumes:
      - loader yields (images, sample_ids)
      - model implements encode_images(images) -> (descriptors, aux)
      - model parameters live on the correct device (CPU/GPU)
    """
    prefix_path = Path(save_path_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    all_vecs: List[torch.Tensor] = []
    all_ids: List[str] = []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for imgs, sample_ids in tqdm(loader, desc=f"Encoding {prefix_path.stem}"):
            imgs = imgs.to(device)
            descriptors, _ = model.encode_images(imgs)
            all_vecs.append(descriptors.cpu())
            all_ids.extend([str(sid) for sid in sample_ids])

    if not all_vecs:
        return torch.empty(0), []

    all_vecs = torch.cat(all_vecs, dim=0)
    torch.save(all_vecs, f"{save_path_prefix}_embeddings.pt")
    np.save(f"{save_path_prefix}_ids.npy", np.array(all_ids))

    return all_vecs, all_ids


def compute_descriptors_for_loader_tta(
    loader: DataLoader,
    model,
    save_path_prefix: str,
    num_views: int = 1,
    aug=None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Encode loader with test-time augmentation.
    If num_views > 1, average descriptors over multiple stochastic views.

    Args:
      loader: yields (images, sample_ids)
      model: has encode_images(images) -> (descriptors, aux)
      num_views: number of stochastic transformations per image
      aug: a torchvision transform applied per view on CPU tensors
    """
    if num_views < 1:
        num_views = 1

    prefix_path = Path(save_path_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    all_vecs: List[torch.Tensor] = []
    all_ids: List[str] = []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for imgs, sample_ids in tqdm(loader, desc=f"TTA x{num_views}: {prefix_path.stem}"):
            imgs = imgs.to(device)

            if num_views == 1 or aug is None:
                desc, _ = model.encode_images(imgs)
            else:
                views = []
                imgs_cpu = imgs.detach().cpu()
                for _ in range(num_views):
                    aug_imgs = torch.stack([aug(img) for img in imgs_cpu]).to(device)
                    desc_v, _ = model.encode_images(aug_imgs)
                    views.append(desc_v)
                desc = torch.stack(views, dim=0).mean(dim=0)

            all_vecs.append(desc.cpu())
            all_ids.extend([str(sid) for sid in sample_ids])

    if not all_vecs:
        return torch.empty(0), []

    all_vecs = torch.cat(all_vecs, dim=0)
    torch.save(all_vecs, f"{save_path_prefix}_tta{num_views}_embeddings.pt")
    np.save(f"{save_path_prefix}_tta{num_views}_ids.npy", np.array(all_ids))
    return all_vecs, all_ids


# -----------------------------
# Metrics: cosine sim, µAP, R@P90, Recall@K, mAP@K
# -----------------------------

def cosine_similarity(queries: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between query and ref embeddings.
    Tensors may live on any device; computation stays on that device.
    """
    queries = F.normalize(queries, dim=-1)
    refs = F.normalize(refs, dim=-1)
    return queries @ refs.t()


def compute_muap_and_rp90(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Global micro-AP and R@P90 from flattened pair scores.

    all_scores: shape [N_pairs]
    all_labels: shape [N_pairs], in {0,1}
    """
    assert all_scores.shape == all_labels.shape

    # sort by score descending
    order = np.argsort(all_scores)[::-1]
    sorted_labels = all_labels[order]

    cum_pos = np.cumsum(sorted_labels)
    total_pos = float(sorted_labels.sum())
    if total_pos == 0:
        return {"muAP": 0.0, "R@P90": 0.0}

    ranks = np.arange(1, len(sorted_labels) + 1)
    precision = cum_pos / ranks
    recall = cum_pos / total_pos

    # µAP: average precision over all positive instances
    muap = float((precision * sorted_labels).sum() / total_pos)

    # R@P90: max recall where precision >= 0.9
    mask = precision >= 0.9
    rp90 = float(recall[mask].max()) if mask.any() else 0.0

    return {"muAP": muap, "R@P90": rp90}


def evaluate_retrieval(
    query_vecs: torch.Tensor,
    query_ids: List[str],
    ref_vecs: torch.Tensor,
    ref_ids: List[str],
    gt_map: Dict[str, List[str]],
    topk_list: List[int] = [10],
) -> Dict[str, float]:
    """
    Compute:
      - Recall@k for each k in topk_list
      - mAP@k for each k in topk_list (per-query AP)
      - global µAP and R@P90 over all (q, r) pairs

    Arguments:
      query_vecs: (Q, D) tensor
      query_ids: list of Q identifiers
      ref_vecs: (R, D) tensor
      ref_ids: list of R identifiers
      gt_map: dict query_id -> list of true reference_ids (same naming convention as ref_ids)
      topk_list: list of K values (e.g. [1,2,5,10,15,20])
    """
    # Normalize IDs to compare: drop extensions
    query_ids_norm = [Path(qid).stem for qid in query_ids]
    ref_ids_norm = [Path(rid).stem for rid in ref_ids]

    ref_ids_norm_arr = np.array(ref_ids_norm)

    ref_vecs = ref_vecs.to(query_vecs.device)
    sims = cosine_similarity(query_vecs, ref_vecs)  # [Q, R]

    max_k = min(max(topk_list), ref_vecs.size(0))
    topk_idx = sims.topk(max_k, dim=1).indices.cpu()  # [Q, max_k]

    num_evaluable = 0
    recall_hits = {k: 0 for k in topk_list}
    map_sums = {k: 0.0 for k in topk_list}

    # For global µAP/R@P90
    global_scores: List[float] = []
    global_labels: List[int] = []

    for q_idx, qid_norm in enumerate(query_ids_norm):
        gt_ids = gt_map.get(qid_norm, [])
        gt_set = set(gt_ids)

        retrieved_norm = ref_ids_norm_arr[topk_idx[q_idx].numpy()]

        # per-query metrics only if we have positives
        if gt_ids:
            num_evaluable += 1

            for k in topk_list:
                k_eff = min(k, max_k)
                topk_refs = retrieved_norm[:k_eff]

                # Recall@k
                if any(rid in gt_set for rid in topk_refs):
                    recall_hits[k] += 1

                # mAP@k
                hits = 0
                ap = 0.0
                for rank, ref_id_norm in enumerate(topk_refs, start=1):
                    if ref_id_norm in gt_set:
                        hits += 1
                        ap += hits / rank
                if hits > 0:
                    ap /= len(gt_ids)
                map_sums[k] += ap

        # global scores/labels for µAP/R@P90: consider all refs
        q_scores = sims[q_idx].detach().cpu().numpy()
        for r_idx, score in enumerate(q_scores):
            rid_norm = ref_ids_norm[r_idx]
            label = 1 if rid_norm in gt_set else 0
            global_scores.append(float(score))
            global_labels.append(label)

    metrics: Dict[str, float] = {}
    if num_evaluable == 0:
        for k in topk_list:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"mAP@{k}"] = 0.0
        metrics["muAP"] = 0.0
        metrics["R@P90"] = 0.0
        return metrics

    for k in topk_list:
        metrics[f"Recall@{k}"] = recall_hits[k] / num_evaluable
        metrics[f"mAP@{k}"] = map_sums[k] / num_evaluable

    global_scores_arr = np.array(global_scores)
    global_labels_arr = np.array(global_labels)
    global_metrics = compute_muap_and_rp90(global_scores_arr, global_labels_arr)
    metrics.update(global_metrics)

    return metrics


# -----------------------------
# Runtime benchmark
# -----------------------------

def benchmark_inference(
    model,
    loader: DataLoader,
    num_batches: int = 10,
    warmup: int = 2,
) -> float:
    """
    Return average milliseconds per image for model.encode_images() on the given loader.
    Uses the first `num_batches` batches (after `warmup` warmup batches).

    Assumes:
      - loader yields (images, ids)
      - model implements encode_images(images)
    """
    model.eval()
    device = next(model.parameters()).device

    n_images = 0
    total_time = 0.0

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(loader):
            if batch_idx >= warmup + num_batches:
                break

            imgs = imgs.to(device)
            # warmup
            if batch_idx < warmup:
                _ = model.encode_images(imgs)
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

            if device.type == "cuda":
                t0.record()
            else:
                import time
                t0 = time.perf_counter()

            _ = model.encode_images(imgs)

            if device.type == "cuda":
                t1.record()
                torch.cuda.synchronize()
                elapsed_ms = t0.elapsed_time(t1)
            else:
                import time
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

            total_time += elapsed_ms
            n_images += imgs.size(0)

    if n_images == 0:
        return 0.0

    ms_per_image = total_time / n_images
    print(f"Avg encode_images time: {ms_per_image:.2f} ms/image")
    return ms_per_image
