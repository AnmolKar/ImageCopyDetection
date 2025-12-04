"""
CED Evaluation Library
======================

Unified evaluation library for Copy-Edit Detection combining:
1. Descriptor computation and caching
2. Two-stage CED evaluation (patch + classifier)
3. Descriptor-only baseline evaluation
4. Metrics computation (µAP, R@P90, Recall@K, mAP@K)
5. Runtime benchmarking

"""

import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ============================================================================
# PART 1: Descriptor Computation and Caching
# ============================================================================

def _load_checkpoint_state(state_path: Path, checkpoint_dir: Path) -> Dict:
    """Best-effort load of descriptor checkpoint metadata."""
    if state_path.exists():
        try:
            with state_path.open("r") as f:
                state = json.load(f)
        except json.JSONDecodeError:
            state = {}
    else:
        state = {}

    chunks: List[Dict] = state.get("chunks", [])
    if not chunks:
        for vec_path in sorted(checkpoint_dir.glob("chunk_*_embeddings.pt")):
            chunk_name = vec_path.name.replace("_embeddings.pt", "")
            ids_path = vec_path.with_name(f"{chunk_name}_ids.npy")
            count = 0
            if ids_path.exists():
                try:
                    count = int(np.load(ids_path, allow_pickle=True).shape[0])
                except Exception:
                    count = 0
            chunks.append({"name": chunk_name, "count": count})
    processed = sum(entry.get("count", 0) for entry in chunks)
    state.setdefault("chunks", chunks)
    state.setdefault("processed", processed)
    state.setdefault("next_chunk_idx", len(chunks))
    state.setdefault("completed", False)
    state.setdefault("final_embeddings", "final_embeddings.pt")
    state.setdefault("final_ids", "final_ids.npy")
    return state


def _save_checkpoint_state(state_path: Path, state: Dict) -> None:
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(state_path)


def _synchronize_final_outputs(
    vecs: torch.Tensor,
    ids: List[str],
    save_path_prefix: str,
) -> None:
    torch.save(vecs, f"{save_path_prefix}_embeddings.pt")
    np.save(f"{save_path_prefix}_ids.npy", np.array(ids, dtype=object))


def compute_descriptors_for_loader(
    loader: DataLoader,
    model,
    save_path_prefix: str,
    checkpoint_root: Optional[str] = None,
    resume: bool = False,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Encode a loader and optionally persist descriptors for fast reuse / resume.

    Assumes:
      - loader yields (images, sample_ids)
      - model implements encode_images(images) -> (descriptors, aux)
      - model parameters live on the correct device (CPU/GPU)

    Args:
      checkpoint_root: when provided, descriptors are cached under
        ``checkpoint_root/<prefix_stem>`` using chunked .pt/.npy files plus a
        metadata JSON. Setting ``resume=True`` allows future invocations to skip
        already-encoded samples and to reuse the final tensor directly when
        available.
      chunk_size: approximate number of samples per chunk flush when
        checkpointing is enabled. Smaller values reduce recomputation after a
        crash at the cost of more files.
    """
    prefix_path = Path(save_path_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    use_checkpointing = checkpoint_root is not None
    checkpoint_dir: Optional[Path] = None
    state_path: Optional[Path] = None
    state: Optional[Dict] = None

    if use_checkpointing:
        checkpoint_dir = Path(checkpoint_root).expanduser() / prefix_path.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / "state.json"
        if not resume:
            for existing in checkpoint_dir.glob("*"):
                if existing.is_file() or existing.is_symlink():
                    existing.unlink()
                elif existing.is_dir():
                    shutil.rmtree(existing)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "chunks": [],
                "processed": 0,
                "next_chunk_idx": 0,
                "completed": False,
                "final_embeddings": "final_embeddings.pt",
                "final_ids": "final_ids.npy",
            }
            _save_checkpoint_state(state_path, state)
        else:
            state = _load_checkpoint_state(state_path, checkpoint_dir)

        final_emb_path = checkpoint_dir / state["final_embeddings"]
        final_ids_path = checkpoint_dir / state["final_ids"]
        if resume and state.get("completed") and final_emb_path.exists() and final_ids_path.exists():
            cached_vecs = torch.load(final_emb_path)
            cached_ids = np.load(final_ids_path, allow_pickle=True).tolist()
            _synchronize_final_outputs(cached_vecs, cached_ids, save_path_prefix)
            return cached_vecs, cached_ids

    if not use_checkpointing:
        all_vecs: List[torch.Tensor] = []
        all_ids: List[str] = []
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for imgs, sample_ids in tqdm(loader, desc=f"Encoding {prefix_path.stem}"):
                imgs = imgs.to(device)
                descriptors, _ = model.encode_images(imgs)
                descriptors = descriptors.float().cpu()
                if not torch.isfinite(descriptors).all():
                    raise RuntimeError(
                        f"Non-finite descriptors encountered in compute_descriptors_for_loader "
                        f"for prefix {save_path_prefix}"
                    )
                all_vecs.append(descriptors)
                all_ids.extend([str(sid) for sid in sample_ids])

        if not all_vecs:
            return torch.empty(0), []

        all_vecs = torch.cat(all_vecs, dim=0)
        _synchronize_final_outputs(all_vecs, all_ids, save_path_prefix)
        return all_vecs, all_ids

    assert checkpoint_dir is not None and state is not None and state_path is not None

    chunk_size = max(int(chunk_size), 1)
    already_processed = state.get("processed", 0) if resume else 0
    if resume and already_processed > 0 and not state.get("completed", False):
        print(
            f"[Eval][Cache] {prefix_path.stem}: skipping {already_processed} previously encoded sample(s)."
        )

    device = next(model.parameters()).device
    model.eval()

    sample_buffer: List[torch.Tensor] = []
    id_buffer: List[str] = []
    buffered = 0

    def flush_buffer():
        nonlocal sample_buffer, id_buffer, buffered, state
        if not sample_buffer:
            return
        chunk_tensor = torch.cat(sample_buffer, dim=0)
        chunk_ids = id_buffer.copy()

        chunk_name = f"chunk_{state['next_chunk_idx']:05d}"
        vec_path = checkpoint_dir / f"{chunk_name}_embeddings.pt"
        ids_path = checkpoint_dir / f"{chunk_name}_ids.npy"
        torch.save(chunk_tensor, vec_path)
        np.save(ids_path, np.array(chunk_ids, dtype=object))

        state["chunks"].append({"name": chunk_name, "count": chunk_tensor.size(0)})
        state["next_chunk_idx"] += 1
        state["processed"] = state.get("processed", 0) + chunk_tensor.size(0)
        state["completed"] = False
        _save_checkpoint_state(state_path, state)

        sample_buffer = []
        id_buffer = []
        buffered = 0

    running_target = already_processed
    seen_total = 0

    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Encoding {prefix_path.stem}")
        for imgs, sample_ids in iterator:
            batch_size = len(sample_ids)
            if seen_total + batch_size <= running_target:
                seen_total += batch_size
                continue

            if running_target > seen_total:
                skip = running_target - seen_total
                seen_total += skip
                if skip >= batch_size:
                    continue
                imgs = imgs[skip:]
                sample_ids = sample_ids[skip:]
                batch_size = len(sample_ids)

            imgs = imgs.to(device)
            descriptors, _ = model.encode_images(imgs)
            descriptors = descriptors.float().cpu()
            if not torch.isfinite(descriptors).all():
                raise RuntimeError(
                    f"Non-finite descriptors encountered in compute_descriptors_for_loader "
                    f"for prefix {save_path_prefix}"
                )

            sample_buffer.append(descriptors)
            id_buffer.extend([str(sid) for sid in sample_ids])
            buffered += descriptors.size(0)
            seen_total += descriptors.size(0)

            if buffered >= chunk_size:
                flush_buffer()

        flush_buffer()

    if not state["chunks"]:
        return torch.empty(0), []

    all_vecs_list: List[torch.Tensor] = []
    all_ids: List[str] = []
    for entry in state["chunks"]:
        chunk_name = entry["name"]
        vec_path = checkpoint_dir / f"{chunk_name}_embeddings.pt"
        ids_path = checkpoint_dir / f"{chunk_name}_ids.npy"
        if not vec_path.exists() or not ids_path.exists():
            continue
        all_vecs_list.append(torch.load(vec_path))
        ids = np.load(ids_path, allow_pickle=True).tolist()
        all_ids.extend([str(i) for i in ids])

    if not all_vecs_list:
        return torch.empty(0), []

    all_vecs = torch.cat(all_vecs_list, dim=0)

    final_emb_path = checkpoint_dir / state["final_embeddings"]
    final_ids_path = checkpoint_dir / state["final_ids"]
    torch.save(all_vecs, final_emb_path)
    np.save(final_ids_path, np.array(all_ids, dtype=object))

    state["completed"] = True
    state["processed"] = all_vecs.size(0)
    _save_checkpoint_state(state_path, state)

    _synchronize_final_outputs(all_vecs, all_ids, save_path_prefix)
    return all_vecs, all_ids


# ============================================================================
# PART 2: Metrics Computation
# ============================================================================

def cosine_similarity(queries: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between query and ref embeddings.
    Tensors may live on any device; computation stays on that device.
    Uses FP32 for robustness.
    """
    queries = F.normalize(queries.float(), dim=-1)
    refs = F.normalize(refs.float(), dim=-1)
    return queries @ refs.t()


def compute_muap_and_rp90(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Global micro-AP and R@P90 from flattened pair scores.

    all_scores: shape [N_pairs]
    all_labels: shape [N_pairs], in {0,1}

    Returns:
        Dictionary with keys 'micro_ap' and 'recall_at_p90'
    """
    assert all_scores.shape == all_labels.shape

    # Sort by score descending
    order = np.argsort(all_scores)[::-1]
    sorted_labels = all_labels[order]

    cum_pos = np.cumsum(sorted_labels)
    total_pos = float(sorted_labels.sum())
    if total_pos == 0:
        return {"micro_ap": 0.0, "recall_at_p90": 0.0}

    ranks = np.arange(1, len(sorted_labels) + 1)
    precision = cum_pos / ranks
    recall = cum_pos / total_pos

    # µAP: average precision over all positive instances
    muap = float((precision * sorted_labels).sum() / total_pos)

    # R@P90: max recall where precision >= 0.9
    mask = precision >= 0.9
    rp90 = float(recall[mask].max()) if mask.any() else 0.0

    return {"micro_ap": muap, "recall_at_p90": rp90}


def evaluate_retrieval(
    query_vecs: torch.Tensor,
    query_ids: List[str],
    ref_vecs: torch.Tensor,
    ref_ids: List[str],
    gt_map: Dict[str, List[str]],
    topk_list: List[int] = [10],
    device: Optional[torch.device] = None,
    query_chunk_size: int = 256,
    ref_chunk_size: int = 8192,
    max_global_pairs_per_query: int = 512,
) -> Dict[str, float]:
    """
    Memory-safe descriptor-only retrieval metrics over very large reference sets.

    Computes per-query Recall@k / mAP@k exactly, and global µAP / R@P90 using
    all positives plus the top ``max_global_pairs_per_query`` negatives per
    query. Restricting negatives keeps memory bounded while preserving recall at
    the decision thresholds of interest (low-scoring negatives do not affect the
    ranking of true positives).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_vecs = F.normalize(query_vecs.float(), dim=-1).contiguous()
    ref_vecs = F.normalize(ref_vecs.float(), dim=-1).contiguous()
    query_vecs_cpu = query_vecs.cpu()
    ref_vecs_cpu = ref_vecs.cpu()

    num_queries = query_vecs_cpu.size(0)
    num_refs = ref_vecs_cpu.size(0)

    if num_refs == 0 or num_queries == 0:
        empty_metrics: Dict[str, float] = {}
        for k in topk_list:
            empty_metrics[f"Recall@{k}"] = 0.0
            empty_metrics[f"mAP@{k}"] = 0.0
        empty_metrics["micro_ap"] = 0.0
        empty_metrics["recall_at_p90"] = 0.0
        return empty_metrics

    topk_max = min(max(topk_list), num_refs)
    global_k = 0 if max_global_pairs_per_query <= 0 else min(max_global_pairs_per_query, num_refs)

    retrieval_indices = torch.full((num_queries, topk_max), -1, dtype=torch.long)
    global_scores_store = (
        torch.full((num_queries, global_k), -float("inf"), dtype=torch.float32)
        if global_k > 0
        else None
    )
    global_indices_store = (
        torch.full((num_queries, global_k), -1, dtype=torch.long)
        if global_k > 0
        else None
    )

    query_chunk_size = max(1, int(query_chunk_size))
    ref_chunk_size = max(1, int(ref_chunk_size))

    with torch.no_grad():
        for q_start in range(0, num_queries, query_chunk_size):
            q_end = min(num_queries, q_start + query_chunk_size)
            q_chunk = query_vecs_cpu[q_start:q_end].to(device)
            chunk_size = q_chunk.size(0)

            best_small_scores = torch.full((chunk_size, topk_max), -float("inf"), device=device)
            best_small_indices = torch.full((chunk_size, topk_max), -1, dtype=torch.long, device=device)

            if global_k > 0:
                best_global_scores = torch.full((chunk_size, global_k), -float("inf"), device=device)
                best_global_indices = torch.full((chunk_size, global_k), -1, dtype=torch.long, device=device)
            else:
                best_global_scores = best_global_indices = None

            for r_start in range(0, num_refs, ref_chunk_size):
                r_end = min(num_refs, r_start + ref_chunk_size)
                r_chunk = ref_vecs_cpu[r_start:r_end].to(device)
                scores = torch.matmul(q_chunk, r_chunk.t())

                ref_indices = torch.arange(r_start, r_end, device=device)
                ref_indices = ref_indices.unsqueeze(0).expand(chunk_size, -1)

                # Update retrieval top-k
                candidate_scores = torch.cat([best_small_scores, scores], dim=1)
                candidate_indices = torch.cat([best_small_indices, ref_indices], dim=1)
                top_scores, top_pos = torch.topk(candidate_scores, k=topk_max, dim=1)
                best_small_scores = top_scores
                best_small_indices = torch.gather(candidate_indices, 1, top_pos)

                if global_k > 0 and best_global_scores is not None and best_global_indices is not None:
                    cand_global_scores = torch.cat([best_global_scores, scores], dim=1)
                    cand_global_indices = torch.cat([best_global_indices, ref_indices], dim=1)
                    g_scores, g_pos = torch.topk(cand_global_scores, k=global_k, dim=1)
                    best_global_scores = g_scores
                    best_global_indices = torch.gather(cand_global_indices, 1, g_pos)

            retrieval_indices[q_start:q_end] = best_small_indices.cpu()
            if global_k > 0 and best_global_scores is not None and best_global_indices is not None:
                global_scores_store[q_start:q_end] = best_global_scores.cpu()
                global_indices_store[q_start:q_end] = best_global_indices.cpu()

    query_ids_norm = [Path(qid).stem for qid in query_ids]
    ref_ids_norm = [Path(rid).stem for rid in ref_ids]
    ref_ids_norm_arr = np.array(ref_ids_norm)
    ref_id_to_index = {rid: idx for idx, rid in enumerate(ref_ids_norm)}

    metrics: Dict[str, float] = {}
    recall_hits = {k: 0 for k in topk_list}
    map_sums = {k: 0.0 for k in topk_list}
    num_evaluable = 0

    global_scores_parts: List[np.ndarray] = []
    global_labels_parts: List[np.ndarray] = []

    for q_idx, qid_norm in enumerate(query_ids_norm):
        gt_ids = gt_map.get(qid_norm, [])
        gt_set = {Path(rid).stem for rid in gt_ids if rid}

        retrieved_idx = retrieval_indices[q_idx].numpy()
        valid_mask = retrieved_idx >= 0
        retrieved_idx = retrieved_idx[valid_mask]
        retrieved_norm = ref_ids_norm_arr[retrieved_idx] if retrieved_idx.size else np.array([])

        if gt_set:
            num_evaluable += 1
            for k in topk_list:
                k_eff = min(k, retrieved_norm.size)
                topk_refs = retrieved_norm[:k_eff]
                if any(rid in gt_set for rid in topk_refs):
                    recall_hits[k] += 1

                hits = 0
                ap = 0.0
                for rank, ref_id_norm in enumerate(topk_refs, start=1):
                    if ref_id_norm in gt_set:
                        hits += 1
                        ap += hits / rank
                if hits > 0:
                    ap /= max(len(gt_set), 1)
                map_sums[k] += ap

        # Global metrics: positives + top-N negatives
        if global_k > 0 and global_scores_store is not None and global_indices_store is not None:
            g_scores = global_scores_store[q_idx].numpy()
            g_indices = global_indices_store[q_idx].numpy()
            mask = g_indices >= 0
            g_scores = g_scores[mask]
            g_indices = g_indices[mask]
        else:
            g_scores = np.empty(0, dtype=np.float32)
            g_indices = np.empty(0, dtype=np.int64)

        included_pos = set()
        if g_indices.size > 0:
            subset_ids = ref_ids_norm_arr[g_indices]
            g_labels = np.isin(subset_ids, list(gt_set)).astype(np.int8)
            included_pos.update(int(idx) for idx, lbl in zip(g_indices, g_labels) if lbl)
        else:
            g_labels = np.empty(0, dtype=np.int8)

        # Ensure every positive pair is present
        extra_pos_indices: List[int] = []
        for rid in gt_set:
            idx = ref_id_to_index.get(rid)
            if idx is not None and idx not in included_pos:
                extra_pos_indices.append(idx)

        if extra_pos_indices:
            pos_vecs = ref_vecs_cpu[extra_pos_indices]
            q_vec = query_vecs_cpu[q_idx]
            extra_scores = torch.matmul(pos_vecs, q_vec).cpu().numpy().astype(np.float32)
            g_scores = np.concatenate([g_scores, extra_scores])
            g_labels = np.concatenate([g_labels, np.ones(len(extra_scores), dtype=np.int8)])
            g_indices = np.concatenate([g_indices, np.array(extra_pos_indices, dtype=np.int64)])

        if g_scores.size > 0:
            global_scores_parts.append(g_scores.astype(np.float32))
            global_labels_parts.append(g_labels.astype(np.int8))

    if num_evaluable == 0:
        for k in topk_list:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"mAP@{k}"] = 0.0
    else:
        for k in topk_list:
            metrics[f"Recall@{k}"] = recall_hits[k] / num_evaluable
            metrics[f"mAP@{k}"] = map_sums[k] / num_evaluable

    if global_scores_parts:
        global_scores_arr = np.concatenate(global_scores_parts)
        global_labels_arr = np.concatenate(global_labels_parts)
        metrics.update(compute_muap_and_rp90(global_scores_arr, global_labels_arr))
    else:
        metrics["micro_ap"] = 0.0
        metrics["recall_at_p90"] = 0.0

    return metrics


# ============================================================================
# PART 3: Patch Generation for Two-Stage Evaluation
# ============================================================================

def make_six_patches(img: torch.Tensor) -> List[torch.Tensor]:
    """
    Generate 6 overlapping patches from a square image.

    Strategy:
    - Patch 0: Full image (global context)
    - Patches 1-4: Four 75% crops from corners (overlapping quadrants)
    - Patch 5: Center 75% crop

    Args:
        img: [C, H, W] tensor, assumed H == W (e.g., 224x224)

    Returns:
        List of 6 patches, each [C, H, W]
    """
    C, H, W = img.shape
    if H != W:
        warnings.warn(f"Expected square image, got {H}x{W}. Using min(H,W) as size.")
        size = min(H, W)
    else:
        size = H

    patches = []

    # Patch 0: Full image (identity)
    patches.append(img)

    # Define crop size (75% of original)
    crop_size = int(size * 0.75)
    step = size - crop_size

    # Patches 1-4: Corner crops
    # 1: Top-left
    p1 = TF.crop(img, top=0, left=0, height=crop_size, width=crop_size)
    patches.append(TF.resize(p1, [size, size]))

    # 2: Top-right
    p2 = TF.crop(img, top=0, left=step, height=crop_size, width=crop_size)
    patches.append(TF.resize(p2, [size, size]))

    # 3: Bottom-left
    p3 = TF.crop(img, top=step, left=0, height=crop_size, width=crop_size)
    patches.append(TF.resize(p3, [size, size]))

    # 4: Bottom-right
    p4 = TF.crop(img, top=step, left=step, height=crop_size, width=crop_size)
    patches.append(TF.resize(p4, [size, size]))

    # Patch 5: Center crop
    offset = (size - crop_size) // 2
    p5 = TF.crop(img, top=offset, left=offset, height=crop_size, width=crop_size)
    patches.append(TF.resize(p5, [size, size]))

    return patches


def make_six_patches_batch(batch_imgs: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """
    Generate 6 patches for each image in a batch.

    Args:
        batch_imgs: [B, C, H, W] batch of images

    Returns:
        patches: [B*6, C, H, W] flattened patches
        patch_to_query: List of length B*6, mapping each patch to its original query index
    """
    B, C, H, W = batch_imgs.shape
    all_patches = []
    patch_to_query = []

    for i in range(B):
        patches_i = make_six_patches(batch_imgs[i])
        all_patches.extend(patches_i)
        patch_to_query.extend([i] * len(patches_i))

    patches_tensor = torch.stack(all_patches, dim=0)
    return patches_tensor, patch_to_query


def build_ref_index_map(ref_ds: Dataset) -> Dict[str, int]:
    """
    Build a mapping from normalized reference IDs to dataset indices.

    Args:
        ref_ds: Reference dataset (assumed to return (img, sample_id) tuples)

    Returns:
        Dictionary mapping ref_id_norm -> dataset index
    """
    mapping = {}
    for idx in range(len(ref_ds)):
        try:
            _, sample_id = ref_ds[idx]
            ref_id_norm = Path(str(sample_id)).stem
            mapping[ref_id_norm] = idx
        except Exception as e:
            warnings.warn(f"Failed to index reference {idx}: {e}")
            continue

    return mapping


def normalize_id(img_id: str) -> str:
    """Normalize image ID by extracting stem (removing path and extension)."""
    return Path(str(img_id)).stem


# ============================================================================
# PART 4: CED Two-Stage Evaluation
# ============================================================================

def ced_two_stage_eval(
    model,  # CEDModel
    query_loader: DataLoader,
    ref_vecs: torch.Tensor,
    ref_ids: List[str],
    ref_ds: Dataset,
    gt_map: Dict[str, List[str]],
    k_candidates: int = 10,
    device: torch.device = torch.device("cuda"),
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run the full CED two-stage evaluation pipeline.

    Pipeline:
    1. For each query image:
       a. Generate 6 patches
       b. Encode patch descriptors
       c. Retrieve top-k reference candidates per patch (using descriptor similarity)
       d. Union candidates across patches
       e. Run CopyEditClassifier on (patch_tokens, ref_tokens) for all patch×candidate pairs
       f. Aggregate via max-over-patches per (query, ref) pair
    2. Compute µAP and R@P90 over all (query, ref) pairs

    Args:
        model: CEDModel with backbone, aggregator, and classifier
        query_loader: DataLoader for query images
        ref_vecs: [N_refs, D] precomputed normalized reference descriptors
        ref_ids: List of reference IDs (aligned with ref_vecs)
        ref_ds: Reference dataset (for loading images by ID)
        gt_map: Ground truth mapping {query_id -> [ref_ids]}
        k_candidates: Number of top candidates to retrieve per patch (paper uses 10)
        device: Device for computation
        verbose: Whether to show progress bars

    Returns:
        Dictionary with keys: 'micro_ap', 'recall_at_p90'
    """
    model.eval()
    ref_vecs = F.normalize(ref_vecs.float(), dim=-1).to(device)

    # Build reference index map
    if verbose:
        print(f"[CED Eval] Building reference index map...")
    ref_index_map = build_ref_index_map(ref_ds)
    ref_ids_norm = [normalize_id(rid) for rid in ref_ids]

    # Containers for all (query, ref) pairs across the dataset
    all_pair_scores = []
    all_pair_labels = []

    with torch.no_grad():
        iterator = tqdm(query_loader, desc="[CED Eval] Processing queries") if verbose else query_loader

        for batch_imgs, batch_q_ids in iterator:
            batch_imgs = batch_imgs.to(device)
            B = batch_imgs.size(0)

            # Step 1: Generate 6 patches per query
            patches, patch_to_query = make_six_patches_batch(batch_imgs)  # [B*6, C, H, W]
            patches = patches.to(device)

            # Step 2: Encode patch descriptors
            patch_descs, patch_feats = model.encode_images(patches, return_local=False)
            patch_descs = F.normalize(patch_descs.float(), dim=-1)  # [B*6, D]

            # Step 3: Retrieve top-k candidates per patch
            # Compute similarity: [B*6, D] @ [D, N_refs] -> [B*6, N_refs]
            sim_scores = patch_descs @ ref_vecs.t()
            topk_scores, topk_indices = torch.topk(sim_scores, k=k_candidates, dim=1)  # [B*6, k]

            # Step 4: For each query, collect unique candidate refs across all its patches
            query_candidates = {i: set() for i in range(B)}
            for patch_idx, query_idx in enumerate(patch_to_query):
                cand_refs = topk_indices[patch_idx].tolist()
                query_candidates[query_idx].update(cand_refs)

            # Step 5: For each query, run classifier on patch×candidate pairs
            for query_idx in range(B):
                q_id_str = str(batch_q_ids[query_idx])
                q_id_norm = normalize_id(q_id_str)

                # Get ground truth refs for this query
                gt_refs_raw = gt_map.get(q_id_norm, [])
                gt_refs_norm = {normalize_id(rid) for rid in gt_refs_raw if rid}

                # Get candidate ref indices
                cand_indices = sorted(query_candidates[query_idx])
                if not cand_indices:
                    continue

                # Extract this query's 6 patches
                query_patch_indices = [i for i, q in enumerate(patch_to_query) if q == query_idx]
                query_patches = patches[query_patch_indices]  # [6, C, H, W]

                # Encode query patch tokens
                q_patch_feats = model.backbone.get_features_from_images(
                    query_patches,
                    output_attentions=False
                )
                q_patch_tokens = q_patch_feats["patch_tokens_flat"]  # [6, N_q, D]
                num_patches = q_patch_tokens.size(0)

                # Load candidate reference images
                ref_imgs = []
                ref_ids_for_cands = []
                valid_cand_indices = []

                for cand_idx in cand_indices:
                    ref_id_norm = ref_ids_norm[cand_idx]
                    if ref_id_norm not in ref_index_map:
                        warnings.warn(f"Reference {ref_id_norm} not found in ref_index_map")
                        continue

                    ds_idx = ref_index_map[ref_id_norm]
                    try:
                        ref_img, _ = ref_ds[ds_idx]
                        ref_imgs.append(ref_img)
                        ref_ids_for_cands.append(ref_id_norm)
                        valid_cand_indices.append(cand_idx)
                    except Exception as e:
                        warnings.warn(f"Failed to load reference {ref_id_norm}: {e}")
                        continue

                if not ref_imgs:
                    continue

                # Stack and encode reference tokens
                ref_imgs_batch = torch.stack(ref_imgs, dim=0).to(device)  # [R, C, H, W]
                r_feats = model.backbone.get_features_from_images(
                    ref_imgs_batch,
                    output_attentions=False
                )
                r_tokens = r_feats["patch_tokens_flat"]  # [R, N_r, D]
                num_refs = r_tokens.size(0)

                # Step 6: Run classifier on all patch×ref pairs
                # For each patch, compute scores against all refs
                logits_per_patch = []  # Will be [num_patches, num_refs]

                for patch_idx in range(num_patches):
                    # Expand query tokens: [1, N_q, D] -> [R, N_q, D]
                    q_tok_expanded = q_patch_tokens[patch_idx].unsqueeze(0).repeat(num_refs, 1, 1)

                    # Run classifier: [R, N_q, D] x [R, N_r, D] -> [R, 1]
                    logits = model.classifier(q_tok_expanded, r_tokens)  # [R, 1]
                    logits_per_patch.append(logits.squeeze(-1))  # [R]

                # Stack: [num_patches, num_refs]
                logits_matrix = torch.stack(logits_per_patch, dim=0)

                # Step 7: Aggregate via max-over-patches
                scores_per_ref, _ = logits_matrix.max(dim=0)  # [num_refs]

                # Step 8: Convert to CPU and record scores + labels
                scores_per_ref_cpu = scores_per_ref.cpu().numpy()

                for ref_local_idx, ref_id_norm in enumerate(ref_ids_for_cands):
                    score = float(scores_per_ref_cpu[ref_local_idx])
                    label = 1 if ref_id_norm in gt_refs_norm else 0

                    all_pair_scores.append(score)
                    all_pair_labels.append(label)

    # Step 9: Compute µAP and R@P90 over all pairs
    if not all_pair_scores:
        warnings.warn("No valid pairs found during evaluation!")
        return {"micro_ap": 0.0, "recall_at_p90": 0.0}

    all_pair_scores = np.array(all_pair_scores, dtype=np.float32)
    all_pair_labels = np.array(all_pair_labels, dtype=np.int32)

    if verbose:
        print(f"[CED Eval] Total pairs evaluated: {len(all_pair_scores)}")
        print(f"[CED Eval] Positive pairs: {all_pair_labels.sum()} ({100*all_pair_labels.mean():.2f}%)")

    metrics = compute_muap_and_rp90(all_pair_scores, all_pair_labels)

    return metrics


# ============================================================================
# PART 5: Runtime Benchmarking
# ============================================================================

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


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Descriptor computation
    'compute_descriptors_for_loader',
    # Metrics
    'cosine_similarity',
    'compute_muap_and_rp90',
    'evaluate_retrieval',
    # Two-stage evaluation
    'make_six_patches',
    'make_six_patches_batch',
    'build_ref_index_map',
    'normalize_id',
    'ced_two_stage_eval',
    # Benchmarking
    'benchmark_inference',
]
