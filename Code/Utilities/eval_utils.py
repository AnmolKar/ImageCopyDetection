import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# -----------------------------
# Descriptor computation (with optional TTA)
# -----------------------------

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

            desc = desc.float().cpu()
            if not torch.isfinite(desc).all():
                raise RuntimeError(
                    f"Non-finite descriptors encountered in compute_descriptors_for_loader_tta "
                    f"for prefix {save_path_prefix}"
                )

            all_vecs.append(desc)
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
    device: Optional[torch.device] = None,
    query_chunk_size: int = 256,
    ref_chunk_size: int = 8192,
    max_global_pairs_per_query: int = 512,
) -> Dict[str, float]:
    """
    Memory-safe retrieval metrics over very large reference sets.

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
        empty_metrics["muAP"] = 0.0
        empty_metrics["R@P90"] = 0.0
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
        metrics["muAP"] = 0.0
        metrics["R@P90"] = 0.0

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
