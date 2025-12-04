"""
CEDetector Loss Functions
=========================

All loss functions for CEDetector training with exact paper implementations:
- NT-Xent (SimCLR) loss: Equation 3-4
- Kozachenko-Leonenko entropy loss: Equation 5-6
- Multi-similarity loss: Equation 7
- ASL (Asymmetrical Similarity Learning) loss
- Binary cross-entropy loss: Equation 9

Paper: "An End-to-End Vision Transformer Approach for Image Copy Detection"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent (SimCLR) loss with safe float32 math.

    Paper Equations 3-4: Normalized temperature-scaled cross entropy loss.

    Args:
        z1, z2: Normalized embeddings [B, D]
        temperature: τ parameter (paper uses 0.025)

    Returns:
        Contrastive loss scalar
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
    Kozachenko–Leonenko differential entropy estimator term.

    Paper Equation 5: Encourages uniform coverage on the hypersphere by
    maximizing nearest-neighbour distances.

    Args:
        z: [B, D] global descriptors (CLS projections), already L2-normalized
        eps: Small constant for numerical stability

    Returns:
        Mean log nearest-neighbour distance
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
    Symmetric KL divergence between similarity distributions.

    Args:
        z1, z2: Normalized embeddings [B, D]
        temperature: Softmax temperature

    Returns:
        Symmetric KL divergence
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

    Paper Equation 7: Learn embedding space where salient regions in positive
    samples are close and negative samples are far apart.

    Args:
        embeddings: [B, D] embeddings
        labels: [B] class labels for each embedding
        alpha: Weight for positive term (paper: 2)
        beta: Weight for negative term (paper: 50)
        margin: Similarity margin γ (paper: 1)

    Returns:
        Multi-similarity loss
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
    ASL (Asymmetrical Similarity Learning) loss.

    Combines norm ratio loss with metric learning (NT-Xent).

    Args:
        v_former, v_latter: Descriptors from former/latter augmentations
        lambda_mtr: Weight for metric term
        eps: Small constant for numerical stability

    Returns:
        ASL loss
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


# Binary cross-entropy loss for classifier
bce_loss = nn.BCEWithLogitsLoss()


__all__ = [
    'nt_xent_loss',
    'kozachenko_leonenko_loss',
    'similarity_kl_loss',
    'multi_similarity_loss',
    'asl_loss',
    'bce_loss',
]
