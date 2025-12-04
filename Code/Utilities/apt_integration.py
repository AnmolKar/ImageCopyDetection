"""
Adaptive Patch Transformer (APT) - Integration Module
======================================================

Simple interface for integrating APT with existing models.

Usage:
    from apt_integration import enable_apt

    # Wrap your existing backbone
    apt_backbone = enable_apt(
        backbone,
        num_scales=3,
        thresholds=[5.5, 4.0],
        use_zero_init=True
    )

    # Use as normal - APT is applied automatically
    features = apt_backbone.get_features_from_images(images, use_apt=True)
"""

import torch
import torch.nn as nn
from typing import Optional, List
from pathlib import Path

from apt_backbone import APTDinoV3Backbone


def enable_apt(
    dinov3_backbone: nn.Module,
    base_patch_size: int = 16,
    num_scales: int = 3,
    thresholds: Optional[List[float]] = None,
    scorer_method: str = "entropy",
    use_zero_init: bool = True,
    fine_tune_epochs: int = 0,
) -> APTDinoV3Backbone:
    """
    Enable APT for an existing DINOv3 backbone.

    Paper Section 4.2: "Short Fine-Tuning"
    APT can be applied to a pretrained ViT and converges in as little as 1 epoch.

    Args:
        dinov3_backbone: Existing DinoV3Backbone instance
        base_patch_size: Base patch size (16 for ViT-B/16, 14 for ViT-L/14)
        num_scales: Number of patch scales (paper uses 3)
        thresholds: Entropy thresholds per scale (default: [5.5, 4.0] from paper)
        scorer_method: Patch scoring method ('entropy', 'laplacian', 'upsampling')
        use_zero_init: Use zero-initialized MLP (recommended for pretrained models)
        fine_tune_epochs: Number of epochs to fine-tune (0 = no fine-tuning)

    Returns:
        apt_backbone: APT-enabled backbone
    """
    # Default thresholds from paper (conservative)
    if thresholds is None:
        if num_scales == 3:
            thresholds = [5.5, 4.0]  # Paper values for ViT-L/14@336
        elif num_scales == 2:
            thresholds = [5.5]
        else:
            thresholds = [5.5] * (num_scales - 1)

    # Create APT-enabled backbone
    apt_backbone = APTDinoV3Backbone(
        dinov3_backbone=dinov3_backbone,
        base_patch_size=base_patch_size,
        num_scales=num_scales,
        thresholds=thresholds,
        use_zero_init=use_zero_init,
    )

    # Set scorer method
    apt_backbone.apt_wrapper.patch_selector.scorer_method = scorer_method

    print(f"[APT] Enabled with {num_scales} scales (base={base_patch_size})")
    print(f"[APT] Thresholds: {thresholds}")
    print(f"[APT] Scorer: {scorer_method}")
    print(f"[APT] Zero-init: {use_zero_init}")

    if fine_tune_epochs > 0:
        print(f"[APT] Note: Fine-tuning for {fine_tune_epochs} epoch(s) recommended")
        print(f"[APT]       Paper shows 1 epoch is sufficient to recover accuracy")

    return apt_backbone


def compute_speedup_estimate(
    image_size: int,
    base_patch_size: int,
    num_scales: int = 3,
    thresholds: Optional[List[float]] = None,
    expected_reduction: float = 0.40,
) -> dict:
    """
    Estimate APT speedup for given configuration.

    Paper results (Table 2):
    - ViT-L/14@224: ~12% speedup, ~25% token reduction
    - ViT-L/14@336: ~33% speedup, ~56% token reduction
    - ViT-H/14@336: ~50% speedup, ~56% token reduction

    Args:
        image_size: Image resolution (e.g., 224, 336, 384)
        base_patch_size: Base patch size
        num_scales: Number of scales
        thresholds: Entropy thresholds
        expected_reduction: Expected token reduction ratio (0-1)

    Returns:
        estimates: Dictionary with speedup estimates
    """
    # Compute base token count
    patches_per_dim = image_size // base_patch_size
    base_tokens = patches_per_dim ** 2

    # Estimate reduced token count
    reduced_tokens = int(base_tokens * (1 - expected_reduction))

    # FLOPS reduction (attention is O(N^2))
    flops_ratio = (reduced_tokens / base_tokens) ** 2

    # Estimated speedup (empirical: not exactly 1/flops_ratio due to overhead)
    # Paper shows ~0.7x speedup for ~0.44 FLOPS ratio at 336 resolution
    estimated_speedup = 1.0 / (0.3 + 0.7 * flops_ratio)

    return {
        "image_size": image_size,
        "base_tokens": base_tokens,
        "reduced_tokens": reduced_tokens,
        "token_reduction": expected_reduction,
        "flops_ratio": flops_ratio,
        "estimated_speedup": estimated_speedup,
        "estimated_throughput_gain": f"+{(estimated_speedup - 1) * 100:.0f}%"
    }


def print_apt_info():
    """Print APT paper information and expected performance."""
    print("=" * 70)
    print("Adaptive Patch Transformer (APT)")
    print("Paper: 'Accelerating Vision Transformers with Adaptive Patch Sizes'")
    print("=" * 70)
    print()
    print("Key Results from Paper:")
    print("  • ViT-B@384: +21% throughput, 14% token reduction")
    print("  • ViT-L@336: +33% throughput, 56% token reduction")
    print("  • ViT-H@336: +50% throughput, 56% token reduction")
    print()
    print("Training:")
    print("  • Full fine-tuning: 29-86% faster training (Table 1)")
    print("  • 1-epoch adaptation: Recovers pretrained accuracy (Table 2)")
    print("  • Zero-init MLP: Enables stable fine-tuning from checkpoint")
    print()
    print("Configuration:")
    print("  • Scales: 3 (16×16, 32×32, 64×64 patches)")
    print("  • Thresholds: τ₁=5.5, τ₂=4.0 (conservative)")
    print("  • Scorer: Entropy-based (other options: Laplacian, upsampling)")
    print()
    print("Usage:")
    print("  1. Enable APT: apt_backbone = enable_apt(backbone)")
    print("  2. Fine-tune: 1 epoch with lr=1e-6, layer_decay=0.99")
    print("  3. Use: features = apt_backbone.get_features(..., use_apt=True)")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    print_apt_info()
    print()

    # Example speedup estimates
    print("Speedup Estimates:")
    print("-" * 70)

    for img_size, expected_red in [(224, 0.25), (336, 0.56), (384, 0.30), (448, 0.60)]:
        est = compute_speedup_estimate(
            image_size=img_size,
            base_patch_size=14,
            expected_reduction=expected_red
        )
        print(f"Resolution {img_size}×{img_size}:")
        print(f"  Tokens: {est['base_tokens']} → {est['reduced_tokens']} "
              f"(-{est['token_reduction']*100:.0f}%)")
        print(f"  Estimated speedup: {est['estimated_speedup']:.2f}× "
              f"({est['estimated_throughput_gain']})")
        print()


__all__ = [
    'enable_apt',
    'compute_speedup_estimate',
    'print_apt_info',
]
