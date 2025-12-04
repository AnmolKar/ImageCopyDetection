"""
APT-CED Integration Wrapper
============================

Simplified integration of APT with CEDModel for easier adoption.

This wrapper provides a drop-in replacement interface that maintains
compatibility with existing CED training code.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import warnings

from ced_models import DinoV3Backbone, CEDModel
from apt_patch_selector import PatchSelector


class APTCEDModel(nn.Module):
    """
    CEDModel with APT acceleration.

    This is a simplified integration that adds APT preprocessing
    while maintaining the existing CED architecture.

    Note: This is a simplified version for initial integration.
    Full performance gains require deeper integration into the transformer.
    """

    def __init__(
        self,
        ced_model: CEDModel,
        enable_apt: bool = True,
        num_scales: int = 3,
        thresholds: Optional[list] = None,
        scorer_method: str = "entropy",
    ):
        """
        Args:
            ced_model: Existing CEDModel instance
            enable_apt: Whether to use APT (can be toggled)
            num_scales: Number of patch scales
            thresholds: Entropy thresholds (default: [5.5, 4.0])
            scorer_method: 'entropy', 'laplacian', or 'upsampling'
        """
        super().__init__()

        self.ced_model = ced_model
        self.enable_apt = enable_apt

        # APT components
        if thresholds is None:
            thresholds = [5.5, 4.0] if num_scales == 3 else [5.5]

        self.patch_selector = PatchSelector(
            base_patch_size=ced_model.backbone.patch_size,
            num_scales=num_scales,
            thresholds=thresholds,
            scorer_method=scorer_method,
        )

        # Statistics
        self._stats = {
            "total_images": 0,
            "total_tokens_base": 0,
            "total_tokens_apt": 0,
        }

    @property
    def backbone(self):
        """Pass-through to CED backbone."""
        return self.ced_model.backbone

    @property
    def aggregator(self):
        """Pass-through to CED aggregator."""
        return self.ced_model.aggregator

    @property
    def classifier(self):
        """Pass-through to CED classifier."""
        return self.ced_model.classifier

    def parameters(self):
        """Return all parameters."""
        return self.ced_model.parameters()

    def _analyze_image_complexity(self, images: torch.Tensor) -> Dict[str, float]:
        """
        Analyze image complexity and estimate potential speedup.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            stats: Complexity statistics
        """
        B, C, H, W = images.shape
        base_patches = (H // self.patch_selector.base_patch_size) * \
                      (W // self.patch_selector.base_patch_size)

        if not self.enable_apt:
            return {
                "base_tokens": base_patches,
                "apt_tokens": base_patches,
                "reduction_ratio": 0.0,
            }

        # Analyze each image
        total_apt_tokens = 0
        for i in range(B):
            img = images[i]
            patch_size_map, patch_list = self.patch_selector.select_patch_sizes(img)
            total_apt_tokens += len(patch_list)

        avg_apt_tokens = total_apt_tokens / B

        reduction_ratio = 1.0 - (avg_apt_tokens / base_patches)

        return {
            "base_tokens": base_patches,
            "apt_tokens": avg_apt_tokens,
            "reduction_ratio": reduction_ratio,
        }

    def encode_images(
        self,
        images: torch.Tensor,
        return_local: bool = False,
        analyze_only: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode images with optional APT analysis.

        Args:
            images: Input images [B, C, H, W]
            return_local: Whether to return local descriptors
            analyze_only: If True, only analyze complexity without APT

        Returns:
            descriptor: Combined descriptor
            feats: Feature dictionary with optional APT stats
        """
        # Analyze complexity if APT enabled
        if self.enable_apt or analyze_only:
            stats = self._analyze_image_complexity(images)
            self._stats["total_images"] += images.shape[0]
            self._stats["total_tokens_base"] += stats["base_tokens"] * images.shape[0]
            self._stats["total_tokens_apt"] += stats["apt_tokens"] * images.shape[0]

        # For now, use standard encoding (full APT requires transformer integration)
        # This provides complexity analysis while maintaining compatibility
        descriptor, feats = self.ced_model.encode_images(images, return_local=return_local)

        # Add APT stats to features
        if self.enable_apt or analyze_only:
            feats["apt_stats"] = stats
            feats["apt_reduction"] = stats["reduction_ratio"]

        return descriptor, feats

    def score_pair(
        self,
        q_images: torch.Tensor,
        r_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Score image pairs (pass-through to CED)."""
        return self.ced_model.score_pair(q_images, r_images)

    def get_reduction_stats(self) -> Dict[str, float]:
        """
        Get accumulated token reduction statistics.

        Returns:
            stats: Statistics dictionary
        """
        if self._stats["total_images"] == 0:
            return {"reduction_ratio": 0.0, "speedup_estimate": 1.0}

        avg_base = self._stats["total_tokens_base"] / self._stats["total_images"]
        avg_apt = self._stats["total_tokens_apt"] / self._stats["total_images"]

        reduction_ratio = 1.0 - (avg_apt / avg_base)

        # Estimate speedup (attention is O(N²))
        # Empirical: speedup ≈ 1 / (0.3 + 0.7 * token_ratio²)
        token_ratio = avg_apt / avg_base
        flops_ratio = token_ratio ** 2
        speedup_estimate = 1.0 / (0.3 + 0.7 * flops_ratio)

        return {
            "images_processed": self._stats["total_images"],
            "avg_base_tokens": avg_base,
            "avg_apt_tokens": avg_apt,
            "reduction_ratio": reduction_ratio,
            "flops_ratio": flops_ratio,
            "speedup_estimate": speedup_estimate,
            "throughput_gain": f"+{(speedup_estimate - 1) * 100:.1f}%"
        }

    def reset_stats(self):
        """Reset accumulated statistics."""
        self._stats = {
            "total_images": 0,
            "total_tokens_base": 0,
            "total_tokens_apt": 0,
        }

    def print_stats(self):
        """Print reduction statistics."""
        stats = self.get_reduction_stats()
        if stats["images_processed"] == 0:
            print("[APT] No images processed yet")
            return

        print("\n" + "="*60)
        print("APT Token Reduction Statistics")
        print("="*60)
        print(f"Images processed:     {stats['images_processed']:,}")
        print(f"Avg tokens (base):    {stats['avg_base_tokens']:.1f}")
        print(f"Avg tokens (APT):     {stats['avg_apt_tokens']:.1f}")
        print(f"Token reduction:      {stats['reduction_ratio']*100:.1f}%")
        print(f"FLOPS ratio:          {stats['flops_ratio']:.3f}")
        print(f"Estimated speedup:    {stats['speedup_estimate']:.2f}×")
        print(f"Throughput gain:      {stats['throughput_gain']}")
        print("="*60 + "\n")


def wrap_ced_with_apt(
    ced_model: CEDModel,
    enable_apt: bool = True,
    num_scales: int = 3,
    thresholds: Optional[list] = None,
    scorer_method: str = "entropy",
) -> APTCEDModel:
    """
    Wrap an existing CEDModel with APT analysis.

    This provides token reduction analysis and estimates speedup
    while maintaining compatibility with existing training code.

    Args:
        ced_model: Existing CEDModel
        enable_apt: Enable APT analysis
        num_scales: Number of scales (3 recommended)
        thresholds: Entropy thresholds (default: [5.5, 4.0])
        scorer_method: Scoring method

    Returns:
        apt_ced: APT-wrapped CED model

    Example:
        >>> backbone = DinoV3Backbone(model_name="facebook/dinov3-vitb16-pretrain-lvd1689m")
        >>> ced_model = CEDModel(backbone=backbone, dim=768)
        >>> apt_ced = wrap_ced_with_apt(ced_model, enable_apt=True)
        >>>
        >>> # Use as normal
        >>> descriptor, feats = apt_ced.encode_images(images)
        >>>
        >>> # Check stats
        >>> apt_ced.print_stats()
    """
    apt_ced = APTCEDModel(
        ced_model=ced_model,
        enable_apt=enable_apt,
        num_scales=num_scales,
        thresholds=thresholds,
        scorer_method=scorer_method,
    )

    print(f"[APT] Wrapped CEDModel with APT analysis")
    print(f"[APT]   Scales: {num_scales}")
    print(f"[APT]   Thresholds: {thresholds or [5.5, 4.0]}")
    print(f"[APT]   Scorer: {scorer_method}")
    print(f"[APT]   Enabled: {enable_apt}")
    print(f"[APT] Note: This provides analysis only. Full speedup requires")
    print(f"[APT]       deeper transformer integration (work in progress).")

    return apt_ced


__all__ = [
    'APTCEDModel',
    'wrap_ced_with_apt',
]
