"""
Adaptive Patch Transformer (APT) - Patch Selection Module
==========================================================

This module implements the entropy-based patch size selection mechanism from:
"Accelerating Vision Transformers with Adaptive Patch Sizes" (APT paper)

Key Features:
- Multi-scale entropy computation
- Hierarchical patch size assignment
- Quadtree-based patch structure
- Support for multiple scoring methods (entropy, Laplacian, upsampling)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class PatchSelector:
    """
    Determines adaptive patch sizes based on image content complexity.

    Paper Section 3.1: Deciding Patch Sizes
    Uses entropy to measure patch compressibility - lower entropy indicates
    higher redundancy and allows for larger patch sizes.
    """

    def __init__(
        self,
        base_patch_size: int = 16,
        num_scales: int = 3,
        thresholds: Optional[List[float]] = None,
        scorer_method: str = "entropy",
        num_bins: int = 256,
    ):
        """
        Args:
            base_patch_size: Smallest patch size (e.g., 16 for 16x16)
            num_scales: Number of patch scales (paper uses S=3: 16, 32, 64)
            thresholds: Entropy thresholds for each scale (τ_i in paper)
                       Lower values = more aggressive reduction
                       Paper uses ~5.5 for first scale, 4.0 for second
            scorer_method: Method to score patches ('entropy', 'laplacian', 'upsampling')
            num_bins: Number of bins for entropy histogram (paper uses 256)
        """
        self.base_patch_size = base_patch_size
        self.num_scales = num_scales
        self.scorer_method = scorer_method
        self.num_bins = num_bins

        # Default thresholds from paper (conservative to avoid information loss)
        if thresholds is None:
            if num_scales == 3:
                self.thresholds = [5.5, 4.0]  # Paper values
            elif num_scales == 2:
                self.thresholds = [5.5]
            else:
                self.thresholds = [5.5] * (num_scales - 1)
        else:
            assert len(thresholds) == num_scales - 1, \
                f"Need {num_scales-1} thresholds for {num_scales} scales"
            self.thresholds = thresholds

    def compute_entropy(self, patch: torch.Tensor) -> float:
        """
        Compute Shannon entropy of a patch using histogram binning.

        Paper Equation 1: H(P) = -Σ p_i log2(p_i)

        Args:
            patch: Image patch tensor [C, H, W]

        Returns:
            entropy: Scalar entropy value
        """
        # Convert to grayscale if color
        if patch.shape[0] == 3:
            # Standard RGB to grayscale conversion
            patch = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
        elif patch.shape[0] == 1:
            patch = patch[0]

        # Normalize to [0, 1] and quantize to bins
        patch_np = patch.detach().cpu().numpy()
        patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min() + 1e-8)
        patch_np = (patch_np * (self.num_bins - 1)).astype(np.int32)

        # Compute histogram
        hist, _ = np.histogram(patch_np.flatten(), bins=self.num_bins, range=(0, self.num_bins))

        # Compute probabilities
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-10)

        # Compute entropy (filter out zero probabilities)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return float(entropy)

    def compute_laplacian_score(self, patch: torch.Tensor) -> float:
        """
        Compute Laplacian edge score for a patch.
        Alternative scorer that detects sharp transitions.

        Args:
            patch: Image patch tensor [C, H, W]

        Returns:
            score: Laplacian variance (higher = more edges)
        """
        # Convert to grayscale
        if patch.shape[0] == 3:
            patch = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
        else:
            patch = patch[0]

        # Apply Laplacian kernel
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=patch.dtype, device=patch.device).unsqueeze(0).unsqueeze(0)

        patch_4d = patch.unsqueeze(0).unsqueeze(0)
        laplacian = F.conv2d(patch_4d, laplacian_kernel, padding=1)

        # Return variance of Laplacian as complexity score
        return float(laplacian.var().item())

    def compute_upsampling_score(self, patch: torch.Tensor, scale: int = 2) -> float:
        """
        Compute upsampling-based score (measures information loss from resize).

        Args:
            patch: Image patch tensor [C, H, W]
            scale: Downsampling factor

        Returns:
            score: MSE between original and upsampled (lower = more compressible)
        """
        # Downsample then upsample
        C, H, W = patch.shape
        patch_4d = patch.unsqueeze(0)

        downsampled = F.interpolate(
            patch_4d,
            size=(H // scale, W // scale),
            mode='bilinear',
            align_corners=False
        )
        upsampled = F.interpolate(
            downsampled,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # Compute MSE
        mse = F.mse_loss(patch_4d, upsampled)
        return float(mse.item())

    def score_patch(self, patch: torch.Tensor, scale_idx: int = 0) -> float:
        """
        Score a patch using the selected scoring method.

        Args:
            patch: Image patch tensor [C, H, W]
            scale_idx: Current scale index (for upsampling method)

        Returns:
            score: Complexity score
        """
        if self.scorer_method == "entropy":
            return self.compute_entropy(patch)
        elif self.scorer_method == "laplacian":
            return self.compute_laplacian_score(patch)
        elif self.scorer_method == "upsampling":
            scale_factor = 2 ** (scale_idx + 1)
            return self.compute_upsampling_score(patch, scale=scale_factor)
        else:
            raise ValueError(f"Unknown scorer method: {self.scorer_method}")

    def select_patch_sizes(
        self,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int, int]]]:
        """
        Hierarchically assign patch sizes to image regions.

        Paper Section 3.1: Uses hierarchical entropy computation to assign
        larger patches to low-entropy regions and smaller patches to complex ones.

        Args:
            image: Input image tensor [C, H, W]

        Returns:
            patch_size_map: Map of patch sizes assigned to each base patch [H_p, W_p]
            patch_list: List of (scale, y, x, h, w) tuples for each selected patch
        """
        C, H, W = image.shape

        # Compute number of base patches
        h_patches = H // self.base_patch_size
        w_patches = W // self.base_patch_size

        # Initialize patch size map (0 = not assigned yet, 1 = base, 2 = 2x, 3 = 4x, etc.)
        patch_size_map = torch.zeros(h_patches, w_patches, dtype=torch.int32, device=image.device)
        patch_list = []

        # Process from coarsest to finest scale
        for scale_idx in range(self.num_scales - 1, 0, -1):
            patch_scale = 2 ** scale_idx  # 4x, 2x for S=3
            scale_patch_size = self.base_patch_size * patch_scale
            threshold = self.thresholds[scale_idx - 1]

            # Iterate over non-overlapping patches at this scale
            for i in range(0, h_patches, patch_scale):
                for j in range(0, w_patches, patch_scale):
                    # Check if this region is already assigned
                    if i + patch_scale > h_patches or j + patch_scale > w_patches:
                        continue

                    region = patch_size_map[i:i+patch_scale, j:j+patch_scale]
                    if region.max() > 0:  # Already assigned at coarser scale
                        continue

                    # Extract patch from image
                    y_start = i * self.base_patch_size
                    x_start = j * self.base_patch_size
                    y_end = min(y_start + scale_patch_size, H)
                    x_end = min(x_start + scale_patch_size, W)

                    patch = image[:, y_start:y_end, x_start:x_end]

                    # Score patch
                    score = self.score_patch(patch, scale_idx)

                    # Assign if below threshold (low complexity)
                    if score < threshold:
                        patch_size_map[i:i+patch_scale, j:j+patch_scale] = scale_idx + 1
                        patch_list.append((scale_idx, y_start, x_start, y_end - y_start, x_end - x_start))

        # Assign remaining patches to base size
        unassigned_mask = (patch_size_map == 0)
        patch_size_map[unassigned_mask] = 1

        # Add base patches to list
        for i in range(h_patches):
            for j in range(w_patches):
                if patch_size_map[i, j] == 1:
                    y_start = i * self.base_patch_size
                    x_start = j * self.base_patch_size
                    patch_list.append((0, y_start, x_start, self.base_patch_size, self.base_patch_size))

        return patch_size_map, patch_list

    def compute_reduction_stats(self, patch_size_map: torch.Tensor) -> Dict[str, float]:
        """
        Compute token reduction statistics.

        Args:
            patch_size_map: Patch size assignment map

        Returns:
            stats: Dictionary with reduction statistics
        """
        h_patches, w_patches = patch_size_map.shape
        total_base_patches = h_patches * w_patches

        # Count base patches assigned to each scale
        # Note: patch_size_map contains the scale index + 1
        # Scale 0 (base 16x16): covers 1 base patch → 1 token
        # Scale 1 (2x, 32x32): covers 4 base patches → 1 token
        # Scale 2 (4x, 64x64): covers 16 base patches → 1 token

        base_patch_counts = {}
        for scale_idx in range(self.num_scales):
            scale_val = scale_idx + 1
            count = (patch_size_map == scale_val).sum().item()
            base_patch_counts[f"scale_{scale_idx}"] = count

        # Calculate actual tokens
        # Each scale covers (2^scale_idx)^2 base patches but produces 1 token
        actual_tokens = 0
        for scale_idx in range(self.num_scales):
            base_patches_at_scale = base_patch_counts[f"scale_{scale_idx}"]
            patches_per_token = (2 ** scale_idx) ** 2  # 1, 4, 16, ...

            # Number of actual APT tokens at this scale
            tokens_at_scale = base_patches_at_scale // patches_per_token
            actual_tokens += tokens_at_scale

        tokens_saved = total_base_patches - actual_tokens
        reduction_ratio = tokens_saved / total_base_patches if total_base_patches > 0 else 0.0

        return {
            "total_base_patches": total_base_patches,
            "actual_tokens": actual_tokens,
            "tokens_saved": tokens_saved,
            "reduction_ratio": reduction_ratio,
            **base_patch_counts
        }


__all__ = ['PatchSelector']
