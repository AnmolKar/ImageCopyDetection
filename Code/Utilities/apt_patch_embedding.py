"""
Adaptive Patch Transformer (APT) - Patch Embedding Module
==========================================================

This module implements the patch aggregation strategy from APT paper:
"Accelerating Vision Transformers with Adaptive Patch Sizes"

Key Features:
- Multi-scale patch embedding with sub-patch aggregation
- Zero-initialized MLP for gradual information incorporation (ControlNet-style)
- Efficient handling of variable-sized patches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ZeroInitMLP(nn.Module):
    """
    Zero-initialized MLP from ControlNet paper.

    Paper Section 3.2: Allows model to gradually incorporate high-resolution
    details without initially degrading performance.

    Reference: "Adding Conditional Control to Text-to-Image Diffusion Models"
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        # Initialize weights and biases to zero
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AdaptivePatchEmbedding(nn.Module):
    """
    Adaptive patch embedding that handles multiple patch sizes.

    Paper Section 3.2: Patch Aggregation
    - Base patches use standard linear projection
    - Larger patches: split into sub-patches → embed → aggregate → combine with resized version
    - Zero-initialized MLP enables stable convergence from pretrained checkpoints
    """

    def __init__(
        self,
        base_patch_size: int = 16,
        num_scales: int = 3,
        embed_dim: int = 768,
        in_channels: int = 3,
        use_zero_init: bool = True,
    ):
        """
        Args:
            base_patch_size: Smallest patch size (e.g., 16)
            num_scales: Number of scales (paper uses 3: 16x16, 32x32, 64x64)
            embed_dim: Embedding dimension (e.g., 768 for ViT-B)
            in_channels: Input image channels (3 for RGB)
            use_zero_init: Whether to use zero-initialized MLP (recommended)
        """
        super().__init__()
        self.base_patch_size = base_patch_size
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.use_zero_init = use_zero_init

        # Standard patch embedding for base patches (reuse pretrained if available)
        # This should be replaced with the actual pretrained embedding
        self.base_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=base_patch_size,
            stride=base_patch_size
        )

        # Convolution aggregators for each larger scale
        # These aggregate sub-patch embeddings back to single token
        self.conv_aggregators = nn.ModuleList()
        for scale_idx in range(1, num_scales):
            scale = 2 ** scale_idx  # 2, 4, 8, ...
            # Convolutional downsampling: (scale x scale) embeddings → 1 embedding
            conv = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=scale,
                stride=scale,
                groups=1  # Full convolution for mixing
            )
            self.conv_aggregators.append(conv)

        # Zero-initialized MLPs for combining sub-patch and resized information
        if use_zero_init:
            self.zero_mlps = nn.ModuleList([
                ZeroInitMLP(embed_dim) for _ in range(num_scales - 1)
            ])
        else:
            self.zero_mlps = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim) for _ in range(num_scales - 1)
            ])

    def load_pretrained_projection(self, pretrained_proj: nn.Module):
        """
        Load pretrained patch projection from existing ViT.

        Args:
            pretrained_proj: Pretrained patch embedding layer
        """
        # Copy pretrained weights
        if hasattr(pretrained_proj, 'weight'):
            self.base_proj.weight.data.copy_(pretrained_proj.weight.data)
        if hasattr(pretrained_proj, 'bias') and pretrained_proj.bias is not None:
            self.base_proj.bias.data.copy_(pretrained_proj.bias.data)

    def embed_base_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Embed a base-size patch using standard projection.

        Args:
            patch: Image patch [C, H, W] where H=W=base_patch_size

        Returns:
            embedding: Patch embedding [embed_dim]
        """
        # Add batch dimension
        patch = patch.unsqueeze(0)  # [1, C, H, W]
        embedding = self.base_proj(patch)  # [1, embed_dim, 1, 1]
        embedding = embedding.squeeze(-1).squeeze(-1).squeeze(0)  # [embed_dim]
        return embedding

    def embed_large_patch(
        self,
        patch: torch.Tensor,
        scale_idx: int
    ) -> torch.Tensor:
        """
        Embed a large patch by combining sub-patch and resized information.

        Paper Equation 2:
        E(P_i) = ZeroMLP(Conv2d^(i)({E(P_j) | P_j ⊂ P_i})) + E(Resize_p(P_i))

        Args:
            patch: Large image patch [C, H, W] where H=W=base_patch_size * 2^(scale_idx+1)
            scale_idx: Scale index (0=base, 1=2x, 2=4x, etc.)

        Returns:
            embedding: Aggregated patch embedding [embed_dim]
        """
        scale = 2 ** (scale_idx + 1)  # 2, 4, 8, ...
        C, H, W = patch.shape

        # Path 1: Split into sub-patches and embed each
        sub_patch_size = self.base_patch_size
        sub_patches = []

        for i in range(0, H, sub_patch_size):
            for j in range(0, W, sub_patch_size):
                if i + sub_patch_size <= H and j + sub_patch_size <= W:
                    sub_patch = patch[:, i:i+sub_patch_size, j:j+sub_patch_size]
                    sub_embedding = self.embed_base_patch(sub_patch)
                    sub_patches.append(sub_embedding)

        # Arrange sub-patch embeddings in grid and apply conv aggregator
        # Grid size is number of sub-patches per dimension
        grid_size = H // sub_patch_size  # e.g., 64/16 = 4 for 2x scale, 64/16 = 4 for 4x scale

        # Verify we have the right number of patches
        expected_patches = grid_size * grid_size
        if len(sub_patches) != expected_patches:
            raise RuntimeError(
                f"Expected {expected_patches} sub-patches ({grid_size}x{grid_size}) "
                f"but got {len(sub_patches)} for patch size {H}x{W}"
            )

        sub_embeddings = torch.stack(sub_patches)  # [N_sub, embed_dim]
        sub_embeddings = sub_embeddings.view(grid_size, grid_size, self.embed_dim)
        sub_embeddings = sub_embeddings.permute(2, 0, 1).unsqueeze(0)  # [1, embed_dim, grid, grid]

        # Apply convolution to aggregate
        aggregated = self.conv_aggregators[scale_idx - 1](sub_embeddings)  # [1, embed_dim, 1, 1]
        aggregated = aggregated.squeeze(-1).squeeze(-1).squeeze(0)  # [embed_dim]

        # Apply zero-initialized MLP
        aggregated = self.zero_mlps[scale_idx - 1](aggregated)

        # Path 2: Resize patch to base size and embed
        patch_resized = F.interpolate(
            patch.unsqueeze(0),  # [1, C, H, W]
            size=(self.base_patch_size, self.base_patch_size),
            mode='bilinear',
            align_corners=False
        )
        resized_embedding = self.embed_base_patch(patch_resized.squeeze(0))

        # Combine paths (Paper Equation 2)
        final_embedding = aggregated + resized_embedding

        return final_embedding

    def forward(
        self,
        image: torch.Tensor,
        patch_list: List[Tuple[int, int, int, int, int]]
    ) -> torch.Tensor:
        """
        Embed all patches from an image using adaptive patch sizes.

        Args:
            image: Input image [C, H, W]
            patch_list: List of (scale_idx, y, x, h, w) tuples

        Returns:
            embeddings: Patch embeddings [N_patches, embed_dim]
        """
        embeddings = []

        for scale_idx, y, x, h, w in patch_list:
            # Extract patch
            patch = image[:, y:y+h, x:x+w]

            # Embed based on scale
            if scale_idx == 0:
                # Base patch
                embedding = self.embed_base_patch(patch)
            else:
                # Large patch
                embedding = self.embed_large_patch(patch, scale_idx)

            embeddings.append(embedding)

        # Stack all embeddings
        embeddings = torch.stack(embeddings)  # [N_patches, embed_dim]

        return embeddings

    def forward_batch(
        self,
        images: torch.Tensor,
        patch_lists: List[List[Tuple[int, int, int, int, int]]]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Embed patches from a batch of images with variable token counts.

        Args:
            images: Batch of images [B, C, H, W]
            patch_lists: List of patch lists for each image

        Returns:
            embeddings: Concatenated embeddings [total_patches, embed_dim]
            token_counts: Number of tokens per image
        """
        all_embeddings = []
        token_counts = []

        for img, patch_list in zip(images, patch_lists):
            embeddings = self.forward(img, patch_list)
            all_embeddings.append(embeddings)
            token_counts.append(len(patch_list))

        # Concatenate all embeddings for sequence packing
        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings, token_counts


class SequencePacker:
    """
    Handles sequence packing for variable-length token sequences.

    Paper Section 3.3: Dynamic Input Sizes
    Uses block-diagonal attention masks to process variable-length sequences
    efficiently in a single batch.
    """

    @staticmethod
    def create_attention_mask(token_counts: List[int], device: torch.device) -> torch.Tensor:
        """
        Create block-diagonal attention mask for sequence packing.

        Args:
            token_counts: Number of tokens per image in batch
            device: Target device

        Returns:
            mask: Attention mask [total_tokens, total_tokens]
        """
        total_tokens = sum(token_counts)
        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)

        start_idx = 0
        for count in token_counts:
            end_idx = start_idx + count
            # Each image can only attend to its own tokens
            mask[start_idx:end_idx, start_idx:end_idx] = True
            start_idx = end_idx

        return mask

    @staticmethod
    def unpack_sequence(
        packed_tokens: torch.Tensor,
        token_counts: List[int]
    ) -> List[torch.Tensor]:
        """
        Unpack a sequence-packed tensor back into individual sequences.

        Args:
            packed_tokens: Packed tokens [total_tokens, ...]
            token_counts: Number of tokens per sequence

        Returns:
            unpacked: List of token tensors
        """
        unpacked = []
        start_idx = 0

        for count in token_counts:
            end_idx = start_idx + count
            unpacked.append(packed_tokens[start_idx:end_idx])
            start_idx = end_idx

        return unpacked


__all__ = [
    'ZeroInitMLP',
    'AdaptivePatchEmbedding',
    'SequencePacker'
]
