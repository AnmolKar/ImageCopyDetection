"""
CEDetector Model Architecture Components
========================================

Contains all model architecture classes for the CEDetector system:
- DinoV3Backbone: Vision Transformer backbone with attention extraction
- CEDFeatureAggregator: Feature aggregation with GeM pooling and projection
- TransformerBlock: Self-attention block for classifier
- CopyEditClassifier: Cross-attention classifier for pair matching
- CEDModel: High-level model orchestrator

Based on: "An End-to-End Vision Transformer Approach for Image Copy Detection"
"""

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from transformers import AutoModel, AutoProcessor

try:  # transformers>=4.26 exposes AutoImageProcessor; fall back otherwise
    from transformers import AutoImageProcessor  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some releases
    AutoImageProcessor = None  # type: ignore


class DinoV3Backbone(nn.Module):
    """Wrapper that exposes token-level DINOv3 activations."""

    def __init__(self, model_name: str, use_processor: bool = True):
        super().__init__()
        dtype = torch.float32  # keep backbone full precision for stability

        # Processor normalizes raw RGB tensors coming from datasets
        # (set use_processor=False only if you feed pre-processed pixel_values)
        self.processor = self._load_processor(model_name) if use_processor else None
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
        )
        self._supports_attn_impl = hasattr(self.model, "set_attn_implementation")
        current_impl = getattr(self.model, "_attn_implementation", None)
        self._attn_impl_is_eager = current_impl == "eager"
        self.patch_size = getattr(self.model.config, "patch_size", 16)
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.hidden_size = self.model.config.hidden_size

    @staticmethod
    def _load_processor(model_name: str):
        """
        Try AutoImageProcessor if available; otherwise fall back to AutoProcessor.
        This avoids ImportError on older transformers versions.
        """
        if AutoImageProcessor is not None:
            try:
                return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                pass
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def preprocess(self, images: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        # Use no_grad just for the CPU-side numpy + processor work
        with torch.no_grad():
            images_np = [img.detach().cpu().permute(1, 2, 0).numpy() for img in images]
            encoded = self.processor(images=images_np, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    def _ensure_eager_attn(self):
        if self._supports_attn_impl and not self._attn_impl_is_eager:
            try:
                self.model.set_attn_implementation("eager")
                self._attn_impl_is_eager = True
            except Exception as exc:
                warnings.warn(
                    f"DinoV3Backbone: unable to switch attention implementation to 'eager': {exc}"
                )

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        if output_attentions:
            self._ensure_eager_attn()
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

    def get_features_from_images(
        self,
        images: torch.Tensor,
        output_attentions: bool = False,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        if self.processor is None:
            raise RuntimeError(
                "DinoV3Backbone was instantiated with use_processor=False, but get_features_from_images() "
                "was called with raw image tensors. Pass use_processor=True or pre-process inputs yourself."
            )
        encoded = self.preprocess(images, device)
        return self.forward(encoded["pixel_values"], output_attentions=output_attentions)


class CEDFeatureAggregator(nn.Module):
    """
    Fuse CLS tokens with pooled local descriptors to create CED embeddings.

    Paper Section 3.1: Feature Aggregation
    - Element-wise attention weighting
    - GeM pooling with p=3
    - Learnable projection (paper uses whitening)
    - L2 normalization
    - Output: v = [z; u] (global + local descriptor)
    """

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
        """GeM (Generalized Mean) Pooling"""
        # Force stable fp32 math and avoid huge values in pow
        x = x.float()
        x = x.clamp(min=eps)
        x = x.pow(p)
        x = x.mean(dim=1)
        return x.pow(1.0 / p)

    def compute_local_embedding(
        self,
        patch_tokens_flat: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute local descriptor u from patch tokens with attention weighting."""
        weighted_tokens = patch_tokens_flat.float()
        if attn_weights is not None:
            weights = attn_weights.view(attn_weights.size(0), -1).float()
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            weights = weights.unsqueeze(-1)
            weighted_tokens = weighted_tokens * (weights + 1e-6)

        local = self.gem(weighted_tokens, p=self.gem_p)
        local = self.proj_loc(local)
        return F.normalize(local, dim=-1)

    def forward_components(
        self,
        cls: torch.Tensor,
        patch_tokens_flat: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return separate components for loss computation.

        Returns:
            cls_global: z (projected CLS token)
            local: u (salient regional features)
            descriptor: v = [z; u] (concatenated descriptor)
        """
        # Global (CLS) embedding: z
        cls_global = self.proj_cls(cls.float())
        cls_global = F.normalize(cls_global, dim=-1)

        # Local pooled embedding: u
        local = self.compute_local_embedding(patch_tokens_flat, attn_weights=attn_weights)

        # Concatenate global + local (paper formula: v = [z; u])
        descriptor = torch.cat([cls_global, local], dim=-1)
        descriptor = F.normalize(descriptor, dim=-1)
        return cls_global, local, descriptor

    def forward(
        self,
        cls: torch.Tensor,
        patch_tokens_flat: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
        return_local: bool = False,
    ) -> torch.Tensor:
        """Standard forward pass returning descriptor v."""
        cls_global, local, descriptor = self.forward_components(
            cls=cls,
            patch_tokens_flat=patch_tokens_flat,
            attn_weights=attn_weights,
        )
        if return_local:
            # kept for backwards compatibility if you ever use it directly
            return descriptor, local
        return descriptor


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
    """
    Cross-attention head that predicts copy-edit likelihoods for query/reference pairs.

    Paper Section 3.2 & Figure 5:
    - Cross-attention layer between query and reference
    - 2 multi-head self-attention blocks
    - Global average pooling
    - Linear layer (+ sigmoid in BCE loss)
    """

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1, use_checkpoint: bool = True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        self.use_checkpoint = use_checkpoint

    def forward(self, q_tokens: torch.Tensor, r_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tokens: Query patch tokens [B, N_q, D]
            r_tokens: Reference patch tokens [B, N_r, D]

        Returns:
            logits: Copy-edit likelihood scores [B]
        """
        q_norm = q_tokens
        r_norm = r_tokens
        cross_out, _ = self.cross_attn(q_norm, r_norm, r_norm)
        x = q_tokens + cross_out

        # Use gradient checkpointing for transformer blocks to save memory
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.head(x)
        return logits


class CEDModel(nn.Module):
    """
    High-level wrapper orchestrating backbone encoding and pair scoring.

    This is the main model that ties together:
    - DinoV3Backbone: Feature extraction
    - CEDFeatureAggregator: Descriptor construction
    - CopyEditClassifier: Pair matching
    """

    def __init__(self, backbone: DinoV3Backbone, dim: int):
        super().__init__()
        self.backbone = backbone
        self.aggregator = CEDFeatureAggregator(dim=dim, gem_p=3.0, use_proj=True)
        self.classifier = CopyEditClassifier(dim=dim)

        # Move to appropriate precision
        device = next(backbone.parameters()).device
        self.aggregator = self.aggregator.to(device=device, dtype=torch.float32)
        self.classifier = self.classifier.to(device=device, dtype=torch.float32)

    def encode_images(
        self,
        images: torch.Tensor,
        return_local: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode images to descriptors.

        Returns:
            descriptor: Combined global+local descriptor v = [z; u]
            feats: Dictionary containing intermediate features
        """
        # Run backbone + aggregator in full precision to avoid fp16 overflow/NANs
        with amp.autocast(device_type="cuda", enabled=False):
            feats = self.backbone.get_features_from_images(images, output_attentions=True)

            # Use forward_components to get cls, local, descriptor
            cls_global, local, descriptor = self.aggregator.forward_components(
                cls=feats["cls"],
                patch_tokens_flat=feats["patch_tokens_flat"],
                attn_weights=feats.get("attn_cls_to_patches"),
            )

        # Stash for losses: z (cls_global) and u (local_descriptor)
        feats["cls_global"] = cls_global  # z in the paper
        if return_local:
            feats["local_descriptor"] = local  # u in the paper

        return descriptor, feats

    def score_pair(
        self,
        q_images: torch.Tensor,
        r_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Score query-reference image pairs using the classifier.

        Returns:
            logits: Similarity scores [B]
            q_feats: Query features
            r_feats: Reference features
        """
        # Classification path can still use patch tokens directly
        with amp.autocast(device_type="cuda", enabled=False):
            q_feats = self.backbone.get_features_from_images(q_images)
            r_feats = self.backbone.get_features_from_images(r_images)
            q_tokens = q_feats["patch_tokens_flat"]
            r_tokens = r_feats["patch_tokens_flat"]
            logits = self.classifier(q_tokens, r_tokens).squeeze(-1)
        return logits, q_feats, r_feats


__all__ = [
    'DinoV3Backbone',
    'CEDFeatureAggregator',
    'TransformerBlock',
    'CopyEditClassifier',
    'CEDModel',
]
