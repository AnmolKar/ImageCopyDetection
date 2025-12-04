"""
CED Augmentation Pipeline
==========================

Comprehensive augmentation pipeline for CEDetector training using 35+ transforms
from AugLy and Albumentations libraries, matching the paper's augmentation strategy.

Paper: "An End-to-End Vision Transformer Approach for Image Copy Detection"

The paper uses extensive augmentations to simulate copy-edit operations including:
- Geometric transforms (rotation, scaling, cropping, perspective)
- Color adjustments (brightness, contrast, saturation, hue)
- Compression artifacts (JPEG, blur)
- Noise and degradation
- Text/watermark overlay
- Filters and effects
"""

import random
from typing import List, Callable, Optional, Tuple, Union
import warnings

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Try to import augmentation libraries
try:
    import augly.image as imaugs
    AUGLY_AVAILABLE = True
except ImportError:
    AUGLY_AVAILABLE = False
    warnings.warn("augly not installed. Install with: pip install augly")

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    warnings.warn("albumentations not installed. Install with: pip install albumentations")


class CEDAugmentationPipeline:
    """
    Comprehensive augmentation pipeline for CEDetector training.

    Implements the paper's augmentation strategy with 35+ transforms that simulate
    real-world copy-edit operations. Randomly applies N transforms per image.

    Args:
        min_ops: Minimum number of augmentations to apply per image
        max_ops: Maximum number of augmentations to apply per image
        img_size: Target image size (used for resize/crop operations)
        seed: Random seed for reproducibility (optional)
    """

    def __init__(
        self,
        min_ops: int = 2,
        max_ops: int = 6,
        img_size: int = 224,
        seed: Optional[int] = None,
    ):
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.img_size = img_size
        self.rng = random.Random(seed)

        # Build augmentation pool
        self.augmentation_pool = self._build_augmentation_pool()

    @staticmethod
    def _ensure_pil_image(img: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
        """Normalize different image types (tensor/ndarray) back to PIL for downstream libs."""
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            tensor = img.detach().cpu()
            # torchvision expects CHW; if HW3 tensor, permute
            if tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.clamp(0.0, 1.0)
            return TF.to_pil_image(tensor)
        if isinstance(img, np.ndarray):
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        raise TypeError(f"Unsupported image type: {type(img)}")

    def _build_augmentation_pool(self) -> List[Tuple[str, Callable]]:
        """
        Build pool of augmentation operations.

        Returns:
            List of (name, function) tuples
        """
        pool = []

        # ============================================================
        # Basic torchvision transforms (always available)
        # ============================================================

        # Geometric transforms
        pool.append(("rotation_15", lambda img: TF.rotate(img, angle=self.rng.uniform(-15, 15))))
        pool.append(("rotation_30", lambda img: TF.rotate(img, angle=self.rng.uniform(-30, 30))))
        pool.append(("hflip", lambda img: TF.hflip(img)))
        pool.append(("vflip", lambda img: TF.vflip(img)))
        pool.append(("random_crop_75", self._random_crop_75))
        pool.append(("random_crop_50", self._random_crop_50))
        pool.append(("center_crop_80", self._center_crop_80))

        # Color adjustments
        pool.append(("brightness", lambda img: TF.adjust_brightness(img, self.rng.uniform(0.7, 1.3))))
        pool.append(("contrast", lambda img: TF.adjust_contrast(img, self.rng.uniform(0.7, 1.3))))
        pool.append(("saturation", lambda img: TF.adjust_saturation(img, self.rng.uniform(0.7, 1.3))))
        pool.append(("hue", lambda img: TF.adjust_hue(img, self.rng.uniform(-0.1, 0.1))))
        pool.append(("grayscale", lambda img: TF.to_grayscale(img, num_output_channels=3)))
        pool.append(("color_jitter", self._color_jitter))

        # Filters and effects
        pool.append(("gaussian_blur_3", lambda img: TF.gaussian_blur(img, kernel_size=3)))
        pool.append(("gaussian_blur_5", lambda img: TF.gaussian_blur(img, kernel_size=5)))
        pool.append(("sharpness", lambda img: TF.adjust_sharpness(img, self.rng.uniform(0.5, 2.0))))
        pool.append(("autocontrast", lambda img: TF.autocontrast(img)))
        pool.append(("equalize", lambda img: TF.equalize(img)))
        pool.append(("invert", lambda img: TF.invert(img)))
        pool.append(("posterize", lambda img: TF.posterize(img, bits=self.rng.choice([4, 5, 6, 7]))))
        pool.append(("solarize", lambda img: TF.solarize(img, threshold=self.rng.uniform(0.3, 0.7))))

        # ============================================================
        # AugLy transforms (if available)
        # ============================================================
        if AUGLY_AVAILABLE:
            # JPEG compression
            pool.append(("jpeg_quality_low", lambda img: self._augly_apply(img, imaugs.EncodingQuality(quality=self.rng.randint(30, 50)))))
            pool.append(("jpeg_quality_med", lambda img: self._augly_apply(img, imaugs.EncodingQuality(quality=self.rng.randint(50, 70)))))

            # Blur effects
            pool.append(("blur_mild", lambda img: self._augly_apply(img, imaugs.Blur(radius=self.rng.uniform(0.5, 2.0)))))
            pool.append(("blur_strong", lambda img: self._augly_apply(img, imaugs.Blur(radius=self.rng.uniform(2.0, 4.0)))))

            # Pixelation
            pool.append(("pixelation", lambda img: self._augly_apply(img, imaugs.Pixelization(ratio=self.rng.uniform(0.3, 0.7)))))

            # Padding with resize
            pool.append(("pad_square", lambda img: self._augly_apply(img, imaugs.PadSquare())))

            # Overlays (text/emoji simulate watermarks)
            pool.append(("overlay_text", lambda img: self._augly_apply_safe(img, imaugs.OverlayText,
                                                                             text="",
                                                                             font_size=0.05,
                                                                             opacity=0.3)))

        # ============================================================
        # Albumentations transforms (if available)
        # ============================================================
        if ALBUMENTATIONS_AVAILABLE:
            # Advanced geometric transforms
            pool.append(("elastic", self._albu_elastic_transform))
            pool.append(("grid_distortion", self._albu_grid_distortion))
            pool.append(("optical_distortion", self._albu_optical_distortion))
            pool.append(("perspective", self._albu_perspective))

            # Noise and artifacts
            pool.append(("gauss_noise", self._albu_gauss_noise))
            pool.append(("iso_noise", self._albu_iso_noise))
            pool.append(("motion_blur", self._albu_motion_blur))
            pool.append(("median_blur", self._albu_median_blur))

            # Color transforms
            pool.append(("clahe", self._albu_clahe))
            pool.append(("channel_shuffle", self._albu_channel_shuffle))
            pool.append(("rgb_shift", self._albu_rgb_shift))
            pool.append(("hue_sat_value", self._albu_hue_sat_value))

        return pool

    # ================================================================
    # Helper methods for torchvision transforms
    # ================================================================

    def _random_crop_75(self, img: Image.Image) -> Image.Image:
        """Random crop to 75% with resize back."""
        w, h = img.size
        crop_size = int(min(w, h) * 0.75)
        i = self.rng.randint(0, h - crop_size) if h > crop_size else 0
        j = self.rng.randint(0, w - crop_size) if w > crop_size else 0
        img = TF.crop(img, i, j, crop_size, crop_size)
        return TF.resize(img, [self.img_size, self.img_size])

    def _random_crop_50(self, img: Image.Image) -> Image.Image:
        """Random crop to 50% with resize back."""
        w, h = img.size
        crop_size = int(min(w, h) * 0.5)
        i = self.rng.randint(0, h - crop_size) if h > crop_size else 0
        j = self.rng.randint(0, w - crop_size) if w > crop_size else 0
        img = TF.crop(img, i, j, crop_size, crop_size)
        return TF.resize(img, [self.img_size, self.img_size])

    def _center_crop_80(self, img: Image.Image) -> Image.Image:
        """Center crop to 80% with resize back."""
        w, h = img.size
        crop_size = int(min(w, h) * 0.8)
        img = TF.center_crop(img, crop_size)
        return TF.resize(img, [self.img_size, self.img_size])

    def _color_jitter(self, img: Image.Image) -> Image.Image:
        """Apply random color jitter."""
        img = TF.adjust_brightness(img, self.rng.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img, self.rng.uniform(0.8, 1.2))
        img = TF.adjust_saturation(img, self.rng.uniform(0.8, 1.2))
        img = TF.adjust_hue(img, self.rng.uniform(-0.05, 0.05))
        return img

    # ================================================================
    # Helper methods for AugLy transforms
    # ================================================================

    def _augly_apply(self, img: Image.Image, transform) -> Image.Image:
        """Apply AugLy transform safely."""
        try:
            img_pil = self._ensure_pil_image(img)
            img_aug = transform(img_pil)
            return self._ensure_pil_image(img_aug)
        except Exception as e:
            warnings.warn(f"AugLy transform failed: {e}")
            return img

    def _augly_apply_safe(self, img: Image.Image, transform_class, **kwargs) -> Image.Image:
        """Apply AugLy transform with kwargs safely."""
        try:
            transform = transform_class(**kwargs)
            img_pil = self._ensure_pil_image(img)
            img_aug = transform(img_pil)
            return self._ensure_pil_image(img_aug)
        except Exception as e:
            warnings.warn(f"AugLy transform failed: {e}")
            return img

    # ================================================================
    # Helper methods for Albumentations transforms
    # ================================================================

    def _albu_elastic_transform(self, img: Image.Image) -> Image.Image:
        """Apply elastic transform."""
        transform = A.ElasticTransform(alpha=50, sigma=5, p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_grid_distortion(self, img: Image.Image) -> Image.Image:
        """Apply grid distortion."""
        transform = A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_optical_distortion(self, img: Image.Image) -> Image.Image:
        """Apply optical distortion."""
        transform = A.OpticalDistortion(distort_limit=(-0.5, 0.5), mode="camera", p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_perspective(self, img: Image.Image) -> Image.Image:
        """Apply perspective transform."""
        transform = A.Perspective(scale=(0.05, 0.1), p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_gauss_noise(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian noise."""
        transform = A.GaussNoise(std_range=(0.02, 0.1), mean_range=(0.0, 0.0), per_channel=True, p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_iso_noise(self, img: Image.Image) -> Image.Image:
        """Apply ISO noise."""
        transform = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_motion_blur(self, img: Image.Image) -> Image.Image:
        """Apply motion blur."""
        transform = A.MotionBlur(blur_limit=7, p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_median_blur(self, img: Image.Image) -> Image.Image:
        """Apply median blur."""
        transform = A.MedianBlur(blur_limit=5, p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_clahe(self, img: Image.Image) -> Image.Image:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        transform = A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_channel_shuffle(self, img: Image.Image) -> Image.Image:
        """Apply channel shuffle."""
        transform = A.ChannelShuffle(p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_rgb_shift(self, img: Image.Image) -> Image.Image:
        """Apply RGB shift."""
        transform = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def _albu_hue_sat_value(self, img: Image.Image) -> Image.Image:
        """Apply hue/saturation/value shift."""
        transform = A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=1.0
        )
        img_np = np.array(img)
        augmented = transform(image=img_np)
        return Image.fromarray(augmented["image"])

    # ================================================================
    # Main augmentation method
    # ================================================================

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply random augmentations to the input image.

        Args:
            img: PIL Image

        Returns:
            Augmented PIL Image
        """
        # Ensure PIL input for downstream libs
        img = self._ensure_pil_image(img)

        # Randomly select number of augmentations to apply
        num_ops = self.rng.randint(self.min_ops, self.max_ops)

        # Randomly sample augmentations without replacement
        selected_augs = self.rng.sample(self.augmentation_pool, min(num_ops, len(self.augmentation_pool)))

        # Apply selected augmentations sequentially
        for aug_name, aug_fn in selected_augs:
            try:
                img = aug_fn(img)
                img = self._ensure_pil_image(img)
            except Exception as e:
                # Skip failed augmentation and continue
                warnings.warn(f"Augmentation '{aug_name}' failed: {e}")
                continue

        return img


def build_ced_transforms(
    img_size_train: int = 224,
    img_size_eval: int = 224,
    use_augmentations: bool = True,
    min_ops: int = 2,
    max_ops: int = 6,
    seed: Optional[int] = None,
) -> Tuple[Callable, Callable]:
    """
    Build CED augmentation pipelines for training and evaluation.

    Args:
        img_size_train: Training image size
        img_size_eval: Evaluation image size
        use_augmentations: Whether to use augmentations for training
        min_ops: Minimum number of augmentations per image (training only)
        max_ops: Maximum number of augmentations per image (training only)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_transform, eval_transform) callables
    """
    import torchvision.transforms as T

    # Build training transform
    if use_augmentations:
        train_pipeline = CEDAugmentationPipeline(
            min_ops=min_ops,
            max_ops=max_ops,
            img_size=img_size_train,
            seed=seed,
        )
        # Compose: augmentations -> resize -> to_tensor
        train_transform = T.Compose([
            train_pipeline,
            T.Resize((img_size_train, img_size_train)),
            T.ToTensor(),
        ])
    else:
        # No augmentations, just resize and to_tensor
        train_transform = T.Compose([
            T.Resize((img_size_train, img_size_train)),
            T.ToTensor(),
        ])

    # Build evaluation transform (no augmentations)
    eval_transform = T.Compose([
        T.Resize((img_size_eval, img_size_eval)),
        T.ToTensor(),
    ])

    return train_transform, eval_transform


def check_dependencies() -> dict:
    """
    Check which augmentation libraries are available.

    Returns:
        Dictionary with availability status
    """
    return {
        "augly": AUGLY_AVAILABLE,
        "albumentations": ALBUMENTATIONS_AVAILABLE,
        "total_transforms": len(CEDAugmentationPipeline()._build_augmentation_pool()),
    }


__all__ = [
    "CEDAugmentationPipeline",
    "build_ced_transforms",
    "check_dependencies",
    "AUGLY_AVAILABLE",
    "ALBUMENTATIONS_AVAILABLE",
]
