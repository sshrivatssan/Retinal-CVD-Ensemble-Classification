from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import binary_opening, binary_closing, disk


# Supported image formats
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# CLAHE
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)


def list_images(root: Path):
    """Recursively list supported image files."""
    return sorted(
        [p for p in root.rglob("*") if p.suffix.lower() in EXTS]
    )


def green_clahe_minmax_from_bgr(bgr, clahe_obj):
    """Apply green-channel extraction, CLAHE, and min-max normalization."""

    # Extract green channel
    g = bgr[:, :, 1]

    # Enhance local contrast
    g = clahe_obj.apply(g)

    # Min-max normalize to [0, 1]
    g = g.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)

    # Convert to 8-bit image
    g8 = (g * 255.0).round().astype(np.uint8)

    return g8


def make_mask_from_processed(g8: np.ndarray):
    """Generate a binary fundus mask from a processed image."""

    # Scale intensities to [0, 1]
    g = g8.astype(np.float32) / 255.0

    # Smooth before thresholding
    blur = cv2.GaussianBlur(g, (31, 31), 0)

    # Segment fundus from dark background
    mask = (blur > 0.05).astype(np.uint8)

    # Remove small isolated regions
    mask = binary_opening(mask, disk(3)).astype(np.uint8)

    # Fill small gaps in the mask
    mask = binary_closing(mask, disk(5)).astype(np.uint8)

    return g, mask