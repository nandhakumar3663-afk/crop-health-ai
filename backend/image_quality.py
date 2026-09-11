"""
backend/image_quality.py - Image Quality Gatekeeper
Validates leaf images prior to running deep learning inference to reject invalid,
blurry, overexposed, or pitch-black images.
"""

from typing import Tuple
from PIL import Image, ImageStat
import numpy as np


def validate_leaf_image(
    image: Image.Image,
    min_size: int = 64,
    min_std: float = 12.0,
    min_mean: float = 15.0,
    max_mean: float = 245.0
) -> Tuple[bool, str]:
    """
    Validates leaf image quality.
    Returns:
        (is_valid: bool, error_message: str)
    """
    if image is None:
        return False, "No image provided."

    width, height = image.size

    # 1. Dimension checks
    if width < min_size or height < min_size:
        return False, f"Image resolution is too small ({width}x{height}). Please upload an image of at least {min_size}x{min_size} pixels."

    aspect_ratio = max(width, height) / max(min(width, height), 1)
    if aspect_ratio > 8.0:
        return False, "Image aspect ratio is too distorted. Please upload a standard leaf photograph."

    # 2. Convert to grayscale for illumination & contrast analysis
    try:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        mean_brightness = stat.mean[0]
        std_contrast = stat.stddev[0]
    except Exception as e:
        return False, f"Could not inspect image data: {str(e)}"

    # 3. Extreme darkness / underexposure
    if mean_brightness < min_mean:
        return False, "Image quality is too low. Image is too dark or underexposed. Please capture a clear close-up in good lighting."

    # 4. Extreme brightness / overexposure
    if mean_brightness > max_mean:
        return False, "Image quality is too low. Image is overexposed or blown out. Please capture a clear close-up of a single leaf."

    # 5. Low contrast / solid blank image / extreme blur
    if std_contrast < min_std:
        return False, "Image quality is too low. Image lacks sufficient detail or contrast. Please capture a clear close-up of a single leaf."

    return True, ""
