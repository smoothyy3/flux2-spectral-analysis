"""Image degradation controls for spectral analysis.

These functions apply classical image degradation transformations to
float64 grayscale arrays so that their spectral signatures can be
compared with those of AI-generated images.  All functions return
arrays in the same format as the input: float64, values in [0, 1].
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def blur_gaussian(img: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """Apply Gaussian blur to a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image of shape (H, W), dtype float64, values in [0, 1].
    radius : float, optional
        Standard deviation (sigma) of the Gaussian kernel.  Defaults to 1.5.

    Returns
    -------
    np.ndarray
        Blurred grayscale image of shape (H, W), dtype float64, values in
        [0, 1].
    """
    blurred = gaussian_filter(img, sigma=radius)
    return blurred.astype(np.float64)


def downscale_upscale(img: np.ndarray, intermediate_size: int = 512) -> np.ndarray:
    """Downscale then upscale an image to simulate resolution loss.

    Converts the array to a uint8 PIL Image, downscales to
    *intermediate_size* × *intermediate_size* with LANCZOS resampling,
    then upscales back to the original dimensions, also with LANCZOS.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image of shape (H, W), dtype float64, values in [0, 1].
    intermediate_size : int, optional
        Side length of the intermediate downscaled image.  Defaults to 512.

    Returns
    -------
    np.ndarray
        Processed grayscale image of shape (H, W), dtype float64, values
        in [0, 1].
    """
    h, w = img.shape
    uint8_img = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img, mode="L")

    small = pil_img.resize((intermediate_size, intermediate_size), Image.LANCZOS)
    restored = small.resize((w, h), Image.LANCZOS)

    result = np.array(restored, dtype=np.float64) / 255.0
    return result


def jpeg_compress(img: np.ndarray, quality: int = 85) -> np.ndarray:
    """Simulate JPEG compression artefacts on a grayscale image.

    Converts the array to a uint8 PIL Image, encodes it as JPEG in memory
    at the specified quality, then decodes the result back to a float64
    array.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image of shape (H, W), dtype float64, values in [0, 1].
    quality : int, optional
        JPEG compression quality, 1–95.  Defaults to 85.

    Returns
    -------
    np.ndarray
        JPEG-compressed grayscale image of shape (H, W), dtype float64,
        values in [0, 1].
    """
    uint8_img = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img, mode="L")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = Image.open(buf).convert("L")

    result = np.array(decoded, dtype=np.float64) / 255.0
    return result


def apply_degradation(
    images: list[np.ndarray],
    method: str,
) -> list[np.ndarray]:
    """Apply a named degradation method to a list of images.

    Dispatches to the appropriate degradation function based on *method*.

    Supported method strings
    ------------------------
    ``"blur_1.5"``
        Gaussian blur with sigma = 1.5.
    ``"downup_512"``
        Downscale to 512 then upscale to original size.
    ``"jpeg_85"``
        JPEG compression at quality = 85.

    Parameters
    ----------
    images : list of np.ndarray
        Grayscale images to degrade, each of shape (H, W) float64.
    method : str
        Degradation method identifier (see above).

    Returns
    -------
    list of np.ndarray
        List of degraded images in the same order as *images*.

    Raises
    ------
    ValueError
        If *method* is not one of the supported identifiers.
    """
    supported = {"blur_1.5", "downup_512", "jpeg_85"}
    if method not in supported:
        raise ValueError(
            f"Unknown degradation method '{method}'. "
            f"Supported methods: {supported}"
        )

    degraded: list[np.ndarray] = []
    for img in tqdm(images, desc=f"Degrading [{method}]", unit="img"):
        if method == "blur_1.5":
            degraded.append(blur_gaussian(img, radius=1.5))
        elif method == "downup_512":
            degraded.append(downscale_upscale(img, intermediate_size=512))
        elif method == "jpeg_85":
            degraded.append(jpeg_compress(img, quality=85))

    return degraded
