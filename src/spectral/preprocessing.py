"""Image loading and preprocessing utilities for spectral analysis.

This module provides functions to load grayscale images from disk and
validate their shapes before spectral decomposition.

Normalization convention
------------------------
Images are normalized to [0, 1] (float64) before FFT, i.e. divided by 255.
This changes absolute power levels compared to Keuper et al. (2020), who
operate on raw [0, 255] uint8 values.  Relative spectral shapes and all
comparative metrics (slope, Wasserstein, band ratios) are unaffected by this
scalar factor because the factor cancels in ratios and log-differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image_as_gray(path: Path) -> np.ndarray:
    """Load a single image file and convert it to a float64 grayscale array.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the image file (PNG, JPEG, etc.).

    Returns
    -------
    np.ndarray
        Grayscale image array of shape (H, W) with dtype float64 and
        values in the range [0, 1].

    Raises
    ------
    FileNotFoundError
        If the file at *path* does not exist.
    OSError
        If Pillow cannot open the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float64) / 255.0
    return arr


def load_images_from_dir(
    directory: Path,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> list[np.ndarray]:
    """Load all images from a directory as float64 grayscale arrays.

    Files are sorted lexicographically for reproducibility. Subdirectories
    are ignored; only files whose suffix matches *extensions* are loaded.

    Parameters
    ----------
    directory : Path
        Directory to search for image files.
    extensions : tuple of str, optional
        File extensions to include (case-insensitive). Defaults to
        ``(".png", ".jpg", ".jpeg")``.

    Returns
    -------
    list of np.ndarray
        Sorted list of grayscale image arrays, each of shape (H, W) with
        dtype float64 and values in [0, 1].

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    ValueError
        If no matching image files are found.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    lower_ext = tuple(e.lower() for e in extensions)
    paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in lower_ext
    )

    if not paths:
        raise ValueError(
            f"No images with extensions {extensions} found in {directory}"
        )

    images: list[np.ndarray] = []
    for p in tqdm(paths, desc=f"Loading images from {directory.name}", unit="img"):
        images.append(load_image_as_gray(p))

    return images


def validate_image_shape(img: np.ndarray, expected_size: int = 1024) -> bool:
    """Check whether an image array has the expected square dimensions.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image array of shape (H, W).
    expected_size : int, optional
        Expected side length in pixels. Defaults to 1024.

    Returns
    -------
    bool
        ``True`` if ``img.shape == (expected_size, expected_size)``,
        ``False`` otherwise.
    """
    return img.ndim == 2 and img.shape == (expected_size, expected_size)
