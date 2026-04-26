"""Spectral I/O — load images and compute radial spectra with no caching.

Replaces the previous cache-based load_or_compute_spectra. Every call
recomputes from source images. For 200 images at 1024×1024 this takes
roughly 8 seconds — fast enough that caching adds complexity with no
real benefit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.spectral.preprocessing import load_images_from_dir
from src.spectral.fft import batch_log_power_spectra
from src.spectral.azimuthal import batch_azimuthal_average


def compute_spectra(directory: Path) -> np.ndarray:
    """Load all images from *directory* and return their radial log-power spectra.

    Implements the Keuper et al. (CVPR 2020) azimuthal averaging pipeline:
    log10(|F(u,v)|² + ε) is computed for each pixel of the 2-D DFT *before*
    radial averaging.  The returned array therefore contains mean log10-power
    per radial bin per image, not mean linear power.

    Parameters
    ----------
    directory : Path
        Directory containing source images (PNG/JPEG).

    Returns
    -------
    np.ndarray
        Shape (N, R) — one radial log10-power spectrum per image.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no images are found.
    """
    images = load_images_from_dir(directory)
    log_power = batch_log_power_spectra(images)
    return batch_azimuthal_average(log_power)
