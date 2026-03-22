"""Azimuthal averaging of 2-D power spectra.

Converts 2-D power spectra to 1-D radially averaged spectra using fast
vectorised operations.  The centre of the spectrum is assumed to be at
the pixel coordinates (H//2, W//2), consistent with ``np.fft.fftshift``.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def _build_radius_map(h: int, w: int) -> np.ndarray:
    """Build an integer radius map for a centred 2-D array.

    Each element contains the Euclidean distance (rounded to the nearest
    integer) from the array centre at (h//2, w//2).

    Parameters
    ----------
    h : int
        Height of the array in pixels.
    w : int
        Width of the array in pixels.

    Returns
    -------
    np.ndarray
        Integer radius array of shape (H, W), dtype int32.
    """
    rows, cols = np.indices((h, w))
    cy, cx = h // 2, w // 2
    radii = np.round(np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)).astype(np.int32)
    return radii


def azimuthal_average(spectrum_2d: np.ndarray) -> np.ndarray:
    """Compute the radially averaged (azimuthal) 1-D power spectrum.

    Only pixels whose integer radius is <= max_radius contribute to the
    average, where max_radius = min(H, W) // 2.  The DC component sits at
    radius 0.

    Uses ``np.bincount`` for performance instead of looping over radii.

    Parameters
    ----------
    spectrum_2d : np.ndarray
        2-D power spectrum of shape (H, W), dtype float64.

    Returns
    -------
    np.ndarray
        1-D radially averaged power spectrum of length (max_radius + 1,),
        dtype float64.  Index 0 corresponds to the DC component.
    """
    h, w = spectrum_2d.shape
    max_radius = min(h, w) // 2

    radii = _build_radius_map(h, w)
    mask = radii <= max_radius

    flat_radii = radii[mask]
    flat_power = spectrum_2d[mask]

    # Sum of power values in each radial bin
    radial_sum = np.bincount(flat_radii, weights=flat_power, minlength=max_radius + 1)
    # Number of pixels in each radial bin
    radial_count = np.bincount(flat_radii, minlength=max_radius + 1)

    # Avoid division by zero (bin 0 always has at least 1 pixel)
    radial_count = np.maximum(radial_count, 1)
    radial_avg = radial_sum / radial_count

    return radial_avg.astype(np.float64)


def batch_azimuthal_average(spectra: np.ndarray) -> np.ndarray:
    """Compute azimuthal averages for a batch of 2-D power spectra.

    Parameters
    ----------
    spectra : np.ndarray
        Array of shape (N, H, W) containing individual power spectra.

    Returns
    -------
    np.ndarray
        Array of shape (N, R) where R = min(H, W) // 2 + 1, containing
        radially averaged spectra for each image, dtype float64.
    """
    n = spectra.shape[0]
    results: list[np.ndarray] = []
    for i in tqdm(range(n), desc="Azimuthal averaging", unit="img"):
        results.append(azimuthal_average(spectra[i]))
    return np.stack(results, axis=0)
