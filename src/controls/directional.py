"""Directional and 2-D spectral analysis utilities.

Provides functions to extract directional profiles from 2-D power spectra,
compute anisotropy ratios, and detect periodic artefacts.
"""

from __future__ import annotations

import numpy as np


_EPSILON = 1e-12


def log_power_spectrum_2d(avg_spectrum: np.ndarray) -> np.ndarray:
    """Compute the log10 of a 2-D average power spectrum for visualisation.

    A small epsilon is added before taking the logarithm to avoid
    log(0) = -inf.

    Parameters
    ----------
    avg_spectrum : np.ndarray
        2-D average power spectrum of shape (H, W), non-negative float64.

    Returns
    -------
    np.ndarray
        log10(avg_spectrum + epsilon) of shape (H, W), dtype float64.
    """
    return np.log10(avg_spectrum.astype(np.float64) + _EPSILON)


def compute_horizontal_profile(spectrum_2d: np.ndarray) -> np.ndarray:
    """Extract the horizontal slice through the centre row of a 2-D spectrum.

    The centre row corresponds to the DC row after ``np.fft.fftshift`` and
    contains the horizontal frequency profile of the image.

    Parameters
    ----------
    spectrum_2d : np.ndarray
        2-D power spectrum of shape (H, W).

    Returns
    -------
    np.ndarray
        1-D array of length W containing the horizontal frequency profile.
    """
    cy = spectrum_2d.shape[0] // 2
    return spectrum_2d[cy, :].copy().astype(np.float64)


def compute_vertical_profile(spectrum_2d: np.ndarray) -> np.ndarray:
    """Extract the vertical slice through the centre column of a 2-D spectrum.

    The centre column corresponds to the DC column after ``np.fft.fftshift``
    and contains the vertical frequency profile of the image.

    Parameters
    ----------
    spectrum_2d : np.ndarray
        2-D power spectrum of shape (H, W).

    Returns
    -------
    np.ndarray
        1-D array of length H containing the vertical frequency profile.
    """
    cx = spectrum_2d.shape[1] // 2
    return spectrum_2d[:, cx].copy().astype(np.float64)


def anisotropy_ratio(spectrum_2d: np.ndarray) -> float:
    """Compute the ratio of horizontal to vertical energy in a 2-D spectrum.

    The DC row and column are excluded.  The ratio is defined as::

        ratio = horizontal_energy / (vertical_energy + epsilon)

    A value near 1.0 indicates an isotropic spectrum; values significantly
    above or below 1.0 indicate directional bias.

    Parameters
    ----------
    spectrum_2d : np.ndarray
        2-D power spectrum of shape (H, W), centred (DC at H//2, W//2).

    Returns
    -------
    float
        Ratio of horizontal to vertical half-plane energy, excluding DC.
    """
    h, w = spectrum_2d.shape
    cy, cx = h // 2, w // 2

    # Horizontal half: all columns, rows above and below DC row (excluding DC row itself)
    horizontal_mask = np.zeros((h, w), dtype=bool)
    horizontal_mask[:cy, :] = True
    horizontal_mask[cy + 1:, :] = True

    # Vertical half: all rows, columns left and right of DC column (excluding DC col)
    vertical_mask = np.zeros((h, w), dtype=bool)
    vertical_mask[:, :cx] = True
    vertical_mask[:, cx + 1:] = True

    h_energy = spectrum_2d[horizontal_mask].sum()
    v_energy = spectrum_2d[vertical_mask].sum()

    return float(h_energy / (v_energy + _EPSILON))


def detect_periodic_peaks(
    spectrum_2d: np.ndarray,
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """Detect pixels in the log-spectrum that are anomalously bright.

    Computes the log10 of the 2-D spectrum and finds pixels whose value
    exceeds the global mean by more than *threshold_sigma* standard
    deviations.  Such peaks often correspond to periodic artefacts or
    grid-like patterns in the spatial domain.

    Parameters
    ----------
    spectrum_2d : np.ndarray
        2-D power spectrum of shape (H, W), non-negative float64.
    threshold_sigma : float, optional
        Number of standard deviations above the mean required to flag a
        pixel as a periodic peak.  Defaults to 3.0.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (H, W).  ``True`` indicates a flagged peak.
    """
    log_spec = log_power_spectrum_2d(spectrum_2d)
    mean = log_spec.mean()
    std = log_spec.std()
    threshold = mean + threshold_sigma * std
    return log_spec > threshold
