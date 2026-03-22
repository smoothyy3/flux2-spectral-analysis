"""2-D Fast Fourier Transform utilities for power spectrum computation.

This module computes 2-D power spectra from grayscale image arrays and
provides batch and averaging helpers for downstream azimuthal analysis.
"""

from __future__ import annotations

import numpy as np


def compute_power_spectrum_2d(img: np.ndarray) -> np.ndarray:
    """Compute the 2-D power spectrum of a grayscale image.

    Applies a 2-D FFT via ``np.fft.fft2``, shifts the zero-frequency
    component to the center with ``np.fft.fftshift``, and returns the
    squared magnitude (power).

    Parameters
    ----------
    img : np.ndarray
        Grayscale image array of shape (H, W), dtype float64, values in
        [0, 1].

    Returns
    -------
    np.ndarray
        2-D power spectrum of shape (H, W), dtype float64.  Each element
        is |F(u, v)|^2 where F is the centred DFT.
    """
    f = np.fft.fft2(img)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2
    return power.astype(np.float64)


def batch_power_spectra(images: list[np.ndarray]) -> np.ndarray:
    """Compute 2-D power spectra for a list of images.

    Parameters
    ----------
    images : list of np.ndarray
        List of N grayscale images, each of shape (H, W).

    Returns
    -------
    np.ndarray
        Stacked power spectra of shape (N, H, W), dtype float64.

    Raises
    ------
    ValueError
        If *images* is empty.
    """
    if not images:
        raise ValueError("images list must not be empty.")
    spectra = [compute_power_spectrum_2d(img) for img in images]
    return np.stack(spectra, axis=0)


def average_power_spectrum_2d(spectra: np.ndarray) -> np.ndarray:
    """Compute the mean 2-D power spectrum across a batch.

    Parameters
    ----------
    spectra : np.ndarray
        Array of shape (N, H, W) containing individual power spectra.

    Returns
    -------
    np.ndarray
        Mean power spectrum of shape (H, W), dtype float64.
    """
    return np.mean(spectra, axis=0)
