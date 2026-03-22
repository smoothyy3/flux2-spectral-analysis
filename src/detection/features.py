"""Feature extraction from radial power spectra for classification.

Converts 1-D radial power spectra into fixed-length feature vectors
suitable for scikit-learn classifiers.  The total feature vector has
133 dimensions: 128 interpolated log-spectrum bins + 3 band energies +
2 slope/intercept coefficients.
"""

from __future__ import annotations

import numpy as np

from src.spectral.metrics import band_energy_ratios, spectral_slope

_EPSILON = 1e-12


def extract_spectrum_features(
    radial_spectrum: np.ndarray,
    n_bins: int = 128,
) -> np.ndarray:
    """Downsample a radial power spectrum to a fixed number of log-bins.

    Uses linear interpolation (``np.interp``) to resample the spectrum from
    its native length to *n_bins* uniformly spaced frequency positions, then
    applies a log10 transform.

    Parameters
    ----------
    radial_spectrum : np.ndarray
        1-D radial power spectrum of length R.
    n_bins : int, optional
        Number of output bins.  Defaults to 128.

    Returns
    -------
    np.ndarray
        1-D feature array of length *n_bins*, dtype float64.
        Values are log10(interpolated_power + epsilon).
    """
    spectrum = radial_spectrum.astype(np.float64)
    r = len(spectrum)
    old_x = np.linspace(0, 1, r)
    new_x = np.linspace(0, 1, n_bins)
    interpolated = np.interp(new_x, old_x, spectrum)
    return np.log10(np.maximum(interpolated, 1.0))


def extract_band_features(
    radial_spectrum: np.ndarray,
    n_bands: int = 3,
) -> np.ndarray:
    """Compute band energy ratio features from a radial power spectrum.

    Parameters
    ----------
    radial_spectrum : np.ndarray
        1-D radial power spectrum.
    n_bands : int, optional
        Number of frequency bands.  Defaults to 3.

    Returns
    -------
    np.ndarray
        1-D array of *n_bands* energy fraction values, dtype float64.
    """
    return band_energy_ratios(radial_spectrum, n_bands=n_bands)


def extract_slope_feature(radial_spectrum: np.ndarray) -> np.ndarray:
    """Extract spectral slope and intercept as a 2-element feature vector.

    Parameters
    ----------
    radial_spectrum : np.ndarray
        1-D radial power spectrum.

    Returns
    -------
    np.ndarray
        Array of shape (2,) containing [slope, intercept], dtype float64.
    """
    slope, intercept = spectral_slope(radial_spectrum)
    return np.array([slope, intercept], dtype=np.float64)


def extract_all_features(
    radial_spectrum: np.ndarray,
    n_bins: int = 128,
    n_bands: int = 3,
) -> np.ndarray:
    """Build the full 133-dimensional feature vector for one spectrum.

    The feature vector concatenates:

    * 128 interpolated log-spectrum bins (``extract_spectrum_features``)
    * *n_bands* band energy ratios (``extract_band_features``)
    * 2 slope/intercept values (``extract_slope_feature``)

    Parameters
    ----------
    radial_spectrum : np.ndarray
        1-D radial power spectrum.
    n_bins : int, optional
        Number of spectrum bins.  Defaults to 128.
    n_bands : int, optional
        Number of frequency bands.  Defaults to 3.

    Returns
    -------
    np.ndarray
        1-D float64 feature array of length n_bins + n_bands + 2.
    """
    spectrum_feats = extract_spectrum_features(radial_spectrum, n_bins=n_bins)
    band_feats = extract_band_features(radial_spectrum, n_bands=n_bands)
    slope_feats = extract_slope_feature(radial_spectrum)
    return np.concatenate([spectrum_feats, band_feats, slope_feats])


def build_feature_matrix(
    real_spectra: np.ndarray,
    gen_spectra: np.ndarray,
    n_bins: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a labelled feature matrix for real vs. generated classification.

    Extracts features for every spectrum in *real_spectra* (label = 0) and
    *gen_spectra* (label = 1), then concatenates them into a single matrix.

    Parameters
    ----------
    real_spectra : np.ndarray
        Array of shape (N_real, R) containing radial spectra of real images.
    gen_spectra : np.ndarray
        Array of shape (N_gen, R) containing radial spectra of generated images.
    n_bins : int, optional
        Number of spectrum bins passed to ``extract_all_features``.
        Defaults to 128.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (N_real + N_gen, 133), dtype float64.
    y : np.ndarray
        Label vector of shape (N_real + N_gen,) with values 0 (real) or
        1 (generated).
    """
    real_features = np.stack(
        [extract_all_features(real_spectra[i], n_bins=n_bins)
         for i in range(len(real_spectra))],
        axis=0,
    )
    gen_features = np.stack(
        [extract_all_features(gen_spectra[i], n_bins=n_bins)
         for i in range(len(gen_spectra))],
        axis=0,
    )

    X = np.concatenate([real_features, gen_features], axis=0)
    y = np.concatenate([
        np.zeros(len(real_features), dtype=np.int64),
        np.ones(len(gen_features), dtype=np.int64),
    ])
    return X, y
