"""Spectral similarity and characterisation metrics.

All public functions operate on 1-D numpy arrays representing mean radial
power spectra.  Normalisation is handled internally where required.
"""

from __future__ import annotations

import numpy as np
from scipy import special as sp_special
from scipy import stats as sp_stats


_EPSILON = 1e-12


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Normalise a power spectrum so that its values sum to 1.

    A small epsilon is added before normalisation to avoid zero values
    that would cause problems in KL divergence or log operations.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D power spectrum array, non-negative values.

    Returns
    -------
    np.ndarray
        Normalised spectrum of the same shape, summing to 1, dtype float64.
    """
    s = spectrum.astype(np.float64) + _EPSILON
    return s / s.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the KL divergence KL(p || q) between two spectra.

    Both spectra are normalised internally before computing the divergence.

    Parameters
    ----------
    p : np.ndarray
        Reference (real) power spectrum, 1-D array.
    q : np.ndarray
        Comparison (generated) power spectrum, 1-D array.

    Returns
    -------
    float
        KL(p || q) = sum(p * log(p / q)).  Always >= 0; equals 0 iff p == q.
    """
    p_norm = normalize_spectrum(p)
    q_norm = normalize_spectrum(q)
    return float(np.sum(sp_special.rel_entr(p_norm, q_norm)))


def wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Wasserstein-1 (earth mover's) distance between two spectra.

    Treats the frequency bin index as the "position" along the real line and
    the normalised spectrum value as the "weight" at each position.  Both
    spectra are normalised internally.

    Parameters
    ----------
    p : np.ndarray
        Reference (real) power spectrum, 1-D array.
    q : np.ndarray
        Comparison (generated) power spectrum, 1-D array.

    Returns
    -------
    float
        Wasserstein-1 distance between the two distributions.
    """
    p_norm = normalize_spectrum(p)
    q_norm = normalize_spectrum(q)
    positions = np.arange(len(p_norm), dtype=np.float64)
    return float(sp_stats.wasserstein_distance(positions, positions, p_norm, q_norm))


def l2_log_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the L2 distance between the log10 power spectra.

    Parameters
    ----------
    p : np.ndarray
        Reference (real) power spectrum, 1-D array, positive values.
    q : np.ndarray
        Comparison (generated) power spectrum, 1-D array, positive values.

    Returns
    -------
    float
        L2 norm of (log10(p) - log10(q)).
    """
    log_p = np.log10(p.astype(np.float64) + _EPSILON)
    log_q = np.log10(q.astype(np.float64) + _EPSILON)
    return float(np.linalg.norm(log_p - log_q))


def band_energy_ratios(spectrum: np.ndarray, n_bands: int = 3) -> np.ndarray:
    """Split the spectrum into equal frequency bands and return energy fractions.

    The DC component (bin 0) is excluded before splitting because its energy
    typically exceeds all other bins combined, which would make mid and high
    band fractions round to zero.  The remaining bins are divided into
    *n_bands* contiguous, equal-length segments.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D power spectrum array (e.g. radially averaged), non-negative.
        Index 0 is the DC component and is excluded from the calculation.
    n_bands : int, optional
        Number of equal-width bands to split the spectrum into.
        Defaults to 3 (low / mid / high).

    Returns
    -------
    np.ndarray
        Array of shape (n_bands,) containing the energy fraction for each
        band.  Values sum to 1.
    """
    spectrum = spectrum.astype(np.float64)
    # Exclude DC bin — its energy (~10^11) dwarfs all other bins combined
    spec_no_dc = spectrum[1:]
    total = spec_no_dc.sum() + _EPSILON
    n = len(spec_no_dc)
    band_size = n // n_bands
    ratios = np.zeros(n_bands, dtype=np.float64)
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else n
        ratios[i] = spec_no_dc[start:end].sum() / total
    return ratios


def spectral_slope(spectrum: np.ndarray) -> tuple[float, float]:
    """Fit a power-law slope to the radial power spectrum in log-log space.

    The fit is performed as::

        log10(power) ~ slope * log10(freq) + intercept

    using ``np.polyfit`` (degree 1).  The DC component (freq = 0) is
    excluded because log10(0) is undefined.  Natural images typically
    exhibit a slope near -2.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D power spectrum array.  Index 0 is the DC component and is
        automatically skipped.

    Returns
    -------
    tuple of float
        ``(slope, intercept)`` from the log-log linear fit.
    """
    spectrum = spectrum.astype(np.float64)
    freqs = np.arange(1, len(spectrum), dtype=np.float64)  # skip DC
    # Use a floor of 1.0 so that near-zero high-frequency bins do not produce
    # log10(1e-12) = -12 values that corrupt the linear fit.
    power = np.maximum(spectrum[1:], 1.0)

    log_f = np.log10(freqs)
    log_p = np.log10(power)

    slope, intercept = np.polyfit(log_f, log_p, 1)
    return float(slope), float(intercept)


def compute_all_metrics(
    real_mean: np.ndarray,
    gen_mean: np.ndarray,
    n_bands: int = 3,
) -> dict:
    """Compute a comprehensive set of spectral comparison metrics.

    Parameters
    ----------
    real_mean : np.ndarray
        Mean radial power spectrum of the real image set, 1-D array.
    gen_mean : np.ndarray
        Mean radial power spectrum of the generated image set, 1-D array.
    n_bands : int, optional
        Number of frequency bands for band energy ratio computation.
        Defaults to 3.

    Returns
    -------
    dict
        Dictionary containing the following keys:

        ``kl_divergence`` : float
            KL(real || gen).
        ``wasserstein_distance`` : float
            Wasserstein-1 distance.
        ``l2_log_distance`` : float
            L2 distance in log10 space.
        ``real_band_energies`` : list of float
            Band energy fractions for the real spectrum.
        ``gen_band_energies`` : list of float
            Band energy fractions for the generated spectrum.
        ``real_slope`` : float
            Spectral slope of the real spectrum.
        ``gen_slope`` : float
            Spectral slope of the generated spectrum.
        ``real_intercept`` : float
            Spectral intercept of the real spectrum.
        ``gen_intercept`` : float
            Spectral intercept of the generated spectrum.
    """
    real_bands = band_energy_ratios(real_mean, n_bands=n_bands)
    gen_bands = band_energy_ratios(gen_mean, n_bands=n_bands)
    real_sl, real_ic = spectral_slope(real_mean)
    gen_sl, gen_ic = spectral_slope(gen_mean)

    return {
        "kl_divergence": kl_divergence(real_mean, gen_mean),
        "wasserstein_distance": wasserstein_distance(real_mean, gen_mean),
        "l2_log_distance": l2_log_distance(real_mean, gen_mean),
        "real_band_energies": real_bands.tolist(),
        "gen_band_energies": gen_bands.tolist(),
        "real_slope": real_sl,
        "gen_slope": gen_sl,
        "real_intercept": real_ic,
        "gen_intercept": gen_ic,
    }
