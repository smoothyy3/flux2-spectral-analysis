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
    """Normalise a log10-power spectrum as a probability distribution.

    Converts from log10-power space to linear, floors at epsilon, and
    normalises to sum to 1.  KL divergence and Wasserstein distance require
    proper probability distributions, so linear conversion is necessary.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D radial spectrum array in log10-power space.

    Returns
    -------
    np.ndarray
        Normalised linear-power distribution of the same shape, summing to
        1, dtype float64.
    """
    linear = 10 ** spectrum.astype(np.float64)
    linear = np.maximum(linear, _EPSILON)
    return linear / linear.sum()


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

        This is one-directional (asymmetric): KL(real || gen) measures how
        many extra bits are needed to encode the real distribution using a
        code optimised for the generated distribution.
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
    """Compute the L2 distance between two log10-power spectra.

    Parameters
    ----------
    p : np.ndarray
        Reference (real) spectrum in log10-power space, 1-D array.
    q : np.ndarray
        Comparison (generated) spectrum in log10-power space, 1-D array.

    Returns
    -------
    float
        L2 norm of (p - q) in log10-power space.
    """
    return float(np.linalg.norm(p.astype(np.float64) - q.astype(np.float64)))


def band_energy_ratios(spectrum: np.ndarray, n_bands: int = 3) -> np.ndarray:
    """Split the spectrum into equal bands and return mean log₁₀ power per band.

    Natural image spectra follow a 1/f² law, so low-frequency bins carry
    >99.9% of total linear energy even after excluding the DC component.
    Fractions of total linear energy are therefore uninformative (mid and
    high bands always round to 0.000).  Instead this function returns the
    mean log₁₀ power in each band, which is well-distributed and directly
    comparable between groups.

    DC (bin 0) is excluded from all bands.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D power spectrum array, non-negative. Index 0 is the DC component.
    n_bands : int, optional
        Number of equal-width bands. Defaults to 3 (low / mid / high).

    Returns
    -------
    np.ndarray
        Shape (n_bands,). Mean log₁₀(power + ε) for each band.
    """
    spectrum = spectrum.astype(np.float64)
    log_spec = spectrum[1:]  # already log10-power; exclude DC at index 0
    n = len(log_spec)
    band_size = n // n_bands
    result = np.zeros(n_bands, dtype=np.float64)
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else n
        result[i] = log_spec[start:end].mean()
    return result


def spectral_slope(
    spectrum: np.ndarray,
    bin_start: int = 1,
    bin_end: int | None = None,
) -> tuple[float, float]:
    """Fit a power-law slope to the radial power spectrum in log-log space.

    The fit is performed as::

        log10(power) ~ slope * log10(freq) + intercept

    using ``np.polyfit`` (degree 1).  Natural images typically exhibit a
    slope near -2.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D power spectrum array.  Index 0 is the DC component.
    bin_start : int, optional
        First bin index to include in the fit.  Defaults to 1 (skips DC).
        Use 10 to also exclude near-DC artefacts that inflate variance.
    bin_end : int or None, optional
        One-past-last bin index.  ``None`` uses the full spectrum length.
        Use 400 to exclude Nyquist-edge noise on 512-bin spectra.

    Returns
    -------
    tuple of float
        ``(slope, intercept)`` from the log-log linear fit.
    """
    spectrum = spectrum.astype(np.float64)
    end = min(bin_end, len(spectrum)) if bin_end is not None else len(spectrum)
    start = max(bin_start, 1)  # always skip DC (log10(0) is undefined)

    freqs = np.arange(start, end, dtype=np.float64)
    # Spectrum is already in log10-power space.  Floor at 0.0 = log10(1):
    # near-Nyquist bins with near-zero power produce log10(ε) ≈ -12, which
    # would corrupt the linear fit.
    log_p = np.maximum(spectrum[start:end], 0.0)
    log_f = np.log10(freqs)

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
    real_sl, real_ic = spectral_slope(real_mean, bin_start=10, bin_end=400)
    gen_sl, gen_ic = spectral_slope(gen_mean, bin_start=10, bin_end=400)

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
