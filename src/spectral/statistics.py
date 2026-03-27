"""Population-level statistics over collections of radial spectra.

Functions in this module operate on (N, R) arrays where N is the number
of images and R is the number of radial frequency bins.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def population_stats(
    radial_spectra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frequency mean and standard deviation across a population.

    Parameters
    ----------
    radial_spectra : np.ndarray
        Array of shape (N, R) containing individual radial power spectra.
        N = number of images, R = number of frequency bins.

    Returns
    -------
    mean : np.ndarray
        Per-frequency mean spectrum of shape (R,), dtype float64.
    std : np.ndarray
        Per-frequency standard deviation of shape (R,), dtype float64.
    """
    mean = np.mean(radial_spectra, axis=0)
    std = np.std(radial_spectra, axis=0, ddof=1)
    return mean.astype(np.float64), std.astype(np.float64)


_LOG_FLOOR = 1.0  # floor before log10 — must match value used in visualization


def per_frequency_ttest(
    spectra_a: np.ndarray,
    spectra_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an independent-samples Welch t-test at every frequency bin.

    Operates on log10-transformed power values.  Raw linear power spans
    ~10^0 to ~10^11 and is heavily right-skewed; the t-test's normality
    assumption holds far better in log space.

    p-values are NOT corrected for multiple comparisons across ~512 bins.
    Bonferroni correction (α/512 ≈ 1e-4) would be highly conservative.
    Uncorrected p-values are suitable for exploratory visualization but
    should not be interpreted as formal per-bin hypothesis tests.

    Parameters
    ----------
    spectra_a : np.ndarray
        Array of shape (N_A, R) for group A (e.g. real images).
    spectra_b : np.ndarray
        Array of shape (N_B, R) for group B (e.g. generated images).

    Returns
    -------
    t_stats : np.ndarray
        t-statistic at each frequency bin, shape (R,).
    p_values : np.ndarray
        Two-tailed p-value at each frequency bin, shape (R,).
    """
    # Log-transform before testing — raw linear power is too skewed for t-test.
    log_a = np.log10(np.maximum(spectra_a.astype(np.float64), _LOG_FLOOR))
    log_b = np.log10(np.maximum(spectra_b.astype(np.float64), _LOG_FLOOR))

    r = log_a.shape[1]
    t_stats = np.zeros(r, dtype=np.float64)
    p_values = np.zeros(r, dtype=np.float64)

    for i in range(r):
        result = sp_stats.ttest_ind(log_a[:, i], log_b[:, i], equal_var=False)
        t_stats[i] = result.statistic
        p_values[i] = result.pvalue

    return t_stats, p_values


def significant_frequencies(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Identify frequency bins with statistically significant differences.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values, one per frequency bin, shape (R,).
    alpha : float, optional
        Significance threshold.  Defaults to 0.05.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (R,) where ``True`` indicates p_value < alpha.
    """
    return p_values < alpha


def confidence_interval_95(
    spectra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 95 % confidence intervals for each frequency bin.

    Uses the normal approximation: mean ± 1.96 * std / sqrt(N).

    Parameters
    ----------
    spectra : np.ndarray
        Array of shape (N, R) containing individual radial power spectra.

    Returns
    -------
    lower : np.ndarray
        Lower 95 % CI bound, shape (R,).
    upper : np.ndarray
        Upper 95 % CI bound, shape (R,).
    """
    n = spectra.shape[0]
    mean = np.mean(spectra, axis=0)
    std = np.std(spectra, axis=0, ddof=1)
    margin = 1.96 * std / np.sqrt(n)
    lower = mean - margin
    upper = mean + margin
    return lower.astype(np.float64), upper.astype(np.float64)
