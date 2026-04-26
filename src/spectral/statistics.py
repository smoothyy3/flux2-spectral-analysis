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


def per_frequency_ttest(
    spectra_a: np.ndarray,
    spectra_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an independent-samples Welch t-test at every frequency bin.

    Expects input spectra already in log10-power space (as returned by
    ``compute_spectra``).  The t-test's normality assumption holds well for
    sample means of log-power values across images.

    p-values are NOT corrected for multiple comparisons across ~512 bins.
    Bonferroni correction (α/512 ≈ 1e-4) would be highly conservative.
    Uncorrected p-values are suitable for exploratory visualization but
    should not be interpreted as formal per-bin hypothesis tests.

    Parameters
    ----------
    spectra_a : np.ndarray
        Array of shape (N_A, R) for group A (e.g. real images),
        values in log10-power space.
    spectra_b : np.ndarray
        Array of shape (N_B, R) for group B (e.g. generated images),
        values in log10-power space.

    Returns
    -------
    t_stats : np.ndarray
        t-statistic at each frequency bin, shape (R,).
    p_values : np.ndarray
        Two-tailed p-value at each frequency bin, shape (R,).
    """
    log_a = spectra_a.astype(np.float64)
    log_b = spectra_b.astype(np.float64)

    r = log_a.shape[1]
    t_stats = np.zeros(r, dtype=np.float64)
    p_values = np.zeros(r, dtype=np.float64)

    for i in range(r):
        result = sp_stats.ttest_ind(log_a[:, i], log_b[:, i], equal_var=False)
        t_stats[i] = result.statistic
        p_values[i] = result.pvalue

    return t_stats, p_values


def per_frequency_paired_ttest(
    spectra_a: np.ndarray,
    spectra_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a paired-samples t-test at every frequency bin.

    Expects input spectra already in log10-power space (as returned by
    ``compute_spectra``).  Requires equal sample sizes (N_A == N_B) and
    that ``spectra_a[i]`` is paired with ``spectra_b[i]`` (e.g. the same
    image before and after a round-trip transformation).

    A paired test is more powerful than an independent-samples test when
    within-pair correlation is high, as in encode-decode experiments.

    p-values are NOT corrected for multiple comparisons across ~512 bins.

    Parameters
    ----------
    spectra_a : np.ndarray
        Array of shape (N, R) for group A, values in log10-power space.
    spectra_b : np.ndarray
        Array of shape (N, R) for group B, values in log10-power space.
        Must have the same shape as *spectra_a*.

    Returns
    -------
    t_stats : np.ndarray
        t-statistic at each frequency bin, shape (R,).
    p_values : np.ndarray
        Two-tailed p-value at each frequency bin, shape (R,).

    Raises
    ------
    ValueError
        If *spectra_a* and *spectra_b* do not have the same shape.
    """
    a = spectra_a.astype(np.float64)
    b = spectra_b.astype(np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"spectra_a and spectra_b must have the same shape for a paired "
            f"test. Got {a.shape} and {b.shape}."
        )

    r = a.shape[1]
    t_stats = np.zeros(r, dtype=np.float64)
    p_values = np.zeros(r, dtype=np.float64)

    for i in range(r):
        result = sp_stats.ttest_rel(a[:, i], b[:, i])
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
