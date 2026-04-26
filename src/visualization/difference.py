"""Visualisation of spectral differences and statistical significance.

Provides two-panel difference plots and significance mask visualisations
that highlight where generated images differ from real images in frequency
space.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import apply_style, get_color

apply_style()



def plot_spectral_difference(
    real_mean: np.ndarray,
    gen_mean: np.ndarray,
    p_values: np.ndarray,
    output_path: Path,
    group_name: str = "generated",
) -> None:
    """Two-panel figure showing spectral overlay and signed log-space difference.

    Top panel
        Overlay of real and generated mean spectra in log10 scale.

    Bottom panel
        Signed difference (log10(gen) - log10(real)) with red shading where
        the generated image over-produces energy (difference > 0, p < 0.05)
        and blue shading where it under-produces (difference < 0, p < 0.05).
        A horizontal dashed line marks zero difference.

    Parameters
    ----------
    real_mean : np.ndarray
        Mean radial power spectrum of the real image set, 1-D array.
    gen_mean : np.ndarray
        Mean radial power spectrum of the generated image set, 1-D array.
    p_values : np.ndarray
        Per-frequency bin p-values from the t-test, same length as spectra.
    output_path : Path
        Destination file path for the saved figure.
    group_name : str, optional
        Label for the generated group.  Defaults to ``"generated"``.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Spectra are already in log10-power space.
    freqs = np.arange(len(real_mean))
    diff = gen_mean - real_mean
    sig_mask = p_values < 0.05

    real_color = get_color("real")
    gen_color = get_color(group_name)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ---- Top panel: overlay ------------------------------------------------
    ax_top.plot(freqs, real_mean, color=real_color, linewidth=1.5, label="Real")
    ax_top.plot(freqs, gen_mean, color=gen_color, linewidth=1.5,
                label=group_name, alpha=0.85)
    ax_top.set_ylabel("log₁₀(Power)")
    ax_top.set_title(f"Mean Radial Power Spectra — Real vs. {group_name}")
    ax_top.legend(loc="upper right", framealpha=0.8)

    # ---- Bottom panel: signed difference -----------------------------------
    ax_bot.axhline(y=0, color="#333333", linestyle="--", linewidth=1.0)
    ax_bot.plot(freqs, diff, color="#555555", linewidth=1.0, alpha=0.7)

    # Significant over-production (gen > real)
    over_mask = sig_mask & (diff > 0)
    if over_mask.any():
        ax_bot.fill_between(
            freqs, 0, diff,
            where=over_mask,
            color="#d7191c",
            alpha=0.45,
            label="Over-production (p<0.05)",
        )

    # Significant under-production (gen < real)
    under_mask = sig_mask & (diff < 0)
    if under_mask.any():
        ax_bot.fill_between(
            freqs, 0, diff,
            where=under_mask,
            color="#2c7bb6",
            alpha=0.45,
            label="Under-production (p<0.05)",
        )

    ax_bot.set_xlabel("Frequency bin (radial)")
    ax_bot.set_ylabel("Δ log₁₀(Power) [gen − real]")
    ax_bot.set_title(f"Signed Spectral Difference — {group_name}")

    handles, labels = ax_bot.get_legend_handles_labels()
    if handles:
        ax_bot.legend(loc="upper right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_significance_mask(
    p_values: np.ndarray,
    output_path: Path,
    alpha: float = 0.05,
) -> None:
    """Plot −log10(p-values) against frequency bin index.

    Frequency bins above the significance threshold (*alpha*) are visually
    indicated by a horizontal reference line at −log10(alpha).

    Parameters
    ----------
    p_values : np.ndarray
        Per-frequency bin p-values from a t-test, 1-D array.
    output_path : Path
        Destination file path for the saved figure.
    alpha : float, optional
        Significance threshold.  Defaults to 0.05.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    neg_log_p = -np.log10(np.maximum(p_values, 1e-300))
    threshold = -np.log10(alpha)
    freqs = np.arange(len(p_values))

    sig_mask = p_values < alpha

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(freqs, neg_log_p, color="#555555", linewidth=1.0, alpha=0.8)
    ax.fill_between(
        freqs, threshold, neg_log_p,
        where=sig_mask,
        color="#d7191c",
        alpha=0.40,
        label=f"Significant (p < {alpha})",
    )
    ax.axhline(
        y=threshold,
        color="#d7191c",
        linestyle="--",
        linewidth=1.2,
        label=f"−log₁₀({alpha}) = {threshold:.2f}",
    )
    ax.set_xlabel("Frequency bin (radial)")
    ax.set_ylabel("−log₁₀(p-value)")
    ax.set_title("Per-Frequency Statistical Significance")
    ax.legend(loc="upper right", framealpha=0.8)

    fig.savefig(output_path)
    plt.close(fig)
