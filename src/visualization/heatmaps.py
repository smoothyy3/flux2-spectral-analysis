"""2-D power spectrum heatmap visualisations.

Provides functions to render log-scale 2-D power spectra as false-colour
heatmaps and to compare 2-D spectra across multiple groups side by side.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import apply_style, get_color
from src.controls.directional import (
    log_power_spectrum_2d,
    compute_horizontal_profile,
    compute_vertical_profile,
)

apply_style()

_LOG_FLOOR = 1.0  # floor before log10; see spectra.py for rationale


def plot_2d_spectrum_heatmap(
    avg_spectrum_2d: np.ndarray,
    title: str,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render the log10 of a 2-D average power spectrum as a heatmap.

    The zero-frequency component is assumed to be at the array centre
    (consistent with ``np.fft.fftshift``).

    Parameters
    ----------
    avg_spectrum_2d : np.ndarray
        2-D average power spectrum of shape (H, W).
    title : str
        Subplot title.
    output_path : Path
        Destination file path for the saved figure.
    vmin : float, optional
        Minimum value for the colour scale.  Inferred from data if ``None``.
    vmax : float, optional
        Maximum value for the colour scale.  Inferred from data if ``None``.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_spec = log_power_spectrum_2d(avg_spectrum_2d)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        log_spec,
        cmap="inferno",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="log₁₀(Power)")
    ax.set_title(title)
    ax.set_xlabel("Horizontal frequency")
    ax.set_ylabel("Vertical frequency")

    fig.savefig(output_path)
    plt.close(fig)


def plot_2d_spectra_comparison(
    spectra: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Side-by-side 2-D spectrum heatmaps for multiple groups.

    All subplots share the same colour scale (vmin/vmax) computed from the
    combined range of all spectra.

    Parameters
    ----------
    spectra : dict
        Mapping of group name → 2-D average power spectrum array (H, W).
    output_path : Path
        Destination file path for the saved figure.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(spectra)
    group_names = list(spectra.keys())

    # Compute global colour range
    log_specs = [log_power_spectrum_2d(s) for s in spectra.values()]
    global_vmin = min(ls.min() for ls in log_specs)
    global_vmax = max(ls.max() for ls in log_specs)

    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5), squeeze=False)

    for ax, group_name, log_spec in zip(axes[0], group_names, log_specs):
        im = ax.imshow(
            log_spec,
            cmap="inferno",
            origin="lower",
            vmin=global_vmin,
            vmax=global_vmax,
            aspect="equal",
        )
        ax.set_title(group_name)
        ax.set_xlabel("H. frequency")
        ax.set_ylabel("V. frequency")

    # Attach colorbar to the rightmost axis only
    cbar = fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04)
    cbar.set_label("log₁₀(Power)")

    fig.suptitle("2-D Power Spectrum Comparison", fontsize=13)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_directional_profiles(
    real_spectrum: np.ndarray,
    gen_spectrum: np.ndarray,
    output_path: Path,
) -> None:
    """2×2 grid of horizontal and vertical directional profiles.

    Plots four panels:
    * Top-left: horizontal profile (real)
    * Top-right: horizontal profile (generated)
    * Bottom-left: vertical profile (real)
    * Bottom-right: vertical profile (generated)

    Parameters
    ----------
    real_spectrum : np.ndarray
        2-D average power spectrum for the real image set, shape (H, W).
    gen_spectrum : np.ndarray
        2-D average power spectrum for the generated image set, shape (H, W).
    output_path : Path
        Destination file path for the saved figure.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_h = np.log10(np.maximum(compute_horizontal_profile(real_spectrum), _LOG_FLOOR))
    gen_h = np.log10(np.maximum(compute_horizontal_profile(gen_spectrum), _LOG_FLOOR))
    real_v = np.log10(np.maximum(compute_vertical_profile(real_spectrum), _LOG_FLOOR))
    gen_v = np.log10(np.maximum(compute_vertical_profile(gen_spectrum), _LOG_FLOOR))

    real_color = get_color("real")
    gen_color = get_color("klein_4b_distilled")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Directional Frequency Profiles", fontsize=13)

    panels = [
        (axes[0, 0], real_h, "Horizontal — Real", real_color),
        (axes[0, 1], gen_h, "Horizontal — Generated", gen_color),
        (axes[1, 0], real_v, "Vertical — Real", real_color),
        (axes[1, 1], gen_v, "Vertical — Generated", gen_color),
    ]

    for ax, profile, panel_title, color in panels:
        ax.plot(profile, color=color, linewidth=1.2)
        ax.set_title(panel_title)
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("log₁₀(Power)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
