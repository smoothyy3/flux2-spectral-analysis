"""Visualisation of 1-D radial power spectra and derived statistics.

All plotting functions save their output directly to disk and close the
figure to avoid accumulating memory during batch runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import apply_style, get_color

apply_style()


def plot_mean_spectra(
    group_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    log_scale: bool = True,
    title: str = "Mean Radial Power Spectra",
) -> None:
    """Plot mean radial power spectra with ±1 std shaded bands.

    Parameters
    ----------
    group_spectra : dict
        Mapping of group name → (mean_spectrum, std_spectrum).  Both arrays
        are 1-D and have the same length R (number of frequency bins).
    output_path : Path
        Destination file path for the saved figure (PNG recommended).
    log_scale : bool, optional
        If ``True``, the y-axis displays log10(power).  Defaults to ``True``.
    title : str, optional
        Figure title.  Defaults to ``"Mean Radial Power Spectra"``.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for group_name, (mean_spec, std_spec) in group_spectra.items():
        color = get_color(group_name)
        freqs = np.arange(len(mean_spec))

        # Spectra are already in log10-power space; plot directly with
        # symmetric ±1 std bands.
        y_mean = mean_spec
        y_upper = mean_spec + std_spec
        y_lower = mean_spec - std_spec

        ax.plot(freqs, y_mean, label=group_name, color=color, linewidth=1.5)
        ax.fill_between(freqs, y_lower, y_upper, alpha=0.2, color=color)

    ax.set_xlabel("Frequency bin (radial)")
    ax.set_ylabel("log₁₀(Power)" if log_scale else "Power")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.8)

    fig.savefig(output_path)
    plt.close(fig)


def plot_band_energy_comparison(
    group_band_energies: dict[str, np.ndarray],
    output_path: Path,
    band_labels: list[str] | None = None,
) -> None:
    """Grouped bar chart comparing band energy fractions across groups.

    Parameters
    ----------
    group_band_energies : dict
        Mapping of group name → 1-D array of energy fractions (length =
        number of bands).
    output_path : Path
        Destination file path for the saved figure.
    band_labels : list of str, optional
        Names for each frequency band.  Defaults to
        ``["Low", "Mid", "High"]``.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if band_labels is None:
        band_labels = ["Low", "Mid", "High"]

    n_bands = len(band_labels)
    n_groups = len(group_band_energies)
    group_names = list(group_band_energies.keys())

    bar_width = 0.8 / n_groups
    x = np.arange(n_bands)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, group_name in enumerate(group_names):
        energies = group_band_energies[group_name]
        offsets = x + (i - n_groups / 2 + 0.5) * bar_width
        bars = ax.bar(
            offsets,
            energies,
            width=bar_width,
            label=group_name,
            color=get_color(group_name),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, energies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylabel("Mean log₁₀(power)")
    ax.set_title("Band Energy Comparison (mean log power per band)")
    ax.legend(loc="upper right", framealpha=0.8)

    fig.savefig(output_path)
    plt.close(fig)


def plot_spectral_slopes(
    group_slopes: dict[str, tuple[float, float] | float],
    output_path: Path,
) -> None:
    """Horizontal bar chart of spectral slopes with a natural-image reference.

    Parameters
    ----------
    group_slopes : dict
        Mapping of group name → slope (float) or (slope_mean, slope_std).
        When a tuple is provided the second element is used as the error bar.
    output_path : Path
        Destination file path for the saved figure.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    slopes: list[float] = []
    errors: list[float] = []

    for name, value in group_slopes.items():
        names.append(name)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            slopes.append(float(value[0]))
            errors.append(float(value[1]))
        else:
            slopes.append(float(value))
            errors.append(0.0)

    y_pos = np.arange(len(names))
    colors = [get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.6 + 1)))

    xerr = errors if any(e > 0 for e in errors) else None
    ax.barh(
        y_pos,
        slopes,
        xerr=xerr,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        capsize=4,
    )
    ax.axvline(x=-2.0, linestyle="--", color="#333333", linewidth=1.2,
               label="Natural image reference (−2)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Spectral slope")
    ax.set_title("Spectral Slope Comparison")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.invert_yaxis()

    fig.savefig(output_path)
    plt.close(fig)
