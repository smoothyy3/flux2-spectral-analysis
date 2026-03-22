"""Main spectral analysis pipeline.

Loads images, computes (or loads cached) radial power spectra, computes
population statistics and comparison metrics, generates all core figures,
and saves a JSON metrics table.

Usage
-----
From the repository root::

    python scripts/run_analysis.py
    python scripts/run_analysis.py --config configs/experiment.yaml
    python scripts/run_analysis.py --force-recompute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Make project root importable when running as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.spectral.preprocessing import load_images_from_dir
from src.spectral.fft import batch_power_spectra
from src.spectral.azimuthal import batch_azimuthal_average
from src.spectral.statistics import (
    population_stats,
    per_frequency_ttest,
)
from src.spectral.metrics import compute_all_metrics
from src.visualization.spectra import (
    plot_mean_spectra,
    plot_band_energy_comparison,
    plot_spectral_slopes,
)
from src.visualization.difference import (
    plot_spectral_difference,
    plot_significance_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_or_compute_spectra(
    directory: Path,
    cache_path: Path,
    force: bool,
) -> np.ndarray:
    """Return cached (N, R) radial spectra or compute and cache them.

    Parameters
    ----------
    directory : Path
        Directory containing raw images.
    cache_path : Path
        Path to the .npz cache file.
    force : bool
        If ``True``, ignore existing cache and recompute.

    Returns
    -------
    np.ndarray
        Array of shape (N, R) with individual radial power spectra.
    """
    if cache_path.exists() and not force:
        print(f"  Loading cached spectra from {cache_path.name} ...")
        data = np.load(cache_path)
        return data["spectra"]

    print(f"  Computing spectra for {directory} ...")
    images = load_images_from_dir(directory)
    power_spectra = batch_power_spectra(images)
    radial_spectra = batch_azimuthal_average(power_spectra)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, spectra=radial_spectra)
    print(f"  Cached → {cache_path}")
    return radial_spectra


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run spectral analysis pipeline for FLUX image generation."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached spectra and recompute from raw images.",
    )
    args = parser.parse_args()

    # ---- Load config -------------------------------------------------------
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    real_dir = Path(cfg["data"]["real_dir"])
    generated_dirs: dict[str, Path] = {
        name: Path(p) for name, p in cfg["data"]["generated_dirs"].items()
    }
    features_dir = Path(cfg["output"]["features_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    tables_dir = Path(cfg["output"]["tables_dir"])
    n_bands: int = cfg["spectral"]["frequency_bands"]

    for d in [figures_dir, tables_dir, features_dir]:
        d.mkdir(parents=True, exist_ok=True)

    force: bool = args.force_recompute

    # ---- Real images -------------------------------------------------------
    print("\n[1/3] Loading / computing real spectra ...")
    real_cache = features_dir / "real_spectra.npz"
    real_spectra = _load_or_compute_spectra(real_dir, real_cache, force)
    real_mean, real_std = population_stats(real_spectra)
    print(f"  Real: {real_spectra.shape[0]} images, {real_spectra.shape[1]} bins")

    # ---- Generated images --------------------------------------------------
    print("\n[2/3] Loading / computing generated spectra ...")
    gen_data: dict[str, np.ndarray] = {}
    for group_name, gen_dir in generated_dirs.items():
        print(f"  Group: {group_name}")
        cache_path = features_dir / f"{group_name}_spectra.npz"
        try:
            spectra = _load_or_compute_spectra(gen_dir, cache_path, force)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  WARNING: skipping {group_name} — {exc}")
            continue
        gen_data[group_name] = spectra
        print(f"  {group_name}: {spectra.shape[0]} images")

    if not gen_data:
        print("No generated groups available.  Exiting.")
        return

    # ---- Statistics and metrics --------------------------------------------
    print("\n[3/3] Computing statistics and metrics ...")
    all_metrics: dict[str, dict] = {}

    group_spectra_for_plot: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "real": (real_mean, real_std),
    }
    group_band_energies: dict[str, np.ndarray] = {}
    group_slopes: dict[str, float] = {}

    from src.spectral.metrics import band_energy_ratios, spectral_slope
    group_band_energies["real"] = band_energy_ratios(real_mean, n_bands=n_bands)
    real_slope_val, _ = spectral_slope(real_mean)
    group_slopes["real"] = real_slope_val

    for group_name, gen_spectra in gen_data.items():
        gen_mean, gen_std = population_stats(gen_spectra)
        t_stats, p_values = per_frequency_ttest(real_spectra, gen_spectra)

        metrics = compute_all_metrics(real_mean, gen_mean, n_bands=n_bands)
        all_metrics[group_name] = metrics

        group_spectra_for_plot[group_name] = (gen_mean, gen_std)
        group_band_energies[group_name] = np.array(metrics["gen_band_energies"])
        group_slopes[group_name] = metrics["gen_slope"]

        # Per-group difference figure
        diff_path = figures_dir / f"spectral_difference_{group_name}.png"
        plot_spectral_difference(
            real_mean, gen_mean, p_values, diff_path, group_name=group_name
        )
        print(f"  Saved: {diff_path.name}")

        sig_path = figures_dir / f"significance_mask_{group_name}.png"
        plot_significance_mask(p_values, sig_path)
        print(f"  Saved: {sig_path.name}")

        # Console summary
        print(f"\n  === {group_name} ===")
        print(f"    KL divergence:        {metrics['kl_divergence']:.4f}")
        print(f"    Wasserstein distance: {metrics['wasserstein_distance']:.4f}")
        print(f"    L2 log distance:      {metrics['l2_log_distance']:.4f}")
        print(f"    Spectral slope (gen): {metrics['gen_slope']:.3f}  "
              f"(real: {metrics['real_slope']:.3f})")

    # ---- Figures -----------------------------------------------------------
    mean_spectra_path = figures_dir / "mean_spectra_comparison.png"
    plot_mean_spectra(group_spectra_for_plot, mean_spectra_path)
    print(f"\nSaved: {mean_spectra_path.name}")

    band_path = figures_dir / "band_energy_comparison.png"
    plot_band_energy_comparison(group_band_energies, band_path)
    print(f"Saved: {band_path.name}")

    slope_path = figures_dir / "spectral_slope_comparison.png"
    plot_spectral_slopes(group_slopes, slope_path)
    print(f"Saved: {slope_path.name}")

    # ---- Save metrics table ------------------------------------------------
    metrics_path = tables_dir / "spectral_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
