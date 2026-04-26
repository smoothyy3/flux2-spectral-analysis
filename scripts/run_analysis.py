"""Main spectral analysis pipeline.

Loads images, computes radial power spectra, computes population statistics
and comparison metrics, generates all core figures, and saves a metrics.json
per model under results/<model_name>/.

Usage
-----
    python scripts/run_analysis.py --model klein_distilled
    python scripts/run_analysis.py --model klein_base_g4_s50
    python scripts/run_analysis.py --model flux2_max
    python scripts/run_analysis.py  # runs all active entries in experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.spectral.io import compute_spectra
from src.spectral.statistics import population_stats, per_frequency_ttest
from src.spectral.metrics import (
    compute_all_metrics,
    band_energy_ratios,
    spectral_slope,
)
from src.visualization.spectra import (
    plot_mean_spectra,
    plot_band_energy_comparison,
    plot_spectral_slopes,
)
from src.visualization.difference import (
    plot_spectral_difference,
    plot_significance_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Run only this model key (must match a key in generated_dirs). "
             "Omit to run all active entries.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    real_dir = _REPO_ROOT / cfg["data"]["real_dir"]
    n_bands: int = cfg["spectral"]["frequency_bands"]

    all_generated: dict[str, Path] = {
        name: _REPO_ROOT / p
        for name, p in cfg["data"]["generated_dirs"].items()
    }

    if args.model:
        if args.model not in all_generated:
            print(f"ERROR: '{args.model}' not found in generated_dirs. "
                  f"Available: {list(all_generated.keys())}")
            sys.exit(1)
        active = {args.model: all_generated[args.model]}
    else:
        active = all_generated

    # ── Real spectra ────────────────────────────────────────────────────────
    print(f"\nLoading real images from {real_dir} ...")
    real_spectra = compute_spectra(real_dir)
    real_mean, real_std = population_stats(real_spectra)
    print(f"  Real: {real_spectra.shape[0]} images, {real_spectra.shape[1]} bins")

    # ── Per-model analysis ──────────────────────────────────────────────────
    for model_name, gen_dir in active.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        figures_dir = _REPO_ROOT / "results" / model_name / "figures"
        metrics_path = _REPO_ROOT / "results" / model_name / "metrics.json"
        figures_dir.mkdir(parents=True, exist_ok=True)

        try:
            gen_spectra = compute_spectra(gen_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  WARNING: skipping {model_name} — {exc}")
            continue

        gen_mean, gen_std = population_stats(gen_spectra)
        print(f"  Generated: {gen_spectra.shape[0]} images")

        t_stats, p_values = per_frequency_ttest(real_spectra, gen_spectra)
        metrics = compute_all_metrics(real_mean, gen_mean, n_bands=n_bands)

        # Console summary
        print(f"  KL divergence:        {metrics['kl_divergence']:.4f}")
        print(f"  Wasserstein distance: {metrics['wasserstein_distance']:.4f}")
        print(f"  Spectral slope real:  {metrics['real_slope']:.3f}")
        print(f"  Spectral slope gen:   {metrics['gen_slope']:.3f}  "
              f"(Δ = {metrics['gen_slope'] - metrics['real_slope']:+.3f})")

        # Figures
        group_spectra = {
            "real": (real_mean, real_std),
            model_name: (gen_mean, gen_std),
        }
        plot_mean_spectra(group_spectra, figures_dir / "mean_spectra_comparison.png")
        plot_spectral_difference(
            real_mean, gen_mean, p_values,
            figures_dir / "spectral_difference.png",
            group_name=model_name,
        )
        plot_significance_mask(p_values, figures_dir / "significance_mask.png")

        group_bands = {
            "real": band_energy_ratios(real_mean, n_bands=n_bands),
            model_name: np.array(metrics["gen_band_energies"]),
        }
        plot_band_energy_comparison(group_bands, figures_dir / "band_energy_comparison.png")

        real_slope_val, _ = spectral_slope(real_mean, bin_start=10, bin_end=400)
        group_slopes = {"real": real_slope_val, model_name: metrics["gen_slope"]}
        plot_spectral_slopes(group_slopes, figures_dir / "spectral_slope_comparison.png")

        print(f"  Figures → {figures_dir}")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics → {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
