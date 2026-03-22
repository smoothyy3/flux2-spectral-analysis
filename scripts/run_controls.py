"""Control experiment pipeline.

Applies classical image degradation operations (blur, downscale/upscale,
JPEG compression) to real images and plots their spectra alongside the
generated images.  Also computes 2-D power spectra and directional profiles.

Usage
-----
From the repository root::

    python scripts/run_controls.py
    python scripts/run_controls.py --config configs/experiment.yaml
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

from src.spectral.preprocessing import load_images_from_dir
from src.spectral.fft import batch_power_spectra, average_power_spectrum_2d
from src.spectral.azimuthal import batch_azimuthal_average
from src.spectral.statistics import population_stats
from src.spectral.metrics import compute_all_metrics
from src.controls.degrade import apply_degradation
from src.visualization.spectra import plot_mean_spectra
from src.visualization.heatmaps import (
    plot_2d_spectrum_heatmap,
    plot_2d_spectra_comparison,
    plot_directional_profiles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_or_compute_spectra(
    directory: Path,
    cache_path: Path,
    force: bool,
) -> np.ndarray:
    if cache_path.exists() and not force:
        print(f"  Loading cached spectra from {cache_path.name} ...")
        return np.load(cache_path)["spectra"]

    print(f"  Computing spectra for {directory} ...")
    images = load_images_from_dir(directory)
    ps = batch_power_spectra(images)
    radial = batch_azimuthal_average(ps)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, spectra=radial)
    print(f"  Cached → {cache_path}")
    return radial


def _load_images(directory: Path) -> list[np.ndarray]:
    try:
        return load_images_from_dir(directory)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  WARNING: could not load images from {directory} — {exc}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run control experiments for spectral analysis."
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

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    real_dir = Path(cfg["data"]["real_dir"])
    generated_dirs: dict[str, Path] = {
        name: Path(p) for name, p in cfg["data"]["generated_dirs"].items()
    }
    degradation_methods: list[str] = cfg["controls"]["degradation_methods"]
    features_dir = Path(cfg["output"]["features_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    tables_dir = Path(cfg["output"]["tables_dir"])
    n_bands: int = cfg["spectral"]["frequency_bands"]
    force: bool = args.force_recompute

    for d in [figures_dir, tables_dir, features_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load real spectra -------------------------------------------------
    print("\n[1/4] Loading real spectra ...")
    real_cache = features_dir / "real_spectra.npz"
    real_spectra = _load_or_compute_spectra(real_dir, real_cache, force)
    real_mean, real_std = population_stats(real_spectra)
    print(f"  Real: {real_spectra.shape[0]} images")

    # Also need raw real images for degradation
    print("  Loading raw real images for degradation ...")
    real_images = _load_images(real_dir)

    # ---- Load generated spectra --------------------------------------------
    print("\n[2/4] Loading generated spectra ...")
    gen_data: dict[str, np.ndarray] = {}
    for group_name, gen_dir in generated_dirs.items():
        cache_path = features_dir / f"{group_name}_spectra.npz"
        try:
            spectra = _load_or_compute_spectra(gen_dir, cache_path, force)
            gen_data[group_name] = spectra
            print(f"  {group_name}: {spectra.shape[0]} images")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  WARNING: skipping {group_name} — {exc}")

    # ---- Apply degradations ------------------------------------------------
    print("\n[3/4] Applying degradation controls ...")
    degraded_spectra: dict[str, np.ndarray] = {}

    if real_images:
        for method in degradation_methods:
            cache_path = features_dir / f"degraded_{method.replace('.', '_')}_spectra.npz"
            if cache_path.exists() and not force:
                print(f"  Loading cached degraded spectra: {method}")
                degraded_spectra[method] = np.load(cache_path)["spectra"]
            else:
                print(f"  Degrading images with method: {method}")
                deg_images = apply_degradation(real_images, method)
                ps = batch_power_spectra(deg_images)
                radial = batch_azimuthal_average(ps)
                np.savez_compressed(cache_path, spectra=radial)
                degraded_spectra[method] = radial
                print(f"  Cached → {cache_path}")
    else:
        print("  No real images available; skipping degradation.")

    # ---- Overlay plot: real + degraded + generated -------------------------
    print("\n[4/4] Generating figures ...")
    overlay_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "real": (real_mean, real_std),
    }
    for method, spectra in degraded_spectra.items():
        m, s = population_stats(spectra)
        overlay_groups[method] = (m, s)
    for group_name, spectra in gen_data.items():
        m, s = population_stats(spectra)
        overlay_groups[group_name] = (m, s)

    overlay_path = figures_dir / "controls_spectra_overlay.png"
    plot_mean_spectra(
        overlay_groups,
        overlay_path,
        title="Controls: Real, Degraded, and Generated Spectra",
    )
    print(f"  Saved: {overlay_path.name}")

    # ---- 2-D heatmaps ------------------------------------------------------
    if real_images:
        print("  Computing 2-D average power spectra for heatmaps ...")

        real_ps_2d = batch_power_spectra(real_images)
        real_avg_2d = average_power_spectrum_2d(real_ps_2d)

        heatmap_path = figures_dir / "2d_spectrum_real.png"
        plot_2d_spectrum_heatmap(real_avg_2d, title="Real — 2D Power Spectrum",
                                 output_path=heatmap_path)
        print(f"  Saved: {heatmap_path.name}")

        comparison_2d: dict[str, np.ndarray] = {"real": real_avg_2d}

        for group_name, gen_dir in generated_dirs.items():
            if group_name not in gen_data:
                continue
            try:
                gen_images = _load_images(gen_dir)
                if not gen_images:
                    continue
                gen_ps_2d = batch_power_spectra(gen_images)
                gen_avg_2d = average_power_spectrum_2d(gen_ps_2d)
                comparison_2d[group_name] = gen_avg_2d

                h_path = figures_dir / f"2d_spectrum_{group_name}.png"
                plot_2d_spectrum_heatmap(
                    gen_avg_2d,
                    title=f"{group_name} — 2D Power Spectrum",
                    output_path=h_path,
                )
                print(f"  Saved: {h_path.name}")

                dir_path = figures_dir / f"directional_profiles_{group_name}.png"
                plot_directional_profiles(real_avg_2d, gen_avg_2d, dir_path)
                print(f"  Saved: {dir_path.name}")

            except Exception as exc:
                print(f"  WARNING: 2D heatmap failed for {group_name}: {exc}")

        if len(comparison_2d) > 1:
            comp_path = figures_dir / "2d_spectrum_comparison.png"
            plot_2d_spectra_comparison(comparison_2d, comp_path)
            print(f"  Saved: {comp_path.name}")

    # ---- Per-category: if metadata.json exists in gen dir ------------------
    for group_name, gen_dir in generated_dirs.items():
        if group_name not in gen_data:
            continue
        metadata_path = gen_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        print(f"\n  Per-category analysis for {group_name} ...")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # metadata expected: list of {"filename": ..., "prompt": ...}
        from collections import defaultdict
        prompt_to_files: dict[str, list[Path]] = defaultdict(list)
        for entry in metadata:
            fname = gen_dir / entry["filename"]
            prompt = entry.get("prompt", "unknown")
            if fname.exists():
                prompt_to_files[prompt].append(fname)

        per_prompt_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "all_generated": population_stats(gen_data[group_name])
        }
        for prompt, file_paths in prompt_to_files.items():
            prompt_images = []
            for fp in file_paths:
                from src.spectral.preprocessing import load_image_as_gray
                prompt_images.append(load_image_as_gray(fp))
            if not prompt_images:
                continue
            ps = batch_power_spectra(prompt_images)
            radial = batch_azimuthal_average(ps)
            m, s = population_stats(radial)
            label = prompt[:40].replace(" ", "_")
            per_prompt_groups[label] = (m, s)

        if len(per_prompt_groups) > 1:
            cat_path = figures_dir / f"per_category_{group_name}.png"
            plot_mean_spectra(
                per_prompt_groups,
                cat_path,
                title=f"Per-Category Spectra — {group_name}",
            )
            print(f"  Saved: {cat_path.name}")

    # ---- Save control metrics ----------------------------------------------
    control_metrics: dict[str, dict] = {}
    for method, spectra in degraded_spectra.items():
        m, _ = population_stats(spectra)
        control_metrics[f"degraded_{method}"] = compute_all_metrics(
            real_mean, m, n_bands=n_bands
        )
    for group_name, spectra in gen_data.items():
        m, _ = population_stats(spectra)
        control_metrics[group_name] = compute_all_metrics(
            real_mean, m, n_bands=n_bands
        )

    metrics_path = tables_dir / "control_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(control_metrics, f, indent=2)
    print(f"\nControl metrics saved to {metrics_path}")
    print("Control experiments complete.")


if __name__ == "__main__":
    main()
