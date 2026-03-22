"""Orchestration script — runs the full analysis pipeline end-to-end.

Executes the three sub-pipelines in sequence:

1. ``run_analysis.py``  — spectral metrics and comparison figures
2. ``run_controls.py``  — degradation controls and 2-D analysis
3. ``run_detection.py`` — classifier-based detection experiment

Timing is reported for each stage.  At the end, a ``results/summary.json``
file is assembled from the individual JSON outputs.

Usage
-----
From the repository root::

    python scripts/run_all.py
    python scripts/run_all.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run_analysis
import run_controls
import run_detection


def _load_json_safe(path: Path) -> dict:
    """Load a JSON file, returning an empty dict if the file does not exist."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete FLUX spectral analysis pipeline."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to the experiment YAML config file.",
    )
    args = parser.parse_args()

    # Inject --config into sys.argv for child main() calls
    # (they parse sys.argv via argparse)
    original_argv = sys.argv.copy()

    timings: dict[str, float] = {}

    # ---- Stage 1: Analysis -------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: Spectral Analysis")
    print("=" * 70)
    sys.argv = ["run_analysis.py", "--config", args.config]
    t0 = time.perf_counter()
    try:
        run_analysis.main()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"  ERROR in run_analysis: {exc}")
    timings["run_analysis"] = time.perf_counter() - t0
    print(f"\n  Stage 1 completed in {timings['run_analysis']:.1f}s")

    # ---- Stage 2: Controls -------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2: Control Experiments")
    print("=" * 70)
    sys.argv = ["run_controls.py", "--config", args.config]
    t0 = time.perf_counter()
    try:
        run_controls.main()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"  ERROR in run_controls: {exc}")
    timings["run_controls"] = time.perf_counter() - t0
    print(f"\n  Stage 2 completed in {timings['run_controls']:.1f}s")

    # ---- Stage 3: Detection ------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 3: Detection Experiment")
    print("=" * 70)
    sys.argv = ["run_detection.py", "--config", args.config]
    t0 = time.perf_counter()
    try:
        run_detection.main()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"  ERROR in run_detection: {exc}")
    timings["run_detection"] = time.perf_counter() - t0
    print(f"\n  Stage 3 completed in {timings['run_detection']:.1f}s")

    sys.argv = original_argv

    # ---- Assemble summary.json ---------------------------------------------
    print("\n" + "=" * 70)
    print("Assembling results/summary.json ...")
    print("=" * 70)

    import yaml
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tables_dir = Path(cfg["output"]["tables_dir"])
    results_dir = _REPO_ROOT / "results"

    spectral_metrics = _load_json_safe(tables_dir / "spectral_metrics.json")
    control_metrics = _load_json_safe(tables_dir / "control_metrics.json")
    detection_results = _load_json_safe(tables_dir / "detection_results.json")

    # Extract key findings
    key_findings: dict = {}

    for group_name, metrics in spectral_metrics.items():
        key_findings[group_name] = {
            "kl_divergence": metrics.get("kl_divergence"),
            "wasserstein_distance": metrics.get("wasserstein_distance"),
            "real_slope": metrics.get("real_slope"),
            "gen_slope": metrics.get("gen_slope"),
            "slope_difference": (
                (metrics.get("gen_slope") or 0) - (metrics.get("real_slope") or 0)
            ),
        }

    for group_name, det_res in detection_results.items():
        if group_name in key_findings:
            best_auc = max(
                det_res.get("roc_auc_full", {}).values(),
                default=None,
            )
            key_findings[group_name]["best_detection_auc"] = best_auc
            key_findings[group_name]["best_cv_auc"] = max(
                (v["roc_auc_mean"] for v in det_res.get("cv_results", {}).values()),
                default=None,
            )

    summary = {
        "timings_seconds": timings,
        "total_time_seconds": sum(timings.values()),
        "key_findings": key_findings,
        "spectral_metrics": spectral_metrics,
        "control_metrics": control_metrics,
        "detection_results": {
            group: {
                "cv_results": res.get("cv_results", {}),
                "roc_auc_full": res.get("roc_auc_full", {}),
            }
            for group, res in detection_results.items()
        },
    }

    summary_path = results_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")

    # ---- Print timing summary ----------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    for stage, elapsed in timings.items():
        print(f"  {stage:<20} {elapsed:>7.1f}s")
    total = sum(timings.values())
    print(f"  {'TOTAL':<20} {total:>7.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
