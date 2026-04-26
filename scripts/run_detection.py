"""Detection experiment pipeline.

Trains and evaluates logistic regression, SVM, and random forest classifiers
on spectral features extracted from real vs. generated image spectra.
Generates ROC curves, CV result charts, and feature importance plots.

Usage
-----
From the repository root::

    python scripts/run_detection.py
    python scripts/run_detection.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_curve, auc

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.spectral.io import compute_spectra
from src.spectral.statistics import population_stats
from src.detection.features import build_feature_matrix
from src.detection.classifier import (
    make_classifiers,
    evaluate_all_classifiers,
    fit_final_classifier,
)
from src.detection.importance import (
    get_feature_names,
    random_forest_importance,
    logistic_regression_coefficients,
    top_k_features,
)
from src.visualization.detection import (
    plot_roc_curves,
    plot_cv_results,
    plot_feature_importance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detection experiment (real vs. generated classifiers)."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Run only this model key. Omit to run all active entries.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    real_dir = _REPO_ROOT / cfg["data"]["real_dir"]
    all_generated: dict[str, Path] = {
        name: _REPO_ROOT / p
        for name, p in cfg["data"]["generated_dirs"].items()
    }
    if args.model:
        all_generated = {args.model: all_generated[args.model]}

    n_bins: int = cfg["detection"]["feature_bins"]
    cv_folds: int = cfg["detection"]["cv_folds"]
    random_seed: int = cfg["detection"]["random_seed"]

    # ---- Load real spectra -------------------------------------------------
    print("\n[1/3] Computing real spectra ...")
    real_spectra = compute_spectra(real_dir)
    print(f"  Real: {real_spectra.shape[0]} images, {real_spectra.shape[1]} bins")

    # ---- Load generated spectra --------------------------------------------
    print("\n[2/3] Computing generated spectra ...")
    gen_data: dict[str, np.ndarray] = {}
    for group_name, gen_dir in all_generated.items():
        try:
            spectra = compute_spectra(gen_dir)
            gen_data[group_name] = spectra
            print(f"  {group_name}: {spectra.shape[0]} images")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  WARNING: skipping {group_name} — {exc}")

    if not gen_data:
        print("  No generated spectra available.  Exiting.")
        return

    # ---- Classification ----------------------------------------------------
    print("\n[3/3] Running classification experiments ...")
    all_detection_results: dict[str, dict] = {}
    feature_names = get_feature_names(n_bins=n_bins)

    for group_name, gen_spectra in gen_data.items():
        print(f"\n  --- Group: {group_name} ---")

        figures_dir = _REPO_ROOT / "results" / group_name / "figures"
        tables_dir  = _REPO_ROOT / "results" / group_name
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Build feature matrix
        X, y = build_feature_matrix(real_spectra, gen_spectra, n_bins=n_bins)
        print(f"  Feature matrix: {X.shape}, labels: {np.bincount(y)}")

        # Cross-validation
        classifiers = make_classifiers(random_seed=random_seed)
        cv_results = evaluate_all_classifiers(
            classifiers, X, y, cv_folds=cv_folds, random_seed=random_seed
        )

        # Print CV summary
        for clf_name, result in cv_results.items():
            print(
                f"    {clf_name}: "
                f"acc={result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}  "
                f"AUC={result['roc_auc_mean']:.3f}±{result['roc_auc_std']:.3f}  "
                f"F1={result['f1_mean']:.3f}±{result['f1_std']:.3f}"
            )

        # ROC curves from out-of-fold predictions (unbiased, matches CV AUC).
        roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for clf_name, cv_res in cv_results.items():
            fpr, tpr, _ = roc_curve(cv_res["oof_y_true"], cv_res["oof_y_proba"])
            roc_data[clf_name] = (fpr, tpr, float(auc(fpr, tpr)))

        # Fit final classifiers on full dataset for feature importance only.
        classifiers_final = make_classifiers(random_seed=random_seed)
        for clf_name, clf in classifiers_final.items():
            fit_final_classifier(clf, X, y)

        # Figures
        plot_roc_curves(roc_data, figures_dir / "roc_curves.png")
        plot_cv_results(cv_results, figures_dir / "cv_results.png")
        print(f"  Figures → {figures_dir}")

        # Random forest feature importance
        rf_clf = classifiers_final["random_forest"]
        rf_importance_df = random_forest_importance(rf_clf, feature_names)
        rf_top = top_k_features(rf_importance_df, k=20)
        plot_feature_importance(rf_top, figures_dir / "feature_importance_rf.png",
                                top_k=20, title=f"RF Feature Importance — {group_name}")

        lr_clf = classifiers_final["logistic_regression"]
        lr_coef_df = logistic_regression_coefficients(lr_clf, feature_names)
        lr_top = top_k_features(lr_coef_df, k=20)
        plot_feature_importance(lr_top, figures_dir / "feature_importance_lr.png",
                                top_k=20, title=f"LR Coefficients — {group_name}")

        # Save per-model detection results
        detection_result = {
            "cv_results": cv_results,
            "roc_auc_oof": {
                clf_name: float(roc_auc)
                for clf_name, (_, _, roc_auc) in roc_data.items()
            },
            "rf_top_features": rf_top["feature"].tolist(),
            "lr_top_features": lr_top["feature"].tolist(),
        }
        all_detection_results[group_name] = detection_result
        det_path = tables_dir / "detection_results.json"
        with open(det_path, "w") as f:
            json.dump(detection_result, f, indent=2)
        print(f"  Detection results → {det_path}")

    # ---- Print summary table -----------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Group':<30} {'Classifier':<25} {'AUC (CV)':<12}")
    print("=" * 70)
    for group_name, result in all_detection_results.items():
        for clf_name, cv_res in result["cv_results"].items():
            print(
                f"{group_name:<30} {clf_name:<25} "
                f"{cv_res['roc_auc_mean']:.3f}±{cv_res['roc_auc_std']:.3f}"
            )
    print("=" * 70)
    print("\nDetection experiment complete.")


if __name__ == "__main__":
    main()
