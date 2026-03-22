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

def _load_spectra(cache_path: Path) -> np.ndarray | None:
    if not cache_path.exists():
        print(f"  WARNING: cache not found: {cache_path}")
        return None
    data = np.load(cache_path)
    return data["spectra"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detection experiment (real vs. generated classifiers)."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to the experiment YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    generated_dirs: dict[str, str] = cfg["data"]["generated_dirs"]
    features_dir = Path(cfg["output"]["features_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    tables_dir = Path(cfg["output"]["tables_dir"])
    n_bins: int = cfg["detection"]["feature_bins"]
    cv_folds: int = cfg["detection"]["cv_folds"]
    random_seed: int = cfg["detection"]["random_seed"]

    for d in [figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load real spectra -------------------------------------------------
    print("\n[1/3] Loading cached real spectra ...")
    real_cache = features_dir / "real_spectra.npz"
    real_spectra = _load_spectra(real_cache)
    if real_spectra is None:
        print("  ERROR: Run run_analysis.py first to compute and cache spectra.")
        return
    print(f"  Real: {real_spectra.shape[0]} images, {real_spectra.shape[1]} bins")

    # ---- Load generated spectra --------------------------------------------
    print("\n[2/3] Loading cached generated spectra ...")
    gen_data: dict[str, np.ndarray] = {}
    for group_name in generated_dirs.keys():
        cache_path = features_dir / f"{group_name}_spectra.npz"
        spectra = _load_spectra(cache_path)
        if spectra is not None:
            gen_data[group_name] = spectra
            print(f"  {group_name}: {spectra.shape[0]} images")

    if not gen_data:
        print("  No generated spectra available.  Exiting.")
        return

    # ---- Classification ----------------------------------------------------
    print("\n[3/3] Running classification experiments ...")
    all_detection_results: dict[str, dict] = {}
    feature_names = get_feature_names(n_bins=n_bins)

    for group_name, gen_spectra in gen_data.items():
        print(f"\n  --- Group: {group_name} ---")

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

        # Fit final classifiers for feature importance and ROC curves
        roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        classifiers_final = make_classifiers(random_seed=random_seed)

        for clf_name, clf in classifiers_final.items():
            fitted_clf = fit_final_classifier(clf, X, y)
            y_proba = fitted_clf.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            roc_data[clf_name] = (fpr, tpr, roc_auc)

        # Figures
        roc_path = figures_dir / f"roc_curves_{group_name}.png"
        plot_roc_curves(roc_data, roc_path)
        print(f"  Saved: {roc_path.name}")

        cv_path = figures_dir / f"cv_results_{group_name}.png"
        plot_cv_results(cv_results, cv_path)
        print(f"  Saved: {cv_path.name}")

        # Random forest feature importance
        rf_clf = classifiers_final["random_forest"]
        rf_importance_df = random_forest_importance(rf_clf, feature_names)
        rf_top = top_k_features(rf_importance_df, k=20)

        rf_imp_path = figures_dir / f"feature_importance_rf_{group_name}.png"
        plot_feature_importance(
            rf_top, rf_imp_path,
            top_k=20,
            title=f"RF Feature Importance — {group_name}",
        )
        print(f"  Saved: {rf_imp_path.name}")

        # Logistic regression coefficients
        lr_clf = classifiers_final["logistic_regression"]
        lr_coef_df = logistic_regression_coefficients(lr_clf, feature_names)
        lr_top = top_k_features(lr_coef_df, k=20)

        lr_imp_path = figures_dir / f"feature_importance_lr_{group_name}.png"
        plot_feature_importance(
            lr_top, lr_imp_path,
            top_k=20,
            title=f"LR Coefficients — {group_name}",
        )
        print(f"  Saved: {lr_imp_path.name}")

        # Store results
        all_detection_results[group_name] = {
            "cv_results": cv_results,
            "roc_auc_full": {
                clf_name: float(roc_auc)
                for clf_name, (_, _, roc_auc) in roc_data.items()
            },
            "rf_top_features": rf_top["feature"].tolist(),
            "lr_top_features": lr_top["feature"].tolist(),
        }

    # ---- Save detection results --------------------------------------------
    results_path = tables_dir / "detection_results.json"
    with open(results_path, "w") as f:
        json.dump(all_detection_results, f, indent=2)
    print(f"\nDetection results saved to {results_path}")

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
