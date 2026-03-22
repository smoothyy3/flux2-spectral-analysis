"""Visualisation of classifier performance and feature importance.

Provides ROC curve plots, cross-validation result bar charts, and feature
importance horizontal bar charts for the detection experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import apply_style, get_color

apply_style()

# Colour mapping for classifiers
_CLF_COLORS = {
    "logistic_regression": "#1b7837",
    "svm_rbf": "#762a83",
    "random_forest": "#e08214",
}

# Colour mapping for feature types
_FEAT_COLORS = {
    "spectrum_bin": "#2c7bb6",
    "band_energy": "#fdae61",
    "spectral": "#1a9641",
}


def _get_clf_color(name: str) -> str:
    for key, color in _CLF_COLORS.items():
        if key in name:
            return color
    return "#888888"


def _get_feat_color(feature_name: str) -> str:
    for key, color in _FEAT_COLORS.items():
        if feature_name.startswith(key):
            return color
    return "#888888"


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """Plot ROC curves for multiple classifiers on the same axes.

    Parameters
    ----------
    roc_data : dict
        Mapping of classifier name → (fpr, tpr, auc).  *fpr* and *tpr* are
        1-D arrays of false positive and true positive rates; *auc* is the
        scalar area under the ROC curve.
    output_path : Path
        Destination file path for the saved figure.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    for clf_name, (fpr, tpr, auc) in roc_data.items():
        color = _get_clf_color(clf_name)
        ax.plot(fpr, tpr, color=color, linewidth=2.0,
                label=f"{clf_name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999",
            linewidth=1.0, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Real vs. Generated")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    fig.savefig(output_path)
    plt.close(fig)


def plot_cv_results(
    cv_results: dict[str, dict],
    output_path: Path,
) -> None:
    """Grouped bar chart of cross-validation metrics across classifiers.

    Parameters
    ----------
    cv_results : dict
        Nested dict mapping classifier name → result dict (as returned by
        ``evaluate_classifier``).  Each result dict must contain keys
        ``accuracy_mean``, ``accuracy_std``, ``roc_auc_mean``,
        ``roc_auc_std``, ``f1_mean``, ``f1_std``.
    output_path : Path
        Destination file path for the saved figure.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "roc_auc", "f1"]
    metric_labels = ["Accuracy", "ROC-AUC", "F1 (macro)"]
    clf_names = list(cv_results.keys())
    n_metrics = len(metrics)
    n_clfs = len(clf_names)

    bar_width = 0.7 / n_clfs
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, clf_name in enumerate(clf_names):
        res = cv_results[clf_name]
        means = [res[f"{m}_mean"] for m in metrics]
        stds = [res[f"{m}_std"] for m in metrics]
        offsets = x + (i - n_clfs / 2 + 0.5) * bar_width
        ax.bar(
            offsets,
            means,
            width=bar_width,
            yerr=stds,
            label=clf_name,
            color=_get_clf_color(clf_name),
            alpha=0.85,
            capsize=4,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Validation Results")
    ax.legend(loc="lower right", framealpha=0.8)

    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_importance(
    importance_df,
    output_path: Path,
    top_k: int = 20,
    title: str = "Top Feature Importances",
) -> None:
    """Horizontal bar chart of the top-k most important features.

    Bars are coloured by feature type:

    * Blue — ``spectrum_bin_*`` features
    * Orange — ``band_energy_*`` features
    * Green — ``spectral_*`` features (slope, intercept)

    Parameters
    ----------
    importance_df : pandas.DataFrame
        DataFrame with columns ``["feature", "importance"]`` or
        ``["feature", "coefficient"]``, pre-sorted in descending order.
    output_path : Path
        Destination file path for the saved figure.
    top_k : int, optional
        Number of features to display.  Defaults to 20.
    title : str, optional
        Figure title.  Defaults to ``"Top Feature Importances"``.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = importance_df.head(top_k).copy()

    # Detect value column (importance or coefficient)
    value_col = "importance" if "importance" in df.columns else "coefficient"
    values = df[value_col].values
    features = df["feature"].values
    colors = [_get_feat_color(f) for f in features]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.35 + 1)))

    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="white",
            linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    ax.invert_yaxis()

    # Legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=_FEAT_COLORS["spectrum_bin"],
                       label="Spectrum bin"),
        plt.Rectangle((0, 0), 1, 1, color=_FEAT_COLORS["band_energy"],
                       label="Band energy"),
        plt.Rectangle((0, 0), 1, 1, color=_FEAT_COLORS["spectral"],
                       label="Spectral slope/intercept"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
