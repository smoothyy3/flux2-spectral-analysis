"""
Guidance × Steps ablation grid analysis for Klein 4B Base.

Processes all available grid conditions, computes spectral metrics for each,
and produces five comparison figures plus a JSON metrics table.

Conditions analysed (skips any whose data directory does not exist):
  A  klein_base_g4_s50  steps=50  guidance=4.0
  B  klein_base_g1_s50  steps=50  guidance=1.0
  C  klein_base_g4_s4   steps=4   guidance=4.0
  D  klein_base_g1_s4   steps=4   guidance=1.0
  E  klein_base_g1_s10  steps=10  guidance=1.0
  F  klein_base_g1_s20  steps=20  guidance=1.0
  distilled  klein_distilled  steps=4  guidance=1.0  (LADD-trained)

Usage:
    python scripts/run_grid_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.linear_model import LogisticRegression

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.spectral.metrics import spectral_slope, wasserstein_distance, kl_divergence
from src.visualization.style import apply_style

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
# Each entry: (steps, guidance, subdir_name, display_label, color, linestyle)
_GRID_CONDITIONS: dict[str, tuple[int, float, str, str, str, str]] = {
    "A":         (50, 4.0, "klein_4b_base",     "Base g=4.0 s=50",       "#d62728", "-"),
    "B":         (50, 1.0, "klein_base_g1_s50",  "Base g=1.0 s=50",       "#1f77b4", "-"),
    "C":         (4,  4.0, "klein_base_g4_s4",   "Base g=4.0 s=4",        "#ff7f0e", "-"),
    "D":         (4,  1.0, "klein_base_g1_s4",   "Base g=1.0 s=4",        "#2ca02c", "-"),
    "E":         (10, 1.0, "klein_base_g1_s10",  "Base g=1.0 s=10",       "#17becf", "-"),
    "F":         (20, 1.0, "klein_base_g1_s20",  "Base g=1.0 s=20",       "#9467bd", "-"),
    "distilled": (4,  1.0, "klein_distilled",    "Distilled g=1.0 s=4",   "#8c564b", "--"),
}

_SLOPE_BIN_START = 10
_SLOPE_BIN_END   = 400

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR     = _REPO_ROOT / "data"
_REAL_DIR     = _DATA_DIR / "real"
_GEN_DIR      = _DATA_DIR / "generated"
_FIG_DIR      = _REPO_ROOT / "results" / "grid_ablation" / "figures"
_METRICS_DIR  = _REPO_ROOT / "results" / "grid_ablation"


# ---------------------------------------------------------------------------
# Slope helper (restricted bins)
# ---------------------------------------------------------------------------

def restricted_slope(mean_spectrum: np.ndarray) -> float:
    """Return spectral slope fitted over bins 10–400 only."""
    slope, _ = spectral_slope(
        mean_spectrum,
        bin_start=_SLOPE_BIN_START,
        bin_end=_SLOPE_BIN_END,
    )
    return slope


# ---------------------------------------------------------------------------
# Detection: LR AUC with class_weight='balanced'
# ---------------------------------------------------------------------------

def lr_auc_cv(
    real_spectra: np.ndarray,
    gen_spectra: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> tuple[float, float]:
    """5-fold stratified CV AUC for Logistic Regression on log-spectrum features.

    StandardScaler is fitted inside each fold (on train split only) via
    evaluate_classifier — no test-set information leaks into the scaler.

    Returns
    -------
    tuple of float
        ``(mean_auc, std_auc)``
    """
    from src.detection.features import build_feature_matrix
    from src.detection.classifier import evaluate_classifier

    X, y = build_feature_matrix(real_spectra, gen_spectra)
    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_seed,
        class_weight="balanced",
    )
    result = evaluate_classifier(clf, X, y, cv_folds=n_folds, random_seed=random_seed)
    return result["roc_auc_mean"], result["roc_auc_std"]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all_spectra() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute real spectra and all available grid condition spectra.

    Returns
    -------
    real_spectra : np.ndarray  shape (N_real, R)
    cond_spectra : dict mapping condition_id → np.ndarray shape (N, R)
    """
    from src.spectral.io import compute_spectra

    print("[1/2] Computing real spectra ...")
    real_spectra = compute_spectra(_REAL_DIR)
    print(f"  real: {len(real_spectra)} images")

    print("[2/2] Computing grid condition spectra ...")
    cond_spectra: dict[str, np.ndarray] = {}
    for cond_id, (steps, guidance, subdir, label, color, ls) in _GRID_CONDITIONS.items():
        gen_dir = _GEN_DIR / subdir
        if not gen_dir.exists() or not any(gen_dir.iterdir()):
            print(f"  [{cond_id}] {subdir}/ not found — skipping")
            continue
        try:
            spectra = compute_spectra(gen_dir)
            cond_spectra[cond_id] = spectra
            print(f"  [{cond_id}] {label}: {len(spectra)} images")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [{cond_id}] {subdir}/ error — {exc}")

    return real_spectra, cond_spectra


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    real_spectra: np.ndarray,
    cond_spectra: dict[str, np.ndarray],
) -> tuple[float, dict]:
    """Compute per-condition spectral metrics and LR AUC.

    Returns
    -------
    real_slope : float
    results : dict  condition_id → metric dict
    """
    real_mean = real_spectra.mean(axis=0)
    real_slope = restricted_slope(real_mean)

    results: dict[str, dict] = {}
    for cond_id, spectra in cond_spectra.items():
        _, _, _, label, _, _ = _GRID_CONDITIONS[cond_id]
        mean = spectra.mean(axis=0)
        slope = restricted_slope(mean)
        w_dist = wasserstein_distance(real_mean, mean)
        kl = kl_divergence(real_mean, mean)

        print(f"  [{cond_id}] {label}: slope={slope:.3f}  Δslope={slope - real_slope:.3f}  W={w_dist:.4f}")

        auc_mean, auc_std = lr_auc_cv(real_spectra, spectra)
        print(f"         LR AUC = {auc_mean:.3f} ± {auc_std:.3f}")

        results[cond_id] = {
            "label":       label,
            "steps":       _GRID_CONDITIONS[cond_id][0],
            "guidance":    _GRID_CONDITIONS[cond_id][1],
            "n_images":    int(len(spectra)),
            "slope":       slope,
            "slope_diff":  slope - real_slope,
            "wasserstein": w_dist,
            "kl_divergence": kl,
            "lr_auc_mean": auc_mean,
            "lr_auc_std":  auc_std,
        }

    return real_slope, results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_spectra_overlay(
    real_spectra: np.ndarray,
    cond_spectra: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """1. Overlaid 1D mean spectra with ±1σ bands."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    real_mean = real_spectra.mean(axis=0)
    real_std  = real_spectra.std(axis=0)
    x = np.arange(len(real_mean))
    eps = 1e-10

    ax.plot(x, np.log10(real_mean + eps), color="black", lw=2, label="Real (FFHQ)", zorder=10)
    ax.fill_between(
        x,
        np.log10(np.maximum(real_mean - real_std, eps)),
        np.log10(real_mean + real_std + eps),
        color="black", alpha=0.10,
    )

    for cond_id, spectra in cond_spectra.items():
        _, _, _, label, color, ls = _GRID_CONDITIONS[cond_id]
        mean = spectra.mean(axis=0)
        std  = spectra.std(axis=0)
        ax.plot(x, np.log10(mean + eps), color=color, lw=1.5, ls=ls, label=label)
        ax.fill_between(
            x,
            np.log10(np.maximum(mean - std, eps)),
            np.log10(mean + std + eps),
            color=color, alpha=0.08,
        )

    ax.set_xlabel("Radial frequency bin")
    ax.set_ylabel("log₁₀ power")
    ax.set_title("Guidance × Steps ablation — mean radial spectra")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_slope_vs_guidance(
    results: dict,
    real_slope: float,
    out_path: Path,
) -> None:
    """2. Scatter: guidance scale vs |slope diff|, coloured by step count."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    all_steps = sorted({v["steps"] for v in results.values() if v["steps"] != 4 or v.get("label", "").startswith("Base")})
    step_vals = np.array([v["steps"] for v in results.values()])
    s_min, s_max = step_vals.min(), step_vals.max()
    norm = plt.Normalize(vmin=s_min, vmax=s_max)
    cmap = cm.viridis

    for cond_id, m in results.items():
        is_distilled = cond_id == "distilled"
        x = m["guidance"]
        y = abs(m["slope_diff"])
        color = cmap(norm(m["steps"]))
        marker = "*" if is_distilled else "o"
        size = 180 if is_distilled else 80
        ax.scatter(x, y, c=[color], s=size, marker=marker, zorder=5,
                   edgecolors="k", linewidths=0.5)
        ax.annotate(
            m["label"].split(" ", 1)[1] if " " in m["label"] else m["label"],
            (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7,
        )

    ax.axhline(0, color="gray", lw=1, ls="--", label="Real (Δ=0)")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Inference steps")

    ax.set_xlabel("Guidance scale")
    ax.set_ylabel("|Slope − Real slope|")
    ax.set_title("Slope deviation vs guidance scale")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_slope_vs_steps(
    results: dict,
    real_slope: float,
    out_path: Path,
) -> None:
    """3. Lines: step count vs slope diff, separated by guidance level."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    guidance_groups: dict[float, list[tuple[int, float, str]]] = {}
    for cond_id, m in results.items():
        g = m["guidance"]
        guidance_groups.setdefault(g, []).append((m["steps"], m["slope_diff"], m["label"]))

    colors_by_guidance = {4.0: "#d62728", 1.0: "#1f77b4"}

    for guidance, points in sorted(guidance_groups.items()):
        points_sorted = sorted(points, key=lambda t: t[0])
        xs = [p[0] for p in points_sorted]
        ys = [p[1] for p in points_sorted]
        color = colors_by_guidance.get(guidance, "gray")
        ax.plot(xs, ys, "o-", color=color, lw=1.5, ms=7,
                label=f"guidance={guidance}")
        for step, diff, label in points_sorted:
            ax.annotate(label.split(" s=")[-1] if " s=" in label else "",
                        (step, diff), textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color=color)

    ax.axhline(0, color="black", lw=1, ls="--", label=f"Real slope ({real_slope:.3f})")
    ax.set_xlabel("Inference steps")
    ax.set_ylabel("Slope − Real slope")
    ax.set_title("Step count effect on spectral slope (by guidance level)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_difference_panel(
    real_spectra: np.ndarray,
    cond_spectra: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """4. 2×3 panel of signed spectral difference plots (gen − real)."""
    cond_ids = [c for c in _GRID_CONDITIONS if c in cond_spectra and c != "distilled"][:6]
    if "distilled" in cond_spectra:
        cond_ids = (cond_ids + ["distilled"])[:6]

    n = len(cond_ids)
    if n == 0:
        return

    ncols = 3
    nrows = (n + ncols - 1) // ncols

    apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)

    real_mean = real_spectra.mean(axis=0)
    eps = 1e-10
    x = np.arange(len(real_mean))

    for i, cond_id in enumerate(cond_ids):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        _, _, _, label, color, _ = _GRID_CONDITIONS[cond_id]
        spectra = cond_spectra[cond_id]
        gen_mean = spectra.mean(axis=0)

        diff = np.log10(gen_mean + eps) - np.log10(real_mean + eps)
        ax.plot(x, diff, color=color, lw=1.2)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.fill_between(x, diff, 0, where=(diff > 0), color=color, alpha=0.25, label="over")
        ax.fill_between(x, diff, 0, where=(diff < 0), color="gray", alpha=0.20, label="under")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Freq bin", fontsize=8)
        ax.set_ylabel("Δ log₁₀ power", fontsize=8)

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Signed spectral difference (generated − real)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_detection_auc(
    results: dict,
    out_path: Path,
) -> None:
    """5. Horizontal bar chart of LR CV AUC per condition."""
    apply_style()

    items = sorted(results.items(), key=lambda kv: kv[1]["lr_auc_mean"])
    labels = [m["label"] for _, m in items]
    aucs   = [m["lr_auc_mean"] for _, m in items]
    errs   = [m["lr_auc_std"] for _, m in items]
    colors = [_GRID_CONDITIONS[cid][4] for cid, _ in items]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(items) + 2))
    y_pos = np.arange(len(items))
    ax.barh(y_pos, aucs, xerr=errs, color=colors, edgecolor="k", linewidth=0.5,
            capsize=4, height=0.6)
    ax.axvline(0.5, color="gray", lw=1, ls="--", label="Chance (0.5)")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("LR cross-validation AUC")
    ax.set_title("Detectability: real vs generated (LR, 5-fold CV)")
    ax.set_xlim(0.4, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: dict, real_slope: float) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'Cond':<6} {'Label':<25} {'Steps':>5} {'Guide':>6} {'N':>4} {'Slope':>7} {'ΔSlope':>8} {'Wass':>8} {'LR AUC':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    order = ["A", "B", "C", "D", "E", "F", "distilled"]
    for cond_id in order:
        if cond_id not in results:
            continue
        m = results[cond_id]
        print(
            f"{cond_id:<6} {m['label']:<25} {m['steps']:>5} {m['guidance']:>6.1f} "
            f"{m['n_images']:>4} {m['slope']:>7.3f} {m['slope_diff']:>+8.3f} "
            f"{m['wasserstein']:>8.4f} {m['lr_auc_mean']:>7.3f}±{m['lr_auc_std']:.3f}"
        )
    print("-" * len(header))
    print(f"{'Real':<6} {'FFHQ':<25} {'':>5} {'':>6} {'':>4} {real_slope:>7.3f} {'0.000':>8}")
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    real_spectra, cond_spectra = load_all_spectra()

    if not cond_spectra:
        print("No grid condition directories found. Generate images first with generate_grid.py.")
        return

    # ── Metrics ─────────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    real_slope, results = compute_metrics(real_spectra, cond_spectra)

    metrics_out = {
        "real_slope": real_slope,
        "conditions": results,
    }
    json_path = _METRICS_DIR / "grid_comparison.json"
    with open(json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved → {json_path}")

    # ── Figures ─────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    plot_spectra_overlay(real_spectra, cond_spectra, _FIG_DIR / "grid_spectra_overlay.png")
    plot_slope_vs_guidance(results, real_slope, _FIG_DIR / "grid_slope_vs_guidance.png")
    plot_slope_vs_steps(results, real_slope, _FIG_DIR / "grid_slope_vs_steps.png")
    plot_difference_panel(real_spectra, cond_spectra, _FIG_DIR / "grid_spectral_difference_panel.png")
    plot_detection_auc(results, _FIG_DIR / "grid_detection_auc.png")

    print_summary_table(results, real_slope)


if __name__ == "__main__":
    main()
