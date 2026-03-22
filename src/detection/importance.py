"""Feature importance and coefficient extraction for detection classifiers.

Provides utilities to retrieve and rank feature importances from fitted
Random Forest and Logistic Regression models.
"""

from __future__ import annotations

import numpy as np


def get_feature_names(n_bins: int = 128, n_bands: int = 3) -> list[str]:
    """Generate ordered feature names matching the ``extract_all_features`` layout.

    Parameters
    ----------
    n_bins : int, optional
        Number of interpolated spectrum bins.  Defaults to 128.
    n_bands : int, optional
        Number of frequency bands.  Currently only 3 is fully labelled.
        Defaults to 3.

    Returns
    -------
    list of str
        List of 133 feature name strings in the order they appear in the
        feature vector produced by ``extract_all_features``:

        * ``"spectrum_bin_000"`` … ``"spectrum_bin_127"`` (128 names)
        * ``"band_energy_low"``, ``"band_energy_mid"``,
          ``"band_energy_high"`` (3 names)
        * ``"spectral_slope"``, ``"spectral_intercept"`` (2 names)
    """
    spectrum_names = [f"spectrum_bin_{i:03d}" for i in range(n_bins)]
    band_labels = ["band_energy_low", "band_energy_mid", "band_energy_high"]
    if n_bands != 3:
        band_labels = [f"band_energy_{i}" for i in range(n_bands)]
    slope_names = ["spectral_slope", "spectral_intercept"]
    return spectrum_names + band_labels + slope_names


def random_forest_importance(rf_clf, feature_names: list[str]):
    """Extract feature importances from a fitted Random Forest classifier.

    Parameters
    ----------
    rf_clf : sklearn.ensemble.RandomForestClassifier
        A fitted ``RandomForestClassifier`` instance.
    feature_names : list of str
        Ordered list of feature names (length must equal the number of
        features the model was trained on).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``["feature", "importance"]``, sorted in
        descending order of importance.
    """
    import pandas as pd

    importances = rf_clf.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def logistic_regression_coefficients(lr_clf, feature_names: list[str]):
    """Extract coefficients from a fitted Logistic Regression classifier.

    For binary classification the coefficient vector is ``coef_[0]``.
    Features are ranked by absolute coefficient magnitude.

    Parameters
    ----------
    lr_clf : sklearn.linear_model.LogisticRegression
        A fitted ``LogisticRegression`` instance.
    feature_names : list of str
        Ordered list of feature names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``["feature", "coefficient"]``, sorted in
        descending order of ``abs(coefficient)``.
    """
    import pandas as pd

    coefs = lr_clf.coef_[0]
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
    })
    df["abs_coefficient"] = df["coefficient"].abs()
    df = df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["abs_coefficient"])
    return df


def top_k_features(importance_df, k: int = 20):
    """Return the top-k rows from a feature importance DataFrame.

    Parameters
    ----------
    importance_df : pandas.DataFrame
        DataFrame as returned by ``random_forest_importance`` or
        ``logistic_regression_coefficients``, already sorted in descending
        importance order.
    k : int, optional
        Number of top features to return.  Defaults to 20.

    Returns
    -------
    pandas.DataFrame
        The first *k* rows of *importance_df*.
    """
    return importance_df.head(k).reset_index(drop=True)
