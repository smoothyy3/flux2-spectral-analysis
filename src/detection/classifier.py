"""Classifier training and cross-validated evaluation for detection experiments.

Wraps scikit-learn classifiers with cross-validation, ROC-AUC computation,
and a final full-dataset fit for feature importance extraction.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm


def make_classifiers(random_seed: int = 42) -> dict:
    """Instantiate the standard set of classifiers used in detection experiments.

    Parameters
    ----------
    random_seed : int, optional
        Random seed for reproducibility.  Defaults to 42.

    Returns
    -------
    dict
        Mapping of classifier name → unfitted sklearn estimator:

        ``"logistic_regression"``
            :class:`sklearn.linear_model.LogisticRegression` with
            ``max_iter=1000``.
        ``"svm_rbf"``
            :class:`sklearn.svm.SVC` with RBF kernel and
            ``probability=True``.
        ``"random_forest"``
            :class:`sklearn.ensemble.RandomForestClassifier` with
            ``n_estimators=100``.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_seed,
        ),
    }


def evaluate_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict:
    """Evaluate a classifier with stratified k-fold cross-validation.

    For each fold, the classifier is fitted on the training split and
    evaluated on the held-out test split using accuracy, ROC-AUC
    (via ``predict_proba``), and macro-averaged F1 score.

    Parameters
    ----------
    clf : sklearn estimator
        An unfitted sklearn classifier that supports ``fit``, ``predict``,
        and ``predict_proba``.
    X : np.ndarray
        Feature matrix of shape (N, D).
    y : np.ndarray
        Label vector of shape (N,) with binary labels.
    cv_folds : int, optional
        Number of cross-validation folds.  Defaults to 5.
    random_seed : int, optional
        Random seed for ``StratifiedKFold``.  Defaults to 42.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``accuracy_mean``, ``accuracy_std`` : float
            Mean and standard deviation of fold accuracies.
        ``roc_auc_mean``, ``roc_auc_std`` : float
            Mean and standard deviation of fold ROC-AUC scores.
        ``f1_mean``, ``f1_std`` : float
            Mean and standard deviation of fold macro F1 scores.
        ``fold_results`` : list of dict
            Per-fold dictionaries each containing
            ``accuracy``, ``roc_auc``, and ``f1``.
    """
    import copy
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_results: list[dict] = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scaler fitted on train split only — prevents test-set leakage.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        fold_clf = copy.deepcopy(clf)
        fold_clf.fit(X_train, y_train)

        y_pred = fold_clf.predict(X_test)
        y_proba = fold_clf.predict_proba(X_test)[:, 1]

        fold_results.append({
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred, average="macro")),
        })

    accuracies = np.array([f["accuracy"] for f in fold_results])
    aucs = np.array([f["roc_auc"] for f in fold_results])
    f1s = np.array([f["f1"] for f in fold_results])

    return {
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std()),
        "roc_auc_mean": float(aucs.mean()),
        "roc_auc_std": float(aucs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "fold_results": fold_results,
    }


def evaluate_all_classifiers(
    classifiers: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict:
    """Evaluate every classifier in *classifiers* with cross-validation.

    Parameters
    ----------
    classifiers : dict
        Mapping of classifier name → sklearn estimator (from
        ``make_classifiers``).
    X : np.ndarray
        Feature matrix of shape (N, D).
    y : np.ndarray
        Label vector of shape (N,).
    cv_folds : int, optional
        Number of cross-validation folds.  Defaults to 5.
    random_seed : int, optional
        Random seed.  Defaults to 42.

    Returns
    -------
    dict
        Nested dictionary mapping classifier name → evaluation result dict
        (as returned by ``evaluate_classifier``).
    """
    results = {}
    for name, clf in tqdm(classifiers.items(), desc="Evaluating classifiers"):
        results[name] = evaluate_classifier(
            clf, X, y, cv_folds=cv_folds, random_seed=random_seed
        )
    return results


def fit_final_classifier(clf, X: np.ndarray, y: np.ndarray):
    """Fit a classifier on the entire dataset.

    Used for extracting feature importances and computing ROC curves over
    the full training set.

    Parameters
    ----------
    clf : sklearn estimator
        An unfitted sklearn classifier.
    X : np.ndarray
        Feature matrix of shape (N, D).
    y : np.ndarray
        Label vector of shape (N,).

    Returns
    -------
    object
        The fitted classifier instance.
    """
    clf.fit(X, y)
    return clf
