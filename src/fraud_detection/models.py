"""Model definitions for supervised and one-class fraud detection baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FittedBenchmark:
    """Fitted preprocessing and validation/test scores for every approach."""

    preprocessor: ColumnTransformer
    validation_scores: dict[str, np.ndarray]
    test_scores: dict[str, np.ndarray]
    model_notes: dict[str, str]


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build train-fitted numeric scaling and categorical one-hot encoding."""
    categorical = list(features.select_dtypes(include=["object", "string", "category"]).columns)
    numeric = [column for column in features.columns if column not in categorical]
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=25,
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def _score_classifier(model: object, features: object) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    fraud_column = classes.index(1)
    return np.asarray(probabilities[:, fraud_column], dtype=np.float64)


def fit_and_score(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    x_test: pd.DataFrame,
    seed: int = 2026,
    mlp_epochs: int = 4,
    mlp_steps_per_epoch: int = 60,
    anomaly_train_rows: int = 100_000,
) -> FittedBenchmark:
    """Fit all models on earlier data and score untouched validation and test months."""
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training requires both fraud and legitimate examples.")
    preprocessor = build_preprocessor(x_train)
    train_matrix = preprocessor.fit_transform(x_train)
    validation_matrix = preprocessor.transform(x_validation)
    test_matrix = preprocessor.transform(x_test)
    if not sparse.issparse(train_matrix):
        train_matrix = sparse.csr_matrix(train_matrix, dtype=np.float32)
        validation_matrix = sparse.csr_matrix(validation_matrix, dtype=np.float32)
        test_matrix = sparse.csr_matrix(test_matrix, dtype=np.float32)
    else:
        train_matrix = train_matrix.tocsr().astype(np.float32, copy=False)
        validation_matrix = validation_matrix.tocsr().astype(np.float32, copy=False)
        test_matrix = test_matrix.tocsr().astype(np.float32, copy=False)

    labels = np.asarray(y_train, dtype=np.int8)
    scores_validation: dict[str, np.ndarray] = {}
    scores_test: dict[str, np.ndarray] = {}

    logistic = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        solver="liblinear",
        max_iter=500,
        random_state=seed,
    )
    logistic.fit(train_matrix, labels)
    scores_validation["logistic_regression"] = _score_classifier(logistic, validation_matrix)
    scores_test["logistic_regression"] = _score_classifier(logistic, test_matrix)

    forest = RandomForestClassifier(
        n_estimators=120,
        max_features="sqrt",
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    forest.fit(train_matrix, labels)
    scores_validation["random_forest"] = _score_classifier(forest, validation_matrix)
    scores_test["random_forest"] = _score_classifier(forest, test_matrix)

    # MLPClassifier does not offer class_weight in its fit API. Balanced mini-batches
    # provide explicit cost-sensitive training while validation/test keep natural prevalence.
    neural = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=1,
        shuffle=True,
        random_state=seed,
        early_stopping=False,
    )
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) < 2 or len(negative) < 4:
        raise ValueError("The neural model needs multiple examples of each class.")
    positives_per_batch = 128
    negatives_per_batch = 384
    first_update = True
    for _ in range(mlp_epochs):
        for _ in range(mlp_steps_per_epoch):
            indices = np.concatenate(
                [
                    rng.choice(positive, positives_per_batch, replace=True),
                    rng.choice(negative, negatives_per_batch, replace=False),
                ]
            )
            rng.shuffle(indices)
            if first_update:
                neural.partial_fit(train_matrix[indices], labels[indices], classes=np.array([0, 1]))
                first_update = False
            else:
                neural.partial_fit(train_matrix[indices], labels[indices])
    scores_validation["neural_mlp"] = _score_classifier(neural, validation_matrix)
    scores_test["neural_mlp"] = _score_classifier(neural, test_matrix)

    # A one-class anomaly baseline is fitted only on known legitimate training examples.
    # Its anomaly score is negated so that larger values always mean higher fraud risk.
    normal_indices = np.flatnonzero(labels == 0)
    if len(normal_indices) > anomaly_train_rows:
        normal_indices = rng.choice(normal_indices, anomaly_train_rows, replace=False)
    anomaly = IsolationForest(
        n_estimators=160,
        max_samples=256,
        contamination="auto",
        n_jobs=-1,
        random_state=seed,
    )
    anomaly.fit(train_matrix[normal_indices])
    scores_validation["isolation_forest"] = -np.asarray(anomaly.score_samples(validation_matrix), dtype=np.float64)
    scores_test["isolation_forest"] = -np.asarray(anomaly.score_samples(test_matrix), dtype=np.float64)

    notes = {
        "logistic_regression": "Standardized numeric + one-hot categorical inputs; class_weight=balanced.",
        "random_forest": "One-hot inputs; class_weight=balanced_subsample; 120 trees.",
        "neural_mlp": "Two hidden layers; mini-batches sampled 1:3 fraud:legitimate; threshold calibrated on natural prevalence.",
        "isolation_forest": f"IsolationForest fit on known legitimate rows only; training cap={anomaly_train_rows:,}; higher score means more anomalous.",
    }
    return FittedBenchmark(preprocessor, scores_validation, scores_test, notes)
