import numpy as np
import pandas as pd

from fraud_detection.models import build_preprocessor, fit_and_score


def test_preprocessor_handles_unseen_categories():
    train = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "channel": ["web", "app", "web"]})
    validation = pd.DataFrame({"amount": [4.0], "channel": ["new-channel"]})
    preprocessor = build_preprocessor(train)
    matrix = preprocessor.fit_transform(train)
    transformed = preprocessor.transform(validation)
    assert matrix.shape[0] == 3
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed.data).all()


def test_all_model_families_return_aligned_risk_scores():
    rng = np.random.default_rng(17)
    month_parts = []
    for month in range(4):
        n = 300
        x = rng.normal(size=n).astype(np.float32)
        y = (rng.random(n) < 0.18).astype(np.int8)
        x[y == 1] += 0.8
        month_parts.append(
            pd.DataFrame(
                {
                    "month": month,
                    "fraud_bool": y,
                    "amount": np.abs(x),
                    "signal": x,
                    "channel": np.where(x > 0.5, "mobile", "web"),
                }
            )
        )
    frame = pd.concat(month_parts, ignore_index=True)
    train = frame.loc[frame.month < 2]
    validation = frame.loc[frame.month == 2]
    test = frame.loc[frame.month == 3]
    result = fit_and_score(
        train.drop(columns=["month", "fraud_bool"]),
        train["fraud_bool"],
        validation.drop(columns=["month", "fraud_bool"]),
        test.drop(columns=["month", "fraud_bool"]),
        seed=9,
        mlp_epochs=1,
        mlp_steps_per_epoch=2,
        anomaly_train_rows=500,
    )
    expected = {"logistic_regression", "random_forest", "neural_mlp", "isolation_forest"}
    assert set(result.validation_scores) == expected
    assert set(result.test_scores) == expected
    for name in expected:
        assert result.validation_scores[name].shape == (len(validation),)
        assert result.test_scores[name].shape == (len(test),)
        assert np.isfinite(result.validation_scores[name]).all()
        assert np.isfinite(result.test_scores[name]).all()
