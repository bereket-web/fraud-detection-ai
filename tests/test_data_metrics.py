import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import load_baf, temporal_split
from fraud_detection.metrics import binary_metrics, select_threshold


def test_temporal_split_uses_last_two_months_and_excludes_time_feature():
    frame = pd.DataFrame(
        {
            "month": np.repeat(np.arange(5), 40),
            "fraud_bool": np.tile([0, 1], 100),
            "amount": np.arange(200, dtype=np.float32),
            "kind": np.tile(["A", "B"], 100),
        }
    )
    split = temporal_split(frame, max_train_rows=50, seed=4)
    assert split.train_months == (0, 1, 2)
    assert split.validation_month == 3
    assert split.test_month == 4
    assert len(split.x_train) == 50
    assert split.train_pool_rows == 120
    assert "month" not in split.x_train
    assert split.sampled_train
    assert split.y_validation.mean() == 0.5
    assert split.y_test.mean() == 0.5


def test_training_subsample_is_reproducible():
    frame = pd.DataFrame(
        {
            "month": np.repeat(np.arange(4), 30),
            "fraud_bool": np.tile([0, 1], 60),
            "amount": np.arange(120),
        }
    )
    left = temporal_split(frame, max_train_rows=20, seed=12)
    right = temporal_split(frame, max_train_rows=20, seed=12)
    pd.testing.assert_frame_equal(left.x_train, right.x_train)
    pd.testing.assert_series_equal(left.y_train, right.y_train)


def test_temporal_split_requires_three_months_and_both_test_labels():
    frame = pd.DataFrame({"month": [0, 0, 1, 1], "fraud_bool": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="three distinct months"):
        temporal_split(frame)


def test_load_baf_rejects_non_binary_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"fraud_bool": [0, 2], "month": [0, 1], "x": [1.0, 2.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary 0/1"):
        load_baf(path)


def test_validation_thresholds_and_test_metrics_are_consistent():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.1, 0.8, 0.7, 0.9, 0.95])
    f1_threshold = select_threshold(y, scores, objective="f1")
    recall_threshold = select_threshold(y, scores, objective="recall", recall_floor=0.8)
    assert f1_threshold > 0
    assert recall_threshold > 0
    report = binary_metrics(y, scores, threshold=0.75)
    assert report["true_positives"] == 2
    assert report["false_positives"] == 1
    assert report["false_negatives"] == 1
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["recall"] == pytest.approx(2 / 3)
    assert report["average_precision"] > 0.5


def test_threshold_selection_rejects_single_class_validation():
    with pytest.raises(ValueError, match="both positive and negative"):
        select_threshold(np.zeros(4, dtype=int), np.arange(4, dtype=float))
