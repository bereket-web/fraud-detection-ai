"""End-to-end model evaluation and aggregate report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud_detection.data import DATASET_PAGE, DatasetSplits
from fraud_detection.metrics import binary_metrics, select_threshold
from fraud_detection.models import fit_and_score

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "neural_mlp": "Neural MLP",
    "isolation_forest": "Isolation Forest",
}


def _partition_stats(y: pd.Series) -> dict[str, float | int]:
    return {
        "transactions": int(len(y)),
        "fraud_count": int(y.sum()),
        "fraud_prevalence": float(y.mean()),
    }


def run_benchmark(
    splits: DatasetSplits,
    output_dir: Path,
    seed: int = 2026,
    mlp_epochs: int = 4,
    mlp_steps_per_epoch: int = 60,
    anomaly_train_rows: int = 100_000,
    dataset_sha256: str | None = None,
) -> dict[str, Any]:
    """Train on train, select thresholds on validation, and report once on the test month."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted = fit_and_score(
        splits.x_train,
        splits.y_train,
        splits.x_validation,
        splits.x_test,
        seed=seed,
        mlp_epochs=mlp_epochs,
        mlp_steps_per_epoch=mlp_steps_per_epoch,
        anomaly_train_rows=anomaly_train_rows,
    )

    metric_rows: list[dict[str, Any]] = []
    threshold_records: dict[str, dict[str, float]] = {}
    for model_name, validation_scores in fitted.validation_scores.items():
        test_scores = fitted.test_scores[model_name]
        threshold_records[model_name] = {}
        for policy_name, objective in [("validation_max_f1", "f1"), ("validation_recall_80", "recall")]:
            threshold = select_threshold(
                splits.y_validation.to_numpy(),
                validation_scores,
                objective=objective,
                recall_floor=0.80,
            )
            threshold_records[model_name][policy_name] = threshold
            row = {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "threshold_policy": policy_name,
                "threshold_selected_on": "validation_month_only",
                **binary_metrics(splits.y_test.to_numpy(), test_scores, threshold),
            }
            metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False, float_format="%.8g")
    summary: dict[str, Any] = {
        "project": "Project 11 — Fraud Detection AI",
        "dataset": {
            "name": "Feedzai Bank Account Fraud (BAF), Base variant",
            "version": 2,
            "url": DATASET_PAGE,
            "paper": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html",
            "license": "CC BY-NC-SA 4.0",
            "source_character": "synthetic, privacy-preserving tabular data derived from an anonymized real-world dataset",
            "sha256": dataset_sha256,
            "raw_data_redistributed": False,
        },
        "protocol": {
            "split": "chronological; latest month held out for test, prior month for validation, earlier months for training",
            "training_months": list(splits.train_months),
            "training_pool_rows_before_sampling": splits.train_pool_rows,
            "validation_month": splits.validation_month,
            "test_month": splits.test_month,
            "training_sampled": splits.sampled_train,
            "seed": seed,
            "threshold_selection": "maximum validation F1, or best validation precision among thresholds meeting 80% recall; never selected on test",
            "test_used_for_tuning": False,
        },
        "partitions": {
            "train": _partition_stats(splits.y_train),
            "validation": _partition_stats(splits.y_validation),
            "test": _partition_stats(splits.y_test),
        },
        "models": {
            name: {"label": MODEL_LABELS[name], "notes": fitted.model_notes[name]} for name in fitted.validation_scores
        },
        "validation_thresholds": threshold_records,
        "test_metrics": metric_rows,
        "caveats": [
            "All benchmark observations are synthetic; these results do not establish real-world fraud performance.",
            "The test set is one later month from the same synthetic suite, not an independent institution or live system.",
            "Thresholds are chosen using the validation month; changing costs or alert capacity would change the operating point.",
            "Precision, recall, F1, and false-alarm counts are descriptive for this split; no confidence interval or production claim is made.",
            "The neural model's class-balanced mini-batches alter the training prior; threshold selection uses the natural validation prevalence.",
            "Isolation Forest is fit on training rows labeled legitimate, so it is a one-class baseline with access to known-normal labels.",
        ],
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _write_error_analysis(metrics, summary, output_dir / "error_analysis.md")
    _plot_precision_recall(splits, fitted.test_scores, threshold_records, output_dir / "precision_recall.png")
    return summary


def _write_error_analysis(metrics: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    max_f1 = metrics.loc[metrics["threshold_policy"] == "validation_max_f1"].sort_values(
        "average_precision", ascending=False
    )
    recall_rows = metrics.loc[metrics["threshold_policy"] == "validation_recall_80"].sort_values(
        "recall", ascending=False
    )
    lines = [
        "# Held-out Error Analysis",
        "",
        f"The test partition contains {summary['partitions']['test']['transactions']:,} later-month applications, "
        f"including {summary['partitions']['test']['fraud_count']:,} synthetic fraud labels "
        f"({summary['partitions']['test']['fraud_prevalence']:.2%} prevalence). Thresholds were selected on "
        "the previous month only.",
        "",
        "## Test results at validation-selected maximum-F1 thresholds",
        "",
        "| Model | AP / PR-AUC | ROC-AUC | Precision | Recall | F1 | False positives | False negatives | Alerts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in max_f1.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.average_precision:.4f} | {row.roc_auc:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {row.false_positives:,} | {row.false_negatives:,} | {row.alerts:,} |"
        )
    lines.extend(
        [
            "",
            "## Test results at thresholds selected for at least 80% validation recall",
            "",
            "| Model | Precision | Recall | F1 | False positives | False negatives | Alerts |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in recall_rows.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | "
            f"{row.false_positives:,} | {row.false_negatives:,} | {row.alerts:,} |"
        )
    lines.extend(
        [
            "",
            "Average precision (the area under the stepwise precision-recall curve) is more informative than accuracy "
            "under rare-event imbalance. ROC-AUC is included but should not be read in isolation. The two threshold "
            "policies expose the operational trade-off: missing fraud versus reviewing legitimate applications.",
            "",
            "## Interpretation limits",
            "",
            "These are synthetic benchmark results, not live-bank evidence. The held-out month captures temporal shift "
            "within one generated suite but cannot measure transfer to another country, institution, product, or fraud "
            "process. Counts are split-specific. No cost matrix, human-review capacity, or calibration guarantee is assumed.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _plot_precision_recall(
    splits: DatasetSplits,
    test_scores: dict[str, np.ndarray],
    thresholds: dict[str, dict[str, float]],
    path: Path,
) -> None:
    y_test = splits.y_test.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.axhline(float(np.mean(y_test)), color="#64748b", linestyle="--", linewidth=1, label="No-skill prevalence")
    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]
    for color, (name, scores) in zip(colors, test_scores.items(), strict=False):
        precision, recall, _ = precision_recall_curve(y_test, scores)
        from sklearn.metrics import average_precision_score

        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, color=color, linewidth=2, label=f"{MODEL_LABELS[name]} (AP={ap:.3f})")
        threshold = thresholds[name]["validation_max_f1"]
        at_threshold = binary_metrics(y_test, scores, threshold)
        ax.scatter(at_threshold["recall"], at_threshold["precision"], color=color, s=32, edgecolor="white", zorder=3)
    ax.set(
        title="Precision–recall on the later-month test partition",
        xlabel="Recall (fraud captured)",
        ylabel="Precision (alerts that are fraud)",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
