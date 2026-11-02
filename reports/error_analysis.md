# Held-out Error Analysis

The test partition contains 96,843 later-month applications, including 1,428 synthetic fraud labels (1.47% prevalence). Thresholds were selected on the previous month only.

## Test results at validation-selected maximum-F1 thresholds

| Model | AP / PR-AUC | ROC-AUC | Precision | Recall | F1 | False positives | False negatives | Alerts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.1783 | 0.8816 | 0.2408 | 0.2626 | 0.2513 | 1,182 | 1,053 | 1,557 |
| Neural MLP | 0.1734 | 0.8825 | 0.2138 | 0.2759 | 0.2409 | 1,449 | 1,034 | 1,843 |
| Random forest | 0.1681 | 0.8719 | 0.2862 | 0.1814 | 0.2220 | 646 | 1,169 | 905 |
| Isolation Forest | 0.0196 | 0.5778 | 0.0237 | 0.0637 | 0.0345 | 3,750 | 1,337 | 3,841 |

## Test results at thresholds selected for at least 80% validation recall

| Model | Precision | Recall | F1 | False positives | False negatives | Alerts |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.0164 | 0.9041 | 0.0323 | 77,240 | 137 | 78,531 |
| Random forest | 0.0524 | 0.8151 | 0.0984 | 21,061 | 264 | 22,225 |
| Neural MLP | 0.0611 | 0.8011 | 0.1135 | 17,583 | 284 | 18,727 |
| Logistic regression | 0.0665 | 0.7745 | 0.1224 | 15,538 | 322 | 16,644 |

Average precision (the area under the stepwise precision-recall curve) is more informative than accuracy under rare-event imbalance. ROC-AUC is included but should not be read in isolation. The two threshold policies expose the operational trade-off: missing fraud versus reviewing legitimate applications.

## Interpretation limits

These are synthetic benchmark results, not live-bank evidence. The held-out month captures temporal shift within one generated suite but cannot measure transfer to another country, institution, product, or fraud process. Counts are split-specific. No cost matrix, human-review capacity, or calibration guarantee is assumed.
