# Contributing

1. Keep BAF raw files, fitted models, and transaction-level predictions out of version control. Check the dataset's CC BY-NC-SA 4.0 terms before using or sharing data.
2. Preserve the chronological train/validation/test design. Do not tune thresholds or hyperparameters against the final test month.
3. Report precision, recall, F1, average precision, false positives, and false negatives; do not headline accuracy on this rare-event task.
4. Label synthetic results as synthetic and avoid claims about production performance.
5. Before proposing a change, run `uv run pytest` and `uv run ruff check .`.
