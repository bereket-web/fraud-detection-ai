"""Dataset access and leakage-aware temporal splitting for the BAF benchmark."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sgpjesus/bank-account-fraud-dataset-neurips-2022/Base.csv?datasetVersionNumber=2"
)
DATASET_PAGE = "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"
TARGET = "fraud_bool"
TIME_COLUMN = "month"


@dataclass(frozen=True)
class DatasetSplits:
    """Chronological partitions and the two held-out month identifiers."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_validation: pd.DataFrame
    y_validation: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    train_months: tuple[int, ...]
    train_pool_rows: int
    validation_month: int
    test_month: int
    sampled_train: bool


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for local provenance checks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_baf(destination: Path, timeout: tuple[int, int] = (30, 180)) -> Path:
    """Download only the official Base.csv file from Kaggle's public dataset API."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_path = destination.with_suffix(destination.suffix + ".zip.part")
    extracted_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with requests.get(DATASET_URL, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "zip" not in content_type.lower():
                raise ValueError(f"Expected a ZIP response from Kaggle, received {content_type!r}.")
            with archive_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)

        with zipfile.ZipFile(archive_path) as archive:
            matches = [name for name in archive.namelist() if Path(name).name.lower() == "base.csv"]
            if len(matches) != 1:
                raise ValueError("The version-pinned BAF archive did not contain exactly one Base.csv.")
            with archive.open(matches[0]) as source, extracted_path.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)

        if extracted_path.stat().st_size == 0:
            raise ValueError("Downloaded Base.csv is empty.")
        extracted_path.replace(destination)
    finally:
        archive_path.unlink(missing_ok=True)
        extracted_path.unlink(missing_ok=True)

    return destination


def load_baf(path: Path) -> pd.DataFrame:
    """Read and validate BAF Base.csv without silently changing the label or time field."""
    frame = pd.read_csv(path, low_memory=False)
    if TARGET not in frame or TIME_COLUMN not in frame:
        raise ValueError(f"Expected columns {TARGET!r} and {TIME_COLUMN!r} in the dataset.")
    if frame[TARGET].isna().any() or frame[TIME_COLUMN].isna().any():
        raise ValueError("The target and month columns must not contain missing values.")
    labels = set(frame[TARGET].unique())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"Expected a binary 0/1 fraud label; found {sorted(labels)}.")
    if not pd.api.types.is_numeric_dtype(frame[TIME_COLUMN]):
        raise ValueError("The month column must be numeric for chronological splitting.")

    for column in frame.select_dtypes(include=["number"]).columns:
        if column != TARGET:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.float32)
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].astype("string")
    frame[TARGET] = frame[TARGET].astype(np.int8)
    return frame


def temporal_split(
    frame: pd.DataFrame,
    max_train_rows: int = 250_000,
    seed: int = 2026,
) -> DatasetSplits:
    """Train on earlier months and reserve the last two months for validation and test."""
    if TARGET not in frame or TIME_COLUMN not in frame:
        raise ValueError("The frame must include fraud_bool and month columns.")
    months = tuple(sorted(int(value) for value in frame[TIME_COLUMN].dropna().unique()))
    if len(months) < 3:
        raise ValueError("At least three distinct months are required for train/validation/test.")

    validation_month, test_month = months[-2:]
    train = frame.loc[frame[TIME_COLUMN] < validation_month]
    validation = frame.loc[frame[TIME_COLUMN] == validation_month]
    test = frame.loc[frame[TIME_COLUMN] == test_month]
    train_pool_rows = len(train)
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal partitioning produced an empty train, validation, or test set.")

    sampled = max_train_rows > 0 and len(train) > max_train_rows
    if sampled:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(train), size=max_train_rows, replace=False))
        train = train.iloc[positions]

    feature_columns = [column for column in frame.columns if column not in {TARGET, TIME_COLUMN}]

    def separate(partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        x = partition.loc[:, feature_columns].reset_index(drop=True)
        y = partition[TARGET].astype(np.int8).reset_index(drop=True)
        return x, y

    x_train, y_train = separate(train)
    x_validation, y_validation = separate(validation)
    x_test, y_test = separate(test)
    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        train_months=tuple(month for month in months if month < validation_month),
        train_pool_rows=train_pool_rows,
        validation_month=validation_month,
        test_month=test_month,
        sampled_train=sampled,
    )
