"""Preprocessing safeguards for the Black Sigatoka synthetic baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from numbers import Real
from typing import Mapping

import numpy as np
import pandas as pd


FEATURES = ["air_temp_c", "estimated_leaf_wet", "consecutive_wetness_hours"]
TARGET = "risk_score"
EXPECTED_ROWS = {"train": 4501, "validation": 937, "test": 524}
SPLIT_FILES = {
    name: f"BS_SOURCE_TIME_{name}.csv" for name in EXPECTED_ROWS
}
CATEGORY_BOUNDS = {
    "No Risk": "score == 0",
    "Low": "0 < score < 25",
    "Moderate": "25 <= score < 75",
    "High": "75 <= score <= 100",
}


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[1]


def _check_contract(root: Path) -> dict:
    contract_path = root / "reports" / "black_sigatoka" / "final_safeguards" / "model_input_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Required model input contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("feature_order") != FEATURES or contract.get("target") != TARGET:
        raise ValueError("The model input contract disagrees with the approved features or target.")
    if contract.get("risk_category_bounds") != CATEGORY_BOUNDS:
        raise ValueError("The model input contract disagrees with the documented risk categories.")
    expected_paths = {
        split: f"datasets\\black_sigatoka\\revised_splits\\{filename}"
        for split, filename in SPLIT_FILES.items()
    }
    if contract.get("allowed_training_files") != expected_paths:
        raise ValueError("The model input contract does not point to the approved revised split files.")
    return contract


def load_and_validate_datasets(
    project_root: str | Path | None = None,
    splits: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Load selected revised partitions and fail clearly on data/contract issues.

    By default all three partitions are loaded for the preprocessing workflow.
    Training experiments can request only train and validation so the held-out
    test file is not read during model selection.
    """
    root = _project_root(project_root)
    _check_contract(root)
    source_dir = root / "datasets" / "black_sigatoka" / "revised_splits"
    selected_splits = tuple(EXPECTED_ROWS) if splits is None else tuple(splits)
    if not selected_splits or len(set(selected_splits)) != len(selected_splits):
        raise ValueError("Specify one or more unique dataset splits.")
    unknown_splits = sorted(set(selected_splits) - set(EXPECTED_ROWS))
    if unknown_splits:
        raise ValueError(f"Unknown dataset split(s): {unknown_splits}")
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    expected_columns: list[str] | None = None

    for split in selected_splits:
        filename = SPLIT_FILES[split]
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required {split} CSV is missing: {path}")
        frame = pd.read_csv(path)
        if len(frame) != EXPECTED_ROWS[split]:
            raise ValueError(
                f"{split} expected {EXPECTED_ROWS[split]} rows, found {len(frame)}."
            )
        if expected_columns is None:
            expected_columns = frame.columns.tolist()
        elif frame.columns.tolist() != expected_columns:
            raise ValueError(f"{split} schema or column order differs from the other splits.")

        required = FEATURES + [TARGET, "risk_level"]
        absent = [column for column in required if column not in frame.columns]
        if absent:
            raise ValueError(f"{split} is missing required columns: {absent}")

        selected = frame[FEATURES + [TARGET]]
        missing = selected.isna().sum()
        if missing.any():
            details = {key: int(value) for key, value in missing[missing > 0].items()}
            raise ValueError(f"{split} has unexpected missing selected values: {details}")
        try:
            numeric = selected.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{split} selected features and target must be numeric.") from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f"{split} selected features and target contain non-finite values.")
        if not frame[TARGET].between(0, 100).all():
            raise ValueError(f"{split} risk scores must be within 0 and 100.")
        if not frame["estimated_leaf_wet"].isin([0, 1]).all():
            raise ValueError(f"{split} estimated_leaf_wet must contain only 0 or 1.")
        if (frame["consecutive_wetness_hours"] < 0).any():
            raise ValueError(f"{split} consecutive wetness duration cannot be negative.")

        expected_levels = frame[TARGET].map(risk_score_to_category)
        mismatches = int(frame["risk_level"].ne(expected_levels).sum())
        if mismatches:
            raise ValueError(f"{split} contains {mismatches} scores with inconsistent risk categories.")

        frames[split] = frame
        summaries[split] = {
            "split": split,
            "source_file": f"datasets/black_sigatoka/revised_splits/{filename}",
            "rows": len(frame),
            "columns": len(frame.columns),
            "schema_consistent": True,
            "selected_missing_values": int(selected.isna().sum().sum()),
            "selected_nonfinite_values": int((~np.isfinite(numeric)).sum()),
            "risk_score_min": float(frame[TARGET].min()),
            "risk_score_max": float(frame[TARGET].max()),
            "risk_category_mismatches": mismatches,
            "input_shape": f"{len(frame)} x {len(FEATURES)}",
            "target_shape": f"{len(frame)}",
        }
    return frames, summaries


def verify_source_time_separation(
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Recalculate donor-time separation directly from all revised source CSVs.

    Sorted nanosecond timestamps and binary search find nearest cross-split
    neighbors without constructing a pairwise Cartesian distance matrix.
    Raises ValueError if timestamps are invalid, overlap, or are within 120 h.
    """
    root = _project_root(project_root)
    source_dir = root / "datasets" / "black_sigatoka" / "revised_splits"
    donor_times: dict[str, pd.DatetimeIndex] = {}

    for split, filename in SPLIT_FILES.items():
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required {split} CSV is missing: {path}")
        try:
            frame = pd.read_csv(path, usecols=["weather_analogue_timestamp_utc"])
        except ValueError as exc:
            raise ValueError(f"{split} CSV lacks weather_analogue_timestamp_utc.") from exc
        raw = frame["weather_analogue_timestamp_utc"]
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        if parsed.empty:
            raise ValueError(f"{split} has no donor timestamps to verify.")
        invalid = parsed.isna()
        if invalid.any():
            rows = (np.flatnonzero(invalid.to_numpy())[:5] + 2).tolist()
            raise ValueError(
                f"{split} has {int(invalid.sum())} missing or invalid donor timestamps "
                f"in CSV row(s) {rows}."
            )
        donor_times[split] = pd.DatetimeIndex(
            parsed.drop_duplicates().sort_values()
        ).as_unit("ns")

    pairs = []
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        left_times = donor_times[left]
        right_times = donor_times[right]
        shared = len(left_times.intersection(right_times))
        left_ns = left_times.asi8
        right_ns = right_times.asi8

        # Check the nearest timestamp at or around each insertion point.
        positions = np.searchsorted(right_ns, left_ns, side="left")
        candidates = np.clip(positions, 0, len(right_ns) - 1)
        distances = np.abs(right_ns[candidates] - left_ns)
        prior_positions = positions - 1
        has_prior = prior_positions >= 0
        distances[has_prior] = np.minimum(
            distances[has_prior],
            np.abs(right_ns[prior_positions[has_prior]] - left_ns[has_prior]),
        )
        minimum_gap_hours = float(distances.min() / pd.Timedelta(hours=1).value)
        passes_gap = minimum_gap_hours > 120
        pairs.append({
            "pair": f"{left}-{right}",
            "left_donor_count": len(left_times),
            "right_donor_count": len(right_times),
            "shared_donor_timestamps": shared,
            "minimum_gap_hours": minimum_gap_hours,
            "passes_120_hour_gap": passes_gap,
        })

    results = pd.DataFrame(pairs)
    failed = results[
        results["shared_donor_timestamps"].ne(0)
        | ~results["passes_120_hour_gap"]
    ]
    if not failed.empty:
        raise ValueError(
            "Source-time separation failed; each pair must have zero shared "
            "timestamps and a minimum gap greater than 120 hours:\n"
            + failed.to_string(index=False)
        )
    return results


def prepare_model_inputs(
    frames: Mapping[str, pd.DataFrame],
    splits: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Select approved columns in fixed order without modifying records."""
    inputs: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.Series] = {}
    selected_splits = tuple(frames) if splits is None else tuple(splits)
    for split in selected_splits:
        if split not in frames:
            raise ValueError(f"Missing required dataset partition: {split}")
        frame = frames[split]
        X = frame.loc[:, FEATURES].copy()
        y = frame.loc[:, TARGET].copy()
        if X.columns.tolist() != FEATURES or X.shape[1] != 3:
            raise ValueError(f"{split} inputs do not match the approved feature order.")
        if len(X) != len(y):
            raise ValueError(f"{split} input and target row counts do not match.")
        if not all(pd.api.types.is_numeric_dtype(X[column]) for column in FEATURES):
            raise ValueError(f"{split} inputs must all have numeric dtypes.")
        if not np.isfinite(X.to_numpy(dtype=np.float64)).all() or not np.isfinite(y.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{split} inputs or targets contain non-finite values.")
        inputs[split], targets[split] = X, y
    return inputs, targets


def validate_prediction_input(data: Mapping[str, object] | pd.DataFrame) -> pd.DataFrame:
    """Validate one or more current-time rows and return the three ordered inputs."""
    if isinstance(data, Mapping):
        frame = pd.DataFrame([dict(data)])
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        raise TypeError("Prediction input must be a mapping or pandas DataFrame.")
    if frame.empty:
        raise ValueError("Prediction input must contain at least one row.")
    absent = [feature for feature in FEATURES if feature not in frame.columns]
    unexpected = [column for column in frame.columns if column not in FEATURES]
    if absent or unexpected:
        raise ValueError(f"Prediction input columns must be exactly {FEATURES}; missing={absent}, unexpected={unexpected}.")
    try:
        frame = frame.loc[:, FEATURES].astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction features must contain numeric values.") from exc
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Prediction features must contain finite values.")
    if not frame["estimated_leaf_wet"].isin([0, 1]).all():
        raise ValueError("estimated_leaf_wet must be 0 or 1.")
    if (frame["consecutive_wetness_hours"] < 0).any():
        raise ValueError("consecutive_wetness_hours cannot be negative.")
    return frame


def risk_score_to_category(score: Real) -> str:
    """Convert one bounded numeric risk score using the documented thresholds."""
    if isinstance(score, bool) or not isinstance(score, Real):
        raise TypeError("Risk score must be a real numeric scalar.")
    numeric = float(score)
    if not math.isfinite(numeric):
        raise ValueError("Risk score must be finite.")
    if not 0 <= numeric <= 100:
        raise ValueError("Risk score must be within 0 and 100.")
    if numeric == 0:
        return "No Risk"
    if numeric < 25:
        return "Low"
    if numeric < 75:
        return "Moderate"
    return "High"
