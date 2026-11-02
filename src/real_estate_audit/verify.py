"""Validate source provenance, report shape, privacy, and locked versions."""

from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

import pandas as pd

from real_estate_audit.constants import RAW_DIR, REPORTS_DIR, SOURCE_SHA256
from real_estate_audit.data import parse_archive, sha256_file

EXPECTED_REPORTS = {
    "RESULTS.md",
    "data_quality.json",
    "fold_metrics.csv",
    "model_metrics.csv",
    "spatial_summary.csv",
    "summary.json",
}
DISALLOWED_COLUMNS = {
    "No",
    "Y house price of unit area",
    "X5 latitude",
    "X6 longitude",
    "prediction",
    "y_pred",
}


def _verify_lock(root: Path) -> int:
    lock_path = root / "requirements.lock"
    if not lock_path.is_file():
        raise ValueError("requirements.lock is missing")
    pinned = 0
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line.strip())
        if match is None:
            raise ValueError(f"Unpinned or non-portable lock entry: {line!r}")
        name, expected = match.groups()
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"Locked dependency is not installed: {name}") from error
        if actual != expected:
            raise ValueError(
                f"{name} version mismatch: lock={expected}, installed={actual}"
            )
        pinned += 1
    if pinned < 20:
        raise ValueError(f"Lock unexpectedly contains only {pinned} distributions")
    return pinned


def verify(root: Path | None = None) -> dict[str, int | str]:
    root = root or Path(__file__).resolve().parents[2]
    archive = RAW_DIR / "real-estate.zip"
    if sha256_file(archive) != SOURCE_SHA256:
        raise ValueError("Raw archive checksum does not match the pinned UCI source")
    frame = parse_archive(archive)
    manifest = json.loads((RAW_DIR.parent / "source_manifest.json").read_text())
    if manifest["archive_sha256"] != SOURCE_SHA256 or manifest["rows"] != len(frame):
        raise ValueError("Source manifest does not match the raw archive")

    actual_reports = {path.name for path in REPORTS_DIR.iterdir() if path.is_file()}
    if actual_reports != EXPECTED_REPORTS:
        raise ValueError(
            f"Unexpected report artifacts: {sorted(actual_reports ^ EXPECTED_REPORTS)}"
        )
    summary = json.loads((REPORTS_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["rows"] != 414 or summary["source_identifier_used_as_feature"]:
        raise ValueError("Summary row count or source identifier exclusion is invalid")
    if any(summary["privacy"].values()):
        raise ValueError("Summary privacy flags must all be false")

    pooled = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
    folds = pd.read_csv(REPORTS_DIR / "fold_metrics.csv")
    spatial = pd.read_csv(REPORTS_DIR / "spatial_summary.csv")
    for name, table in (
        ("model metrics", pooled),
        ("fold metrics", folds),
        ("spatial summary", spatial),
    ):
        exposed = DISALLOWED_COLUMNS.intersection(table.columns)
        if exposed:
            raise ValueError(f"{name} expose disallowed columns: {sorted(exposed)}")
    if len(pooled) != 8 or len(folds) != 48 or len(spatial) != 24:
        raise ValueError(
            "Report row counts do not match the declared evaluation design"
        )
    if folds["evaluation"].value_counts().to_dict() != {"spatial": 24, "shuffled": 24}:
        raise ValueError("Expected 24 spatial and 24 shuffled model-fold records")
    if folds[folds["evaluation"] == "spatial"]["spatial_cluster"].nunique() != 6:
        raise ValueError("Expected six aggregate spatial cluster labels")
    lock_count = _verify_lock(root)
    return {
        "source_sha256": SOURCE_SHA256,
        "source_rows": len(frame),
        "pooled_metric_rows": len(pooled),
        "fold_metric_rows": len(folds),
        "spatial_aggregate_rows": len(spatial),
        "locked_distributions": lock_count,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
