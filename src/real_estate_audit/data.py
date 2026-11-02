"""Workbook parsing, provenance, and source-quality checks."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from real_estate_audit.constants import (
    FEATURE_COLUMNS,
    RAW_DIR,
    SOURCE_BYTES,
    SOURCE_ID,
    SOURCE_MEMBER,
    SOURCE_SHA256,
    SOURCE_URL,
    TARGET_COLUMN,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_archive(archive_path: Path | None = None) -> pd.DataFrame:
    archive_path = archive_path or RAW_DIR / "real-estate.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing source archive at {archive_path}; run make fetch first."
        )
    if archive_path.stat().st_size != SOURCE_BYTES:
        raise ValueError("UCI archive byte count differs from the pinned snapshot")
    digest = sha256_file(archive_path)
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"UCI archive checksum mismatch: expected {SOURCE_SHA256}, got {digest}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        if SOURCE_MEMBER not in archive.namelist():
            raise ValueError(f"Expected workbook is missing: {SOURCE_MEMBER}")
        payload = archive.read(SOURCE_MEMBER)
    frame = pd.read_excel(
        io.BytesIO(payload),
        sheet_name=0,
        engine="openpyxl",
    )
    required = {SOURCE_ID, *FEATURE_COLUMNS, TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"UCI workbook is missing columns: {sorted(missing)}")
    frame = frame[[SOURCE_ID, *FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    for column in (SOURCE_ID, *FEATURE_COLUMNS, TARGET_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any():
        raise ValueError("UCI workbook contains missing values")
    if (frame[TARGET_COLUMN] <= 0).any():
        raise ValueError("Unit-area price must be positive")
    if frame[SOURCE_ID].duplicated().any():
        raise ValueError("Source row identifiers are not unique")
    return frame.reset_index(drop=True)


def data_quality_report(frame: pd.DataFrame, archive_path: Path) -> dict[str, Any]:
    features = frame[FEATURE_COLUMNS]
    return {
        "source": {
            "url": SOURCE_URL,
            "archive": "real-estate.zip",
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
            "workbook": SOURCE_MEMBER,
        },
        "rows": len(frame),
        "feature_count": len(FEATURE_COLUMNS),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_source_ids": int(frame[SOURCE_ID].duplicated().sum()),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "unique_feature_rows": int(features.drop_duplicates().shape[0]),
        "target_summary": {
            "mean": float(frame[TARGET_COLUMN].mean()),
            "first_quartile": float(frame[TARGET_COLUMN].quantile(0.25)),
            "median": float(frame[TARGET_COLUMN].median()),
            "third_quartile": float(frame[TARGET_COLUMN].quantile(0.75)),
            "unit": "10,000 New Taiwan dollars per ping",
        },
        "coordinate_bounds_rounded_3dp": {
            "latitude_min": round(float(frame["X5 latitude"].min()), 3),
            "latitude_max": round(float(frame["X5 latitude"].max()), 3),
            "longitude_min": round(float(frame["X6 longitude"].min()), 3),
            "longitude_max": round(float(frame["X6 longitude"].max()), 3),
        },
        "modeling": {
            "source_identifier_in_features": False,
            "exact_coordinates_published": False,
            "row_level_predictions_published": False,
        },
    }


def write_source_manifest(
    archive_path: Path, frame: pd.DataFrame
) -> dict[str, str | int]:
    manifest = {
        "dataset": "UCI Real Estate Valuation",
        "source_url": SOURCE_URL,
        "doi": "10.24432/C5J30W",
        "license": "CC BY 4.0",
        "citation": (
            "Yeh, I. (2018). Real Estate Valuation [Dataset]. "
            "UCI Machine Learning Repository."
        ),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "expected_sha256": SOURCE_SHA256,
        "workbook": SOURCE_MEMBER,
        "rows": len(frame),
        "features": len(FEATURE_COLUMNS),
    }
    manifest_path = RAW_DIR.parent / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
