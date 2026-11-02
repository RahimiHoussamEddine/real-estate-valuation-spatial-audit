"""Tests for data scope, spatial folds, and report API behavior."""

import asyncio

import httpx
import numpy as np
from sklearn.model_selection import GroupKFold

from real_estate_audit.api import app
from real_estate_audit.constants import (
    FEATURE_COLUMNS,
    N_CV_SPLITS,
    RAW_DIR,
    SOURCE_ID,
    TARGET_COLUMN,
)
from real_estate_audit.data import parse_archive
from real_estate_audit.evaluation import regression_metrics
from real_estate_audit.modeling import MODEL_FEATURES
from real_estate_audit.spatial import assign_spatial_clusters


def test_source_parser_and_feature_scope():
    frame = parse_archive(RAW_DIR / "real-estate.zip")
    assert frame.shape == (414, 8)
    assert frame.columns.tolist() == [SOURCE_ID, *FEATURE_COLUMNS, TARGET_COLUMN]
    assert frame[SOURCE_ID].is_unique
    assert not frame.isna().any().any()
    assert all(SOURCE_ID not in columns for columns in MODEL_FEATURES.values())
    assert all(TARGET_COLUMN not in columns for columns in MODEL_FEATURES.values())


def test_spatial_groups_are_disjoint_across_grouped_folds():
    frame = parse_archive(RAW_DIR / "real-estate.zip")
    clusters = assign_spatial_clusters(frame)
    assert len(np.unique(clusters)) == N_CV_SPLITS
    held_out = []
    for train_index, test_index in GroupKFold(n_splits=N_CV_SPLITS).split(
        frame[FEATURE_COLUMNS], groups=clusters
    ):
        train_groups = set(clusters[train_index])
        test_groups = set(clusters[test_index])
        assert len(test_groups) == 1
        assert not train_groups.intersection(test_groups)
        held_out.extend(test_groups)
    assert sorted(held_out) == sorted(np.unique(clusters).tolist())


def test_regression_metrics_known_values():
    result = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert result["mae"] == 2 / 3
    assert np.isclose(result["rmse"], np.sqrt(4 / 3))
    assert result["median_absolute_error"] == 0.0
    assert np.isclose(result["r2"], -1.0)


def test_api_only_returns_aggregate_artifacts():
    async def request_api():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            assert (await client.get("/health")).json() == {"status": "ok"}
            summary = await client.get("/summary")
            assert summary.json()["rows"] == 414
            models = await client.get("/models")
            assert len(models.json()) == 8
            folds = await client.get("/spatial-folds", params={"model": "ridge"})
            assert folds.status_code == 200
            assert len(folds.json()) == N_CV_SPLITS
            assert all(
                "No" not in row and "prediction" not in row for row in folds.json()
            )
            unknown = await client.get("/spatial-folds", params={"model": "invented"})
            assert unknown.status_code == 404

    asyncio.run(request_api())
