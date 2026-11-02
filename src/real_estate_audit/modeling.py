"""Fixed, predeclared regression baselines."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from real_estate_audit.constants import COORDINATE_COLUMNS, FEATURE_COLUMNS

MODEL_FEATURES = {
    "median": FEATURE_COLUMNS,
    "ridge": FEATURE_COLUMNS,
    "random_forest": FEATURE_COLUMNS,
    "ridge_without_coordinates": [
        column for column in FEATURE_COLUMNS if column not in COORDINATE_COLUMNS
    ],
}


def make_regressor(name: str, feature_columns: Sequence[str]):
    if name == "median":
        return DummyRegressor(strategy="median")
    if name in {"ridge", "ridge_without_coordinates"}:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("regressor", Ridge(alpha=10.0)),
            ]
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=4,
            max_features=0.8,
            random_state=2026,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {name}")
