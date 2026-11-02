"""Cross-validation evaluation with aggregate-only public outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold

from real_estate_audit.constants import (
    COORDINATE_COLUMNS,
    FEATURE_COLUMNS,
    N_CV_SPLITS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from real_estate_audit.modeling import MODEL_FEATURES, make_regressor
from real_estate_audit.spatial import assign_spatial_clusters

MODEL_LABELS = {
    "median": "Median baseline",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "ridge_without_coordinates": "Ridge without coordinates",
}
EVALUATION_LABELS = {
    "spatial": "Spatial grouped CV",
    "shuffled": "Shuffled CV",
}


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def _splits(frame: pd.DataFrame, clusters: np.ndarray, evaluation: str):
    if evaluation == "spatial":
        splitter = GroupKFold(n_splits=N_CV_SPLITS)
        return splitter.split(frame, groups=clusters)
    if evaluation == "shuffled":
        splitter = KFold(
            n_splits=N_CV_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return splitter.split(frame)
    raise ValueError(f"Unknown evaluation mode: {evaluation}")


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return pooled metrics, fold metrics, and spatial cluster aggregates."""
    clusters = assign_spatial_clusters(frame)
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    pooled_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []

    for evaluation in ("spatial", "shuffled"):
        splits = list(_splits(x, clusters, evaluation))
        for model_name, feature_columns in MODEL_FEATURES.items():
            out_of_fold = np.full(len(frame), np.nan, dtype=float)
            for fold_index, (train_index, test_index) in enumerate(splits, start=1):
                estimator = make_regressor(model_name, feature_columns)
                estimator.fit(x.iloc[train_index][feature_columns], y[train_index])
                predictions = estimator.predict(x.iloc[test_index][feature_columns])
                out_of_fold[test_index] = predictions
                row: dict[str, object] = {
                    "evaluation": evaluation,
                    "model": model_name,
                    "fold": fold_index,
                    "n_test": int(len(test_index)),
                    **regression_metrics(y[test_index], predictions),
                }
                if evaluation == "spatial":
                    row["spatial_cluster"] = int(clusters[test_index][0])
                    if len(np.unique(clusters[test_index])) != 1:
                        raise AssertionError(
                            "A spatial fold contains multiple clusters"
                        )
                fold_rows.append(row)
                if evaluation == "spatial":
                    cluster_rows.append(
                        {
                            "model": model_name,
                            "spatial_cluster": int(clusters[test_index][0]),
                            "n_test": int(len(test_index)),
                            **regression_metrics(y[test_index], predictions),
                        }
                    )
            if np.isnan(out_of_fold).any():
                raise AssertionError("Every record must receive one OOF prediction")
            pooled_rows.append(
                {
                    "evaluation": evaluation,
                    "model": model_name,
                    "n": len(frame),
                    **regression_metrics(y, out_of_fold),
                }
            )

    pooled = pd.DataFrame(pooled_rows)
    folds = pd.DataFrame(fold_rows)
    spatial = pd.DataFrame(cluster_rows).sort_values(["model", "spatial_cluster"])
    return pooled, folds, spatial


def _write_results(
    pooled: pd.DataFrame,
    folds: pd.DataFrame,
    quality: dict[str, object],
) -> None:
    spatial = pooled[pooled["evaluation"] == "spatial"].copy()
    shuffled = pooled[pooled["evaluation"] == "shuffled"].copy()
    comparison = spatial[["model", "mae", "rmse", "median_absolute_error", "r2"]].merge(
        shuffled[["model", "mae", "rmse", "median_absolute_error", "r2"]],
        on="model",
        suffixes=("_spatial", "_shuffled"),
    )
    comparison["model_label"] = comparison["model"].map(MODEL_LABELS)
    display = comparison[
        [
            "model_label",
            "mae_spatial",
            "mae_shuffled",
            "rmse_spatial",
            "rmse_shuffled",
            "r2_spatial",
            "r2_shuffled",
        ]
    ].rename(
        columns={
            "model_label": "Model",
            "mae_spatial": "Spatial MAE",
            "mae_shuffled": "Shuffled MAE",
            "rmse_spatial": "Spatial RMSE",
            "rmse_shuffled": "Shuffled RMSE",
            "r2_spatial": "Spatial R²",
            "r2_shuffled": "Shuffled R²",
        }
    )
    display = display.round(3)
    lowest = spatial.loc[spatial["mae"].idxmin()]
    fold_spread = (
        folds[folds["evaluation"] == "spatial"]
        .groupby("model")["mae"]
        .agg(["min", "max"])
        .round(3)
    )
    spatial_fold_sizes = folds[folds["evaluation"] == "spatial"].drop_duplicates(
        "spatial_cluster"
    )["n_test"]
    ridge_spatial = spatial.loc[spatial["model"] == "ridge"].iloc[0]
    ridge_shuffled = shuffled.loc[shuffled["model"] == "ridge"].iloc[0]
    no_coordinate_spatial = spatial.loc[
        spatial["model"] == "ridge_without_coordinates"
    ].iloc[0]
    no_coordinate_shuffled = shuffled.loc[
        shuffled["model"] == "ridge_without_coordinates"
    ].iloc[0]
    lines = [
        "# Results: spatial generalization audit",
        "",
        (
            f"The source table contains **{quality['rows']} records**. "
            "The target is historical unit-area price in "
            f"{quality['target_summary']['unit']}."
        ),
        "",
        "## Pooled out-of-fold metrics",
        "",
        "| "
        + " | ".join(
            [
                "Model",
                "Spatial MAE",
                "Shuffled MAE",
                "Spatial RMSE",
                "Shuffled RMSE",
                "Spatial R²",
                "Shuffled R²",
            ]
        )
        + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in display.itertuples(index=False, name=None)
        ],
        "",
        (
            f"Among the fixed models, **{MODEL_LABELS[str(lowest['model'])]}** "
            "had the lowest pooled spatial MAE "
            f"({lowest['mae']:.3f} target units). This is a cross-validation "
            "comparison on a small historical benchmark, not an estimate of "
            "current home values."
        ),
        "",
        "## Spatial fold variation",
        "",
        (
            "The table below gives the minimum and maximum MAE across the six "
            "held-out spatial clusters. This spread is descriptive; six "
            "clusters do not support a population confidence interval."
        ),
        (
            f"Held-out clusters contain {spatial_fold_sizes.min()} to "
            f"{spatial_fold_sizes.max()} records, so per-cluster errors have "
            "unequal support."
        ),
        "",
        "| Model | Minimum fold MAE | Maximum fold MAE |",
        "|---|---:|---:|",
    ]
    for name, row in fold_spread.iterrows():
        lines.append(f"| {MODEL_LABELS[name]} | {row['min']:.3f} | {row['max']:.3f} |")
    lines.extend(
        [
            "",
            (
                "The shuffled comparison mixes nearby transactions between "
                "training and validation folds and can make geographic "
                "generalization look easier. Ridge spatial MAE was "
                f"{ridge_spatial['mae']:.3f} with coordinates and "
                f"{no_coordinate_spatial['mae']:.3f} without; shuffled MAE "
                f"was {ridge_shuffled['mae']:.3f} with coordinates and "
                f"{no_coordinate_shuffled['mae']:.3f} without. This is a "
                "descriptive sensitivity, not a causal effect."
            ),
            "",
            "## Data and privacy notes",
            "",
            (
                f"There are {quality['duplicate_feature_rows']} repeated "
                f"feature rows among {quality['rows']} observations. Reports "
                "contain only model, fold, and coarse cluster aggregates; "
                "exact coordinates, source IDs, and row-level predictions "
                "are not published."
            ),
            "",
            (
                "See [methodology](../docs/METHODOLOGY.md), "
                "[data dictionary](../docs/DATA_DICTIONARY.md), and "
                "[source notice](../NOTICE.md) for definitions, scope, and "
                "attribution."
            ),
            "",
        ]
    )
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(frame: pd.DataFrame, quality: dict[str, object]) -> dict[str, object]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pooled, folds, spatial = evaluate(frame)
    pooled.to_csv(REPORTS_DIR / "model_metrics.csv", index=False, float_format="%.8f")
    folds.to_csv(REPORTS_DIR / "fold_metrics.csv", index=False, float_format="%.8f")
    spatial.to_csv(
        REPORTS_DIR / "spatial_summary.csv", index=False, float_format="%.8f"
    )
    (REPORTS_DIR / "data_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset": "UCI Real Estate Valuation",
        "rows": len(frame),
        "target": TARGET_COLUMN,
        "target_unit": quality["target_summary"]["unit"],
        "feature_count": len(FEATURE_COLUMNS),
        "source_identifier_used_as_feature": False,
        "coordinate_features": COORDINATE_COLUMNS,
        "spatial_clusters": 6,
        "cv_splits": 6,
        "metrics": pooled.to_dict(orient="records"),
        "privacy": {
            "row_level_predictions_published": False,
            "exact_coordinates_published": False,
            "source_ids_published": False,
        },
    }
    (REPORTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_results(pooled, folds, quality)
    return summary
