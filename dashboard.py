"""Aggregate-only Streamlit dashboard."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from real_estate_audit.constants import REPORTS_DIR
from real_estate_audit.evaluation import EVALUATION_LABELS, MODEL_LABELS

st.set_page_config(page_title="Spatial valuation audit", layout="wide")
st.title("Real estate valuation: spatial generalization audit")
st.caption(
    "Historical benchmark analysis only. These results are not current home values "
    "or individual valuations."
)

summary_path = REPORTS_DIR / "summary.json"
if not summary_path.exists():
    st.info("Build the report first with `make build`.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
folds = pd.read_csv(REPORTS_DIR / "fold_metrics.csv")
quality = json.loads((REPORTS_DIR / "data_quality.json").read_text(encoding="utf-8"))

first, second, third = st.columns(3)
first.metric("Source observations", f"{summary['rows']:,}")
second.metric("Features", summary["feature_count"])
third.metric("Spatial folds", summary["spatial_clusters"])

metric_labels = {
    "mae": "Mean absolute error",
    "rmse": "Root mean squared error",
    "median_absolute_error": "Median absolute error",
    "r2": "R-squared",
}
metric = st.selectbox("Metric", list(metric_labels), format_func=metric_labels.get)
plot = metrics.copy()
plot["model_label"] = plot["model"].map(MODEL_LABELS)
plot["evaluation_label"] = plot["evaluation"].map(EVALUATION_LABELS)
figure = px.bar(
    plot,
    x="model_label",
    y=metric,
    color="evaluation_label",
    barmode="group",
    labels={
        "model_label": "Model",
        metric: metric_labels[metric],
        "evaluation_label": "Evaluation",
    },
    title="Pooled out-of-fold comparison",
)
st.plotly_chart(figure, width="stretch")

selected_model = st.selectbox(
    "Spatial fold detail",
    list(MODEL_LABELS),
    format_func=MODEL_LABELS.get,
)
spatial_folds = folds[
    (folds["evaluation"] == "spatial") & (folds["model"] == selected_model)
].sort_values("spatial_cluster")
fold_figure = px.bar(
    spatial_folds,
    x="spatial_cluster",
    y="mae",
    labels={"spatial_cluster": "Coarse cluster", "mae": "MAE (target units)"},
    title="MAE by held-out spatial cluster",
)
st.plotly_chart(fold_figure, width="stretch")

st.subheader("Data quality summary")
st.dataframe(
    pd.DataFrame(
        [
            {"Check": "Missing cells", "Value": quality["missing_cells"]},
            {
                "Check": "Repeated feature rows",
                "Value": quality["duplicate_feature_rows"],
            },
            {
                "Check": "Unique feature rows",
                "Value": quality["unique_feature_rows"],
            },
            {
                "Check": "Duplicate source identifiers",
                "Value": quality["duplicate_source_ids"],
            },
        ]
    ),
    hide_index=True,
    width="stretch",
)
st.markdown(
    "Spatial fold variation is descriptive because the benchmark has only six groups. "
    "No source identifiers, exact coordinates, or row-level predictions are shown."
)
