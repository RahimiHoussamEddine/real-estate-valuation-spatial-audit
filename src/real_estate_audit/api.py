"""Read-only API exposing aggregate report artifacts."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from real_estate_audit.constants import REPORTS_DIR

app = FastAPI(
    title="Real Estate Valuation Spatial Audit API",
    version="0.1.0",
    description="Read-only access to aggregate cross-validation results.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    ready = (REPORTS_DIR / "summary.json").is_file()
    return {"status": "ok" if ready else "not_ready"}


@app.get("/summary")
async def summary() -> dict[str, object]:
    path = REPORTS_DIR / "summary.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run `make build` first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/models")
async def models() -> list[dict[str, object]]:
    path = REPORTS_DIR / "model_metrics.csv"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run `make build` first")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/spatial-folds")
async def spatial_folds(
    model: str = Query(..., min_length=1),
) -> list[dict[str, object]]:
    path = REPORTS_DIR / "fold_metrics.csv"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run `make build` first")
    folds = pd.read_csv(path)
    available = sorted(folds.loc[folds["evaluation"] == "spatial", "model"].unique())
    if model not in available:
        raise HTTPException(
            status_code=404,
            detail={"message": "Unknown model", "available": available},
        )
    result = folds[(folds["evaluation"] == "spatial") & (folds["model"] == model)]
    return result.drop(columns="evaluation").to_dict(orient="records")
