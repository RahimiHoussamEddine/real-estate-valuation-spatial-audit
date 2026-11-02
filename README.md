# Real Estate Valuation & Spatial Generalization Audit

An educational regression audit using the UCI Real Estate Valuation benchmark
from Sindian District, New Taipei City, Taiwan. It compares a median baseline,
scaled Ridge regression, a small random forest, and a no-coordinate Ridge
sensitivity under both shuffled and geographically grouped cross-validation.

The target is a historical estimate of unit-area price in 10,000 New Taiwan
dollars per ping. This benchmark is small and local. It is not a current
valuation, a mortgage or investment tool, or a basis for an individual housing
decision.

## Key result

The fixed random forest reached a pooled spatial-CV MAE of **6.194 target
units** (R² 0.569), compared with shuffled-CV MAE 4.822 (R² 0.702). Spatial
fold errors varied across the six held-out groups. These are historical
benchmark results, not current property valuations.

## Run locally

Use Python 3.12:

~~~bash
python3 -m venv .venv
. .venv/bin/activate
make install-dev
make fetch
make build
make verify
make test
make quality
make check-deps
make app
~~~

The geographic analysis uses six coordinate-only spatial clusters and
GroupKFold. Cluster membership is created from latitude and longitude in local
metric coordinates; every reported spatial out-of-fold prediction comes from
a model that did not train on that cluster. The shuffled split is included
only as a comparison and can overstate performance where nearby sales repeat
across folds. No model tuning uses the evaluation folds.

The raw workbook and any local row-level artifacts remain ignored. Checked-in
reports contain fold, model, and coarse cluster aggregates only. Dependency
versions are pinned in requirements.lock.

## API

The API is read-only:

~~~text
GET /health
GET /summary
GET /models
GET /spatial-folds?model=ridge
~~~

Run it with make api. The dashboard runs with make app.

See reports/RESULTS.md, docs/METHODOLOGY.md, docs/DATA_DICTIONARY.md, and
NOTICE.md for results, definitions, source credit, and limitations.
