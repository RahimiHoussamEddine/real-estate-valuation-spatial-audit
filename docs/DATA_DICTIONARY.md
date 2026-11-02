# Data dictionary

## UCI source fields

| Source column | Project name | Handling |
|---|---|---|
| No | source_id | Quality checks only; excluded from predictors |
| X1 transaction date | transaction_date | Numeric feature, fractional year |
| X2 house age | house_age_years | Numeric feature |
| X3 distance to the nearest MRT station | distance_to_mrt_meters | Numeric feature |
| X4 number of convenience stores | nearby_store_count | Numeric feature |
| X5 latitude | latitude | Numeric feature and spatial grouping input |
| X6 longitude | longitude | Numeric feature and spatial grouping input |
| Y house price of unit area | price_per_ping_10k_twd | Regression target |

## Checked-in aggregate reports

- reports/model_metrics.csv contains pooled out-of-fold metrics by model and
  split strategy.
- reports/fold_metrics.csv contains metrics by held-out fold; it has no
  property-level records or coordinates.
- reports/spatial_summary.csv contains coarse cluster counts and aggregate
  error metrics.
- reports/summary.json contains analysis settings and pooled metrics; the
  separate data/source_manifest.json records source provenance.
- reports/data_quality.json contains source counts and audit checks.
- reports/RESULTS.md presents the comparison and limitations.

The pipeline does not publish exact coordinates, source IDs, row-level prices,
or row-level predictions.
