# Methodology

## Scope and intended use

This project studies out-of-sample regression error on a 414-row historical
dataset from one district in New Taipei City. The target is the source's
unit-area price in 10,000 New Taiwan dollars per ping (one ping is 3.3 square
meters). The reported value is a dataset label, not an estimate of current
market value.

Do not use the models for a home purchase, sale, mortgage, lending, taxation,
insurance, or an individual valuation. The source is small, historical, and
geographically narrow.

## Source and audit

- Download the UCI archive and validate its byte count and SHA-256.
- Read the workbook's main table; retain the source row identifier only for
  data-quality checks, never as a model feature.
- Check row counts, target validity, missingness, duplicate IDs, duplicate
  feature rows, and coordinate ranges.
- Keep the downloaded workbook and any row-level intermediates ignored.

## Spatial folds

The main evaluation uses six spatial groups created by KMeans on local
metric-coordinate approximations derived from latitude and longitude only.
The conversion accounts for longitude's latitude-dependent distance using the
mean latitude of this small study area. No target values enter cluster
construction.

GroupKFold holds one complete spatial cluster out at a time. Each row receives
one out-of-fold prediction from a model that did not train on its cluster.
We report pooled out-of-fold metrics and metrics for each held-out cluster.
With only six clusters, fold-to-fold spread is descriptive and is not a
population confidence interval. Cluster sizes are uneven, so each per-cluster
metric has different support; pooled metrics weight every observation equally.

A shuffled six-fold cross-validation is included as a sensitivity. Nearby
transactions can land in both training and validation folds, so shuffled
results are not the primary generalization estimate.

## Models and metrics

The fixed model set is:

1. Median-only DummyRegressor baseline.
2. Standardized Ridge regression with alpha 10.
3. A bounded RandomForestRegressor with 300 trees, maximum depth 5, and
   minimum leaf size 4.
4. Standardized Ridge regression without latitude and longitude, as a location
   dependence sensitivity.

No hyperparameters are tuned on evaluation folds. Metrics are mean absolute
error (MAE), root mean squared error (RMSE), median absolute error, and
coefficient of determination (R-squared). They are computed on pooled
out-of-fold predictions and separately for each held-out spatial cluster.

## Limits

- There are only 414 observations from one district and a short historical
  window.
- The source describes its original split as random; this project creates a
  separate spatial cross-validation audit rather than reusing it.
- Coordinates are used as model features in the primary models and to create
  spatial groups. A no-coordinate Ridge sensitivity helps show dependence on
  location.
- Fold spread across six geographic clusters is not a confidence interval.
- Cluster size varies, and the individual fold metrics should be interpreted
  alongside their test-row counts.
- Results do not establish current market performance or individual
  valuation accuracy.
- Public reports suppress precise property coordinates and row-level prices.
