# Results: spatial generalization audit

The source table contains **414 records**. The target is historical unit-area price in 10,000 New Taiwan dollars per ping.

## Pooled out-of-fold metrics

| Model | Spatial MAE | Shuffled MAE | Spatial RMSE | Shuffled RMSE | Spatial R² | Shuffled R² |
|---|---:|---:|---:|---:|---:|---:|
| Median baseline | 14.002 | 10.808 | 16.884 | 13.702 | -0.544 | -0.017 |
| Ridge regression | 9.159 | 6.287 | 12.239 | 8.963 | 0.189 | 0.565 |
| Random forest | 6.194 | 4.822 | 8.921 | 7.422 | 0.569 | 0.702 |
| Ridge without coordinates | 8.666 | 6.502 | 11.589 | 9.187 | 0.273 | 0.543 |

Among the fixed models, **Random forest** had the lowest pooled spatial MAE (6.194 target units). This is a cross-validation comparison on a small historical benchmark, not an estimate of current home values.

## Spatial fold variation

The table below gives the minimum and maximum MAE across the six held-out spatial clusters. This spread is descriptive; six clusters do not support a population confidence interval.
Held-out clusters contain 23 to 173 records, so per-cluster errors have unequal support.

| Model | Minimum fold MAE | Maximum fold MAE |
|---|---:|---:|
| Median baseline | 5.270 | 21.869 |
| Random forest | 3.923 | 8.242 |
| Ridge regression | 6.038 | 21.930 |
| Ridge without coordinates | 6.416 | 16.227 |

The shuffled comparison mixes nearby transactions between training and validation folds and can make geographic generalization look easier. Ridge spatial MAE was 9.159 with coordinates and 8.666 without; shuffled MAE was 6.287 with coordinates and 6.502 without. This is a descriptive sensitivity, not a causal effect.

## Data and privacy notes

There are 20 repeated feature rows among 414 observations. Reports contain only model, fold, and coarse cluster aggregates; exact coordinates, source IDs, and row-level predictions are not published.

See [methodology](../docs/METHODOLOGY.md), [data dictionary](../docs/DATA_DICTIONARY.md), and [source notice](../NOTICE.md) for definitions, scope, and attribution.
