# Model card

## Intended use
Educational failure-risk ranking for maintenance exploration on the UCI AI4I 2020 synthetic industrial dataset. It is not a safety controller or a replacement for a qualified maintenance process.

## Data and target
UCI dataset 601, 10,000 rows. Predict `Machine failure` from five sensor values and product type. Failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded to prevent target leakage.

## Method
Median imputation and standard scaling for numeric signals; most-frequent imputation and one-hot encoding for product type; class-weighted logistic regression. Rows are split by ascending UDI (70/15/15) as an ordered-record proxy. UDI is not a true timestamp, so this is not proof of temporal generalization.

## Evaluation
`metrics.json` records holdout average precision, ROC AUC, and thresholded class metrics after a reproducible run. The threshold is 0.5 by default and should be selected against maintenance costs before operational use.

## Risks and limits
The dataset is synthetic, small, and not representative of a particular plant. Probabilities are not calibrated for real machinery, and no field deployment or operational outcome is claimed. Validate sensor ranges, calibration, drift, and consequences with domain experts first.
