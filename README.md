# Multiclass Metrics

`multiclass_metrics` provides scikit-learn-style metrics for probabilistic
binary and multiclass classifiers. Its main purpose is to make ROC-AUC and
area under the precision-recall curve usable when the classes present in
`y_true` and the columns returned by `predict_proba` do not line up exactly.

## What It Provides

The package exposes two public metric functions:

- `multiclass_metrics.roc_auc_score(...)`
- `multiclass_metrics.auprc(...)`

Both accept `y_true`, `y_score`, optional `labels`, `average`, `sample_weight`,
and `multi_class` arguments in the same general shape as scikit-learn metrics.
They support binary inputs and multiclass inputs, with one-vs-one multiclass
averaging by default.

## Why It Exists

Scikit-learn's probabilistic multiclass metrics expect the score matrix and
label set to agree. That can be awkward when, for example:

- a classifier never learned to predict a class that appears in the test set
- a score matrix includes classes that are absent from `y_true`
- score columns are not naturally inferable from `np.unique(y_true)`

This package aligns those cases before scoring, so evaluation can proceed
without manually rebuilding the probability matrix.

## How It Works

When you pass `labels` explicitly and its length does not match the number of
unique values in `y_true`, the shared multiclass wrapper aligns the score
matrix before calling the underlying scikit-learn metric logic:

- if `y_true` contains a class missing from `labels`/`y_score`, it inserts a
  score column filled with `0.0`
- if `labels`/`y_score` contains a class absent from `y_true`, it removes that
  score column
- after class alignment, labels are sorted and score columns are reordered to
  match
- binary inputs fall back to the standard binary scikit-learn metric behavior
- multiclass inputs use `multi_class="ovo"` by default and also accept
  `multi_class="ovr"`

Unlike scikit-learn's multiclass ROC-AUC entry point, `roc_auc_score` here does
not require each row of `y_score` to sum to 1.

## Installation

```bash
pip install multiclass_metrics
```

The package requires Python 3.8 or newer and depends on NumPy, pandas, and
scikit-learn.

## Usage

```python
import numpy as np
import multiclass_metrics

y_true = ["Healthy", "Ebola", "HIV", "Healthy", "Covid"]

# Columns are ordered as: Covid, HIV, Healthy.
# There is no Ebola column, so multiclass_metrics inserts it with 0.0 scores.
y_score = np.array(
    [
        [0.10, 0.10, 0.80],
        [0.33, 0.33, 0.34],
        [0.10, 0.80, 0.10],
        [0.05, 0.05, 0.90],
        [0.80, 0.10, 0.10],
    ]
)

auc = multiclass_metrics.roc_auc_score(
    y_true=y_true,
    y_score=y_score,
    labels=["Covid", "HIV", "Healthy"],
    multi_class="ovo",
    average="macro",
)

pr_auc = multiclass_metrics.auprc(
    y_true=y_true,
    y_score=y_score,
    labels=["Covid", "HIV", "Healthy"],
)
```

When labels are provided, `y_score` columns must be in the same order as
`labels`. If labels are omitted, they are inferred from `np.unique(y_true)`;
for multiclass problems where `y_score` has columns that cannot be inferred
from `y_true`, pass `labels` explicitly.

## Important Behavior and Limitations

- `average` is intended for `"macro"` or `"weighted"` averaging.
- `multi_class` must be `"ovo"` or `"ovr"`.
- `sample_weight` is passed through for binary and OvR scoring, but is ignored
  for OvO multiclass scoring.
- In binary mode, a 1D score vector is accepted; with two labels, the second
  label is treated as the positive class for `auprc`.
- If class alignment leaves only one class in `y_true`, probability-based
  scoring is undefined and the function raises `ValueError`.

## Development

```bash
pip install -r requirements_dev.txt
pip install -e .
make test
make lint
make docs
```

The repository includes pytest coverage for missing classes in `y_score`,
missing classes in `y_true`, binary inputs, label ordering, non-normalized score
rows, and auPRC behavior.
