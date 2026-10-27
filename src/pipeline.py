"""AI4I 2020 predictive maintenance: reproducible temporal holdout pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Machine failure"
ORDER = "UDI"
NUMERIC = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
CATEGORICAL = ["Type"]
FEATURES = NUMERIC + CATEGORICAL
LEAKAGE = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_uci() -> pd.DataFrame:
    """Fetch the credited UCI dataset; requires network access on first run."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=601)
    frame = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return frame


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use ordered UDI as a time proxy; never shuffle future rows into training."""
    clean = frame.sort_values(ORDER).reset_index(drop=True)
    n = len(clean)
    a, b = int(n * .70), int(n * .85)
    if n < 20 or a == b:
        raise ValueError("Need at least 20 ordered observations")
    return clean.iloc[:a].copy(), clean.iloc[a:b].copy(), clean.iloc[b:].copy()


def build_model() -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("features", prep), ("model", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42))])


def evaluate(model: Pipeline, frame: pd.DataFrame, threshold: float = .5) -> dict[str, Any]:
    y = frame[TARGET].astype(int)
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    predicted = (probability >= threshold).astype(int)
    return {
        "rows": int(len(frame)), "failure_rate": float(y.mean()), "threshold": threshold,
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() == 2 else None,
        "report": classification_report(y, predicted, output_dict=True, zero_division=0),
    }


def train(frame: pd.DataFrame, out_dir: str | Path) -> dict[str, Any]:
    required = {ORDER, TARGET, *FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if set(LEAKAGE) & set(FEATURES):
        raise AssertionError("Failure-mode target columns must not be predictors")
    train_df, valid_df, test_df = temporal_split(frame)
    model = build_model().fit(train_df[FEATURES], train_df[TARGET].astype(int))
    report = {"validation": evaluate(model, valid_df), "test": evaluate(model, test_df),
              "split": {"train_rows": len(train_df), "validation_rows": len(valid_df), "test_rows": len(test_df),
                        "ordering": "UDI ascending; UDI is an ordered-record proxy, not a physical timestamp"}}
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    (output / "feature_schema.json").write_text(json.dumps({"features": FEATURES, "threshold": .5}, indent=2))
    return report


def predict(model: Pipeline, sample: dict[str, Any], threshold: float = .5) -> dict[str, Any]:
    row = pd.DataFrame([{name: sample[name] for name in FEATURES}])
    p = float(model.predict_proba(row)[0, 1])
    return {"failure_probability": round(p, 6), "risk": "high" if p >= threshold else "low", "threshold": threshold}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    metrics = train(load_uci(), args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
