"""Single training-only feature-removal hypothesis; no test data is read."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
FEATURES = [f"feat_{i}" for i in range(1, 166)]
known = pd.read_csv(
    Path(__file__).resolve().parents[2] / "Data Set" / "Train Data.csv"
).dropna(subset=["label"]).copy()
known["label"] = known["label"].astype(int)


def make_model(rounds, *, stopping=False):
    return XGBClassifier(
        n_estimators=int(rounds), learning_rate=0.05,
        tree_method="hist", grow_policy="lossguide", max_depth=0,
        max_leaves=15, max_bin=127, min_child_weight=3, reg_lambda=10,
        subsample=1.0, colsample_bytree=1.0, scale_pos_weight=1.0,
        objective="binary:logistic", eval_metric="auc",
        early_stopping_rounds=60 if stopping else None,
        n_jobs=2, random_state=42,
    )


def training_selection(learning):
    end = int(learning["time_step"].max())
    inner = learning.loc[learning["time_step"] <= end - 3]
    correlations = inner[FEATURES].corrwith(inner["time_step"], method="spearman")
    removed = correlations.index[correlations.abs() > 0.9].tolist()
    return [column for column in FEATURES if column not in removed], removed, correlations


def fit(learning, columns):
    end = int(learning["time_step"].max())
    fitting = learning.loc[learning["time_step"] <= end - 3]
    stopping = learning.loc[learning["time_step"] > end - 3]
    assert fitting["time_step"].max() < stopping["time_step"].min()
    with threadpool_limits(limits=2):
        initial = make_model(400, stopping=True)
        initial.fit(
            fitting[columns], fitting["label"],
            eval_set=[(stopping[columns], stopping["label"])], verbose=False,
        )
        rounds = int(initial.best_iteration + 1)
        fitted = make_model(rounds)
        fitted.fit(learning[columns], learning["label"])
    return fitted, rounds


def main():
    records, predictions, selection_records = [], {}, {}
    for end, start, stop in [(14, 15, 28), (21, 22, 35), (28, 29, 35)]:
        learning = known.loc[known["time_step"] <= end]
        checking = known.loc[known["time_step"].between(start, stop)]
        columns, removed, correlations = training_selection(learning)
        selection_records[str(end)] = {
            "selection_training_steps": f"1-{end - 3}",
            "removed_features": removed,
            "retained_features": columns,
            "spearman_correlations": correlations.to_dict(),
        }
        for name, selected in [("current_xgboost", FEATURES), ("drop_time_proxies", columns)]:
            started = time.perf_counter()
            fitted, rounds = fit(learning, selected)
            with threadpool_limits(limits=2):
                scores = fitted.predict_proba(checking[selected])[:, 1]
            predictions[f"{name}_{end}"] = scores
            predictions[f"labels_{end}"] = checking["label"].to_numpy()
            predictions[f"time_steps_{end}"] = checking["time_step"].to_numpy()
            record = {
                "model": name, "train_end": end,
                "validation_start": start, "validation_end": stop,
                "feature_count": len(selected), "rounds": rounds,
                "auc": float(roc_auc_score(checking["label"], scores)),
                "seconds": round(time.perf_counter() - started, 2),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            pd.DataFrame(records).to_csv(HERE / "results.csv", index=False)
            np.savez_compressed(HERE / "predictions.npz", **predictions)

    columns, removed, correlations = training_selection(known)
    selection_records["35_final"] = {
        "selection_training_steps": "1-32",
        "removed_features": removed,
        "retained_features": columns,
        "spearman_correlations": correlations.to_dict(),
    }
    (HERE / "feature_selections.json").write_text(json.dumps(selection_records, indent=2))
    print("Final all-known-data feature selection:", removed, flush=True)
    print(pd.DataFrame(records).pivot(index="model", columns="train_end", values="auc").to_string())


if __name__ == "__main__":
    main()
