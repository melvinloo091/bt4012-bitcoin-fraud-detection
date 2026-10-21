"""Regenerate Claude's recency candidates (12 Oct 2026) from the supplied competition data.

  submission_lowreg_pair_recent4.csv   - two-model average (less-regularised XGBoost without the six
                                         time-tracking features + LightGBM leaf 5), recency half-life 4
  submission_recent7_lgb_leaf5_50.csv  - LightGBM leaf 5 + LightGBM leaf 50 (ChatGPT's leaf-size blend),
                                         recency half-life 7, no XGBoost
Known labels only (31,235 rows, steps 1-35). Run from the Kaggle project folder:
    python claude_experiments/scripts/make_recency_candidates.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "Data Set"
OUT_DIR = PROJECT_DIR / "claude_experiments" / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED, THREADS = 42, 2
FEATURES = [f"feat_{i}" for i in range(1, 166)]
TIME_TRACKING = ["feat_100", "feat_101", "feat_103", "feat_136", "feat_137", "feat_139"]
NO_TIME = [c for c in FEATURES if c not in TIME_TRACKING]

train = pd.read_csv(DATA_DIR / "Train Data.csv")
test = pd.read_csv(DATA_DIR / "Test Data.csv")
sample = pd.read_csv(DATA_DIR / "Sample Submission" / "BT4012 Competition Submission.csv")
labels = pd.to_numeric(train["label"], errors="coerce")
known = train.loc[labels.notna()].copy()
known["label"] = labels.loc[labels.notna()].astype(int)

def recency_weights(half_life):
    age = known["time_step"].max() - known["time_step"]       # 0 for step 35
    weights = (0.5 ** (age / half_life)).to_numpy()
    return weights / weights.mean()

def lowreg_xgboost():
    return XGBClassifier(n_estimators=200, learning_rate=0.05, tree_method="hist", grow_policy="lossguide",
                         max_depth=0, max_leaves=15, max_bin=127, min_child_weight=1, reg_lambda=1,
                         objective="binary:logistic", n_jobs=THREADS, random_state=SEED)

def lowreg_lightgbm(min_leaf):
    return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63,
                          min_child_samples=min_leaf, min_child_weight=1e-3, reg_lambda=1, objective="binary",
                          deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=THREADS, random_state=SEED)

def save(probabilities, filename):
    submission = sample[["index"]].copy()
    submission["target"] = submission["index"].map(pd.Series(probabilities, index=test["index"]))
    assert submission["index"].equals(sample["index"]) and submission["target"].between(0, 1).all()
    submission.to_csv(OUT_DIR / filename, index=False)
    print("Saved", OUT_DIR / filename)

# 1. Two-model average with stronger recency (half-life 4)
w = recency_weights(4)
xgb = lowreg_xgboost().fit(known[NO_TIME], known["label"], sample_weight=w)
lgb = lowreg_lightgbm(5).fit(known[FEATURES], known["label"], sample_weight=w)
save(0.5 * xgb.predict_proba(test[NO_TIME])[:, 1] + 0.5 * lgb.predict_proba(test[FEATURES])[:, 1],
     "submission_lowreg_pair_recent4.csv")

# 2. LightGBM leaf 5 + leaf 50 with recency half-life 7
w = recency_weights(7)
p5 = lowreg_lightgbm(5).fit(known[FEATURES], known["label"], sample_weight=w).predict_proba(test[FEATURES])[:, 1]
p50 = lowreg_lightgbm(50).fit(known[FEATURES], known["label"], sample_weight=w).predict_proba(test[FEATURES])[:, 1]
save(0.5 * p5 + 0.5 * p50, "submission_recent7_lgb_leaf5_50.csv")
