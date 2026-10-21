"""Regenerate Claude's two submission files from the supplied competition data.

  claude_experiments/submissions/submission_lowreg_pair_recent7.csv
  claude_experiments/submissions/submission_lowreg_pair_local_features.csv

Both are the Section 28 "two-model average" (less-regularised XGBoost + less-regularised
LightGBM, equal weights), trained on all 31,235 known labels from steps 1-35:
  * recent7        - all features (XGBoost without the six time-tracking features);
                     training rows weighted by recency, half-life 7 time steps
  * local_features - only the transaction's own features feat_1 ... feat_93, no weights
Unknown labels are never used. Run from the Kaggle project folder:
    python claude_experiments/scripts/make_recent7_and_local_features.py
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
LOCAL = FEATURES[:93]
TIME_TRACKING = ["feat_100", "feat_101", "feat_103", "feat_136", "feat_137", "feat_139"]  # Section 24 rule on steps 1-32
NO_TIME = [c for c in FEATURES if c not in TIME_TRACKING]

train = pd.read_csv(DATA_DIR / "Train Data.csv")
test = pd.read_csv(DATA_DIR / "Test Data.csv")
sample = pd.read_csv(DATA_DIR / "Sample Submission" / "BT4012 Competition Submission.csv")
labels = pd.to_numeric(train["label"], errors="coerce")
known = train.loc[labels.notna()].copy()
known["label"] = labels.loc[labels.notna()].astype(int)

def lowreg_xgboost():
    return XGBClassifier(n_estimators=200, learning_rate=0.05, tree_method="hist", grow_policy="lossguide",
                         max_depth=0, max_leaves=15, max_bin=127, min_child_weight=1, reg_lambda=1,
                         objective="binary:logistic", n_jobs=THREADS, random_state=SEED)

def lowreg_lightgbm():
    return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63,
                          min_child_samples=5, min_child_weight=1e-3, reg_lambda=1, objective="binary",
                          deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=THREADS, random_state=SEED)

def two_model_average(xgb_columns, lgb_columns, half_life=None):
    weights = None
    if half_life is not None:
        age = known["time_step"].max() - known["time_step"]          # 0 for step 35, 34 for step 1
        weights = (0.5 ** (age / half_life)).to_numpy()
        weights = weights / weights.mean()                          # average weight stays 1
    xgb = lowreg_xgboost().fit(known[xgb_columns], known["label"], sample_weight=weights)
    lgb = lowreg_lightgbm().fit(known[lgb_columns], known["label"], sample_weight=weights)
    return 0.5 * xgb.predict_proba(test[xgb_columns])[:, 1] + 0.5 * lgb.predict_proba(test[lgb_columns])[:, 1]

def save(probabilities, filename):
    submission = sample[["index"]].copy()
    submission["target"] = submission["index"].map(pd.Series(probabilities, index=test["index"]))
    assert submission["index"].equals(sample["index"]) and submission["target"].between(0, 1).all()
    submission.to_csv(OUT_DIR / filename, index=False)
    print("Saved", OUT_DIR / filename)

save(two_model_average(NO_TIME, FEATURES, half_life=7), "submission_lowreg_pair_recent7.csv")
save(two_model_average(LOCAL, LOCAL), "submission_lowreg_pair_local_features.csv")
