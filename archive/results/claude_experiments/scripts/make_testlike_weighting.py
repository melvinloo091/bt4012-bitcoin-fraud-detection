"""Regenerate claude_experiments/submissions/submission_testlike_weighting_mild.csv

Hypothesis tested (12 Oct 2026): seven-step recency weighting helped on Kaggle (0.94989) because it emphasised
training transactions that resemble the test period. Here each training transaction is instead weighted by how
test-like its features look: a classifier learns to tell known-label training rows from test rows (features only,
the six time-tracking features excluded, out-of-fold predictions), and weight = (test odds)^0.25, clipped to
[0.2, 5], mean 1. Model otherwise identical to the 0.94989 model (no date-based recency). Known labels only.
Run from the Kaggle project folder:  python claude_experiments/scripts/make_testlike_weighting.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "Data Set"
OUT_DIR = PROJECT_DIR / "claude_experiments" / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED, THREADS = 42, 2
FEATURES = [f"feat_{i}" for i in range(1, 166)]
NO_TIME = [c for c in FEATURES if c not in ["feat_100", "feat_101", "feat_103", "feat_136", "feat_137", "feat_139"]]

train = pd.read_csv(DATA_DIR / "Train Data.csv")
test = pd.read_csv(DATA_DIR / "Test Data.csv")
sample = pd.read_csv(DATA_DIR / "Sample Submission" / "BT4012 Competition Submission.csv")
labels = pd.to_numeric(train["label"], errors="coerce")
known = train.loc[labels.notna()].copy()
known["label"] = labels.loc[labels.notna()].astype(int)

# 1. How test-like is each training row? (no labels used)
X = pd.concat([known[NO_TIME], test[NO_TIME]]).to_numpy()
is_test = np.r_[np.zeros(len(known)), np.ones(len(test))]
p = np.zeros(len(X))
for fit_rows, score_rows in StratifiedKFold(3, shuffle=True, random_state=0).split(X, is_test):
    judge = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, min_child_samples=50,
                           verbosity=-1, n_jobs=THREADS, random_state=0).fit(X[fit_rows], is_test[fit_rows])
    p[score_rows] = judge.predict_proba(X[score_rows])[:, 1]
p_train = np.clip(p[:len(known)], 1e-4, 1 - 1e-4)
odds = p_train / (1 - p_train) * len(known) / len(test)
weights = np.clip(odds ** 0.25, 0.2, 5)
weights = weights / weights.mean()

# 2. Same two-model average as the 0.94989 model, with these weights instead of recency
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, tree_method="hist", grow_policy="lossguide", max_depth=0,
                    max_leaves=15, max_bin=127, min_child_weight=1, reg_lambda=1, objective="binary:logistic",
                    n_jobs=THREADS, random_state=SEED).fit(known[NO_TIME], known["label"], sample_weight=weights)
lgb = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5,
                     min_child_weight=1e-3, reg_lambda=1, objective="binary", deterministic=True, force_col_wise=True,
                     verbosity=-1, n_jobs=THREADS, random_state=SEED).fit(known[FEATURES], known["label"], sample_weight=weights)
probabilities = 0.5 * xgb.predict_proba(test[NO_TIME])[:, 1] + 0.5 * lgb.predict_proba(test[FEATURES])[:, 1]

submission = sample[["index"]].copy()
submission["target"] = submission["index"].map(pd.Series(probabilities, index=test["index"]))
assert submission["index"].equals(sample["index"]) and submission["target"].between(0, 1).all()
submission.to_csv(OUT_DIR / "submission_testlike_weighting_mild.csv", index=False)
print("Saved", OUT_DIR / "submission_testlike_weighting_mild.csv")
