"""Regenerate claude_experiments/submissions/submission_recent7_local_features.csv

Question it answers: with everything else identical to the best public model (0.94989: seven-step
recency weighting, less-regularised XGBoost + LightGBM leaf 5, equal average), does removing the
neighbour-summary features (feat_94-feat_165) change the public score? Only feat_1-feat_93 are used.
Known labels only. Run from the Kaggle project folder:
    python claude_experiments/scripts/make_recent7_local_features.py
"""
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "Data Set"
OUT_DIR = PROJECT_DIR / "claude_experiments" / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED, THREADS, HALF_LIFE = 42, 2, 7
LOCAL = [f"feat_{i}" for i in range(1, 94)]   # transaction's own features; none of the six time-tracking features are in this range

train = pd.read_csv(DATA_DIR / "Train Data.csv")
test = pd.read_csv(DATA_DIR / "Test Data.csv")
sample = pd.read_csv(DATA_DIR / "Sample Submission" / "BT4012 Competition Submission.csv")
labels = pd.to_numeric(train["label"], errors="coerce")
known = train.loc[labels.notna()].copy()
known["label"] = labels.loc[labels.notna()].astype(int)

age = known["time_step"].max() - known["time_step"]
weights = (0.5 ** (age / HALF_LIFE)).to_numpy()
weights = weights / weights.mean()

xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, tree_method="hist", grow_policy="lossguide",
                    max_depth=0, max_leaves=15, max_bin=127, min_child_weight=1, reg_lambda=1,
                    objective="binary:logistic", n_jobs=THREADS, random_state=SEED)
lgb = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5,
                     min_child_weight=1e-3, reg_lambda=1, objective="binary", deterministic=True,
                     force_col_wise=True, verbosity=-1, n_jobs=THREADS, random_state=SEED)
xgb.fit(known[LOCAL], known["label"], sample_weight=weights)
lgb.fit(known[LOCAL], known["label"], sample_weight=weights)
probabilities = 0.5 * xgb.predict_proba(test[LOCAL])[:, 1] + 0.5 * lgb.predict_proba(test[LOCAL])[:, 1]

submission = sample[["index"]].copy()
submission["target"] = submission["index"].map(pd.Series(probabilities, index=test["index"]))
assert submission["index"].equals(sample["index"]) and submission["target"].between(0, 1).all()
submission.to_csv(OUT_DIR / "submission_recent7_local_features.csv", index=False)
print("Saved", OUT_DIR / "submission_recent7_local_features.csv")
