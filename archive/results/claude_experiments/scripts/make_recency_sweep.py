"""Regenerate Claude's recency-strength candidates (prepared 12 Oct 2026 for 13 Oct).

Same model as the best public submission (submission_lowreg_pair_recent4.csv, 0.95471): equal average of
less-regularised XGBoost (without the six time-tracking features) and less-regularised LightGBM (leaf 5),
known labels only. Only the recency weighting changes:
    weight = 0.5 ** ((35 - time_step) / half_life), rescaled to mean 1.
Usage (from the Kaggle project folder):
    python claude_experiments/scripts/make_recency_sweep.py 3 2 5     # half-lives to build
Writes claude_experiments/submissions/submission_lowreg_pair_recent<h>.csv  (e.g. recent2p5 for 2.5)
"""
import sys
from pathlib import Path
import pandas as pd
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

def build(half_life):
    age = known["time_step"].max() - known["time_step"]
    weights = (0.5 ** (age / half_life)).to_numpy()
    weights = weights / weights.mean()
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, tree_method="hist", grow_policy="lossguide",
                        max_depth=0, max_leaves=15, max_bin=127, min_child_weight=1, reg_lambda=1,
                        objective="binary:logistic", n_jobs=THREADS, random_state=SEED)
    lgb = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5,
                         min_child_weight=1e-3, reg_lambda=1, objective="binary", deterministic=True,
                         force_col_wise=True, verbosity=-1, n_jobs=THREADS, random_state=SEED)
    xgb.fit(known[NO_TIME], known["label"], sample_weight=weights)
    lgb.fit(known[FEATURES], known["label"], sample_weight=weights)
    p = 0.5 * xgb.predict_proba(test[NO_TIME])[:, 1] + 0.5 * lgb.predict_proba(test[FEATURES])[:, 1]
    submission = sample[["index"]].copy()
    submission["target"] = submission["index"].map(pd.Series(p, index=test["index"]))
    assert submission["index"].equals(sample["index"]) and submission["target"].between(0, 1).all()
    name = f"submission_lowreg_pair_recent{str(half_life).replace('.', 'p')}.csv"
    submission.to_csv(OUT_DIR / name, index=False)
    share = weights[known["time_step"].to_numpy() >= 29].sum() / weights.sum()
    print(f"half-life {half_life}: {share:.0%} of training weight on steps 29-35 -> {OUT_DIR / name}")

for h in (sys.argv[1:] or ["3"]):
    build(float(h) if "." in h else int(h))
