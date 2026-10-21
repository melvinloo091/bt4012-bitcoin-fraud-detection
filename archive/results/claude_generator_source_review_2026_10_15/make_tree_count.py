"""Builds submission_lowreg_pair_recent4_xgb500.csv and (with --exams) xgb_tree_count_validation.csv.

Model: the best model of e1156285.ipynb (half-life 4 recency weights; XGBoost without the six calendar
features + LightGBM 150 trees, averaged 50/50), with XGBoost given 500 trees instead of 300.
The --exams option reruns the 15 practice exams for 200/300/400/500 XGBoost trees (slow: ~15-30 min).
The same 500-tree file is also produced by e1156285.ipynb with FINAL_XGB_TREES = 500.

Usage, from the Kaggle project folder:
    python archive/results/claude_experiments/scripts/make_tree_count.py            # test file only
    python archive/results/claude_experiments/scripts/make_tree_count.py --exams    # plus practice exams
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT = Path.cwd(); DATA = PROJECT / "Data Set"
OUT = PROJECT / "archive" / "results" / "claude_experiments" / "submissions"; OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(DATA / "Train Data.csv"); test = pd.read_csv(DATA / "Test Data.csv")
sample = pd.read_csv(DATA / "Sample Submission" / "BT4012 Competition Submission.csv")
FEATURES = [f"feat_{i}" for i in range(1, 166)]
labels = pd.to_numeric(train["label"], errors="coerce")
known = train.loc[labels.notna()].copy(); known["label"] = labels.loc[labels.notna()].astype(int)

def calendar(rows, thr=0.9):
    c = rows[FEATURES].rank().corrwith(rows["time_step"].rank()); return set(c[c.abs() > thr].index)
def weights(steps, hl):
    w = 0.5 ** ((steps.max() - steps) / hl); return (w / w.mean()).to_numpy()
def xgb(n): return XGBClassifier(n_estimators=n, learning_rate=0.05, tree_method="hist", grow_policy="lossguide", max_depth=0,
                                  max_leaves=15, max_bin=127, min_child_weight=1, reg_lambda=1, objective="binary:logistic", n_jobs=2, random_state=42)
def lgb(): return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5, min_child_weight=1e-3,
                                 reg_lambda=1, objective="binary", deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=2, random_state=42)
TREES = [200, 300, 400, 500]

def fit_predict(rows, target, hl=4):
    # Fit XGBoost once with 500 trees and read the 200/300/400-tree versions from its first trees (identical models)
    w = weights(rows.time_step, hl); xf = [c for c in FEATURES if c not in calendar(rows)]
    pl = lgb().fit(rows[FEATURES], rows.label, sample_weight=w).predict_proba(target[FEATURES])[:, 1]
    m = xgb(max(TREES)).fit(rows[xf], rows.label, sample_weight=w)
    return {n: 0.5 * m.predict_proba(target[xf], iteration_range=(0, n))[:, 1] + 0.5 * pl for n in TREES}

if "--exams" in sys.argv:
    rows = []
    for k in range(7, 22):
        L = known[known.time_step <= k]; V = known[known.time_step.between(k + 1, k + 14)]
        p = fit_predict(L, V)
        rows.append({"exam": f"{k+1}-{k+14}", **{f"xgb{n}": roc_auc_score(V.label, p[n]) for n in TREES}}); print(rows[-1], flush=True)
    res = pd.DataFrame(rows).set_index("exam"); res.to_csv(OUT / "xgb_tree_count_validation.csv")
    print(res.mean().round(5))

p = fit_predict(known, test)[500]
sub = sample[["index"]].copy(); sub["target"] = sub["index"].map(pd.Series(p, index=test["index"]))
assert sub["index"].equals(sample["index"]) and sub.target.notna().all() and sub.target.between(0, 1).all()
sub.to_csv(OUT / "submission_lowreg_pair_recent4_xgb500.csv", index=False)
print("Saved", OUT / "submission_lowreg_pair_recent4_xgb500.csv")
