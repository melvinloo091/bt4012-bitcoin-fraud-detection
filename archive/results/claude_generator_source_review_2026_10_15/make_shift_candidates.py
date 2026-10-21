"""Builds two candidates aimed at the post-shutdown test period, from the best model of e1156285.ipynb
(half-life 4; XGBoost 300 trees without the six calendar features + LightGBM 150 trees; 50/50):

  submission_half4_xgb300_drift_drop.csv    (A) drop every feature whose values barely overlap between the
        last 7 training steps (29-35) and the test rows: drift = |AUC - 0.5| * 2 of telling the two groups apart
        with that one feature, dropped if > 0.3. No labels are used. Dropped from both models.
        For the test set this drops 15 features: the 6 calendar features plus 2, 3, 106, 107, 109, 113, 115, 149, 151.
  submission_half4_xgb300_recent7_blend.csv (B) average the best model 50/50 with the same pair trained only on
        steps 29-35, without recency weights. (Rejected: much worse on the practice exams.)

--exams reruns the 15 practice exams (drift measured between each exam's last 7 training steps and its exam rows)
and writes shift_candidates_validation.csv (slow: ~30 min).

Usage, from the Kaggle project folder:
    python archive/results/claude_experiments/scripts/make_shift_candidates.py            # test files only
    python archive/results/claude_experiments/scripts/make_shift_candidates.py --exams    # plus practice exams
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
DRIFT_LIMIT = 0.3

def calendar(rows, thr=0.9):
    c = rows[FEATURES].rank().corrwith(rows["time_step"].rank()); return set(c[c.abs() > thr].index)
def drifting(recent_rows, target_rows, limit=DRIFT_LIMIT):
    y = np.r_[np.zeros(len(recent_rows)), np.ones(len(target_rows))]
    return {f for f in FEATURES
            if abs(roc_auc_score(y, np.r_[recent_rows[f].to_numpy(), target_rows[f].to_numpy()]) - 0.5) * 2 > limit}
def weights(steps, hl):
    if hl is None: return np.ones(len(steps))
    w = 0.5 ** ((steps.max() - steps) / hl); return (w / w.mean()).to_numpy()
def xgb(): return XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method="hist", grow_policy="lossguide", max_depth=0,
                                max_leaves=15, max_bin=127, min_child_weight=1, reg_lambda=1, objective="binary:logistic", n_jobs=2, random_state=42)
def lgb(): return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5, min_child_weight=1e-3,
                                 reg_lambda=1, objective="binary", deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=2, random_state=42)
def pair(rows, target, hl=4, drop=frozenset()):
    w = weights(rows.time_step, hl)
    xf = [c for c in FEATURES if c not in calendar(rows) and c not in drop]
    lf = [c for c in FEATURES if c not in drop]
    px = xgb().fit(rows[xf], rows.label, sample_weight=w).predict_proba(target[xf])[:, 1]
    pl = lgb().fit(rows[lf], rows.label, sample_weight=w).predict_proba(target[lf])[:, 1]
    return 0.5 * px + 0.5 * pl

if "--exams" in sys.argv:
    rows = []
    for k in range(7, 22):
        L = known[known.time_step <= k]; V = known[known.time_step.between(k + 1, k + 14)]
        recentL = L[L.time_step > k - 7]; drop = drifting(recentL, V)
        best = pair(L, V)
        rows.append({"exam": f"{k+1}-{k+14}", "best": roc_auc_score(V.label, best),
                     "A_drift_drop": roc_auc_score(V.label, pair(L, V, drop=drop)),
                     "B_recent_blend": roc_auc_score(V.label, 0.5 * best + 0.5 * pair(recentL, V, hl=None)),
                     "n_dropped": len(drop)}); print(rows[-1], flush=True)
    res = pd.DataFrame(rows).set_index("exam"); res.to_csv(OUT / "shift_candidates_validation.csv"); print(res.mean().round(5))

recent = known[known.time_step >= 29]
drop = drifting(recent, test); print("Dropped for the test set:", sorted(drop, key=lambda c: int(c.split("_")[1])))
best = pair(known, test)
outputs = {"submission_half4_xgb300_drift_drop.csv": pair(known, test, drop=drop),
           "submission_half4_xgb300_recent7_blend.csv": 0.5 * best + 0.5 * pair(recent, test, hl=None)}
for name, p in outputs.items():
    sub = sample[["index"]].copy(); sub["target"] = sub["index"].map(pd.Series(p, index=test["index"]))
    assert sub["index"].equals(sample["index"]) and sub.target.notna().all() and sub.target.between(0, 1).all()
    sub.to_csv(OUT / name, index=False); print("Saved", OUT / name)
