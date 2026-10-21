"""Covariate-shift ("similarity") weighting instead of / on top of date-based recency.

Step 1. A 'period detector' (LightGBM) learns to tell training rows apart from the rows being predicted,
        using every feature except the 6 calendar features (no labels involved). Training rows get
        out-of-fold scores p = P(looks like the target period).
Step 2. Weight = (p / (1 - p)) ** gamma, capped at the 99th percentile, rescaled to mean 1.
        gamma is chosen so the effective number of training rows equals half-life 4's (a fair 'data budget').
Candidates (best model otherwise unchanged: XGB 300 trees without calendar features + LGB 150 trees, 50/50):
  S  similarity weights only (no date-based recency)
  SH similarity x half-life 8 recency, again matched to half-life 4's effective rows
--exams runs the 15 practice exams (detector trained on each exam's own training rows vs its exam rows; slow, ~1 hour)
and writes similarity_weighting_validation.csv.

Outputs (archive/results/claude_experiments/submissions/):
  submission_similarity_weighting.csv   (S)
  submission_similarity_x_recent8.csv   (SH)
Usage, from the Kaggle project folder:
    python archive/results/claude_experiments/scripts/make_similarity_weighting.py            # test files only (~10 min)
    python archive/results/claude_experiments/scripts/make_similarity_weighting.py --exams    # plus practice exams"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT = Path.cwd(); DATA = PROJECT / "Data Set"
OUT = PROJECT / "archive" / "results" / "claude_experiments" / "submissions"; OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(DATA / "Train Data.csv"); test = pd.read_csv(DATA / "Test Data.csv")
sample = pd.read_csv(DATA / "Sample Submission" / "BT4012 Competition Submission.csv")
FEATURES = [f'feat_{i}' for i in range(1, 166)]
labels = pd.to_numeric(train['label'], errors='coerce')
known = train.loc[labels.notna()].copy(); known['label'] = labels.loc[labels.notna()].astype(int)

def calendar(rows, thr=0.9):
    c = rows[FEATURES].rank().corrwith(rows['time_step'].rank()); return set(c[c.abs() > thr].index)
def recency(steps, hl):
    w = 0.5 ** ((steps.max() - steps) / hl); return (w / w.mean()).to_numpy()
def ess(w): return w.sum() ** 2 / (w ** 2).sum()
def xgb(): return XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method='hist', grow_policy='lossguide', max_depth=0,
                                max_leaves=15, max_bin=127, min_child_weight=1, reg_lambda=1, objective='binary:logistic', n_jobs=2, random_state=42)
def lgb(): return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5, min_child_weight=1e-3,
                                 reg_lambda=1, objective='binary', deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=2, random_state=42)
def detector(): return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, verbosity=-1, n_jobs=2, random_state=42)

def similarity_ratio(rows, target):
    cols = [c for c in FEATURES if c not in calendar(rows)]
    X = pd.concat([rows[cols], target[cols]]); y = np.r_[np.zeros(len(rows)), np.ones(len(target))]
    p = np.zeros(len(rows))
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = detector().fit(X.iloc[tr_i], y[tr_i])
        idx = te_i[te_i < len(rows)]
        p[idx] = m.predict_proba(X.iloc[idx])[:, 1]
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return p / (1 - p)

def temper(ratio, base, target_ess):
    # find gamma so that ess(base * ratio**gamma) ~= target_ess (bisection), cap at 99th percentile
    def make(g):
        w = base * ratio ** g; w = np.minimum(w, np.quantile(w, 0.99)); return w / w.mean()
    lo, hi = 0.0, 3.0
    if ess(make(hi)) > target_ess: return make(hi), hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if ess(make(mid)) > target_ess: lo = mid
        else: hi = mid
    return make(hi), hi

def pair(rows, target, w):
    xf = [c for c in FEATURES if c not in calendar(rows)]
    px = xgb().fit(rows[xf], rows.label, sample_weight=w).predict_proba(target[xf])[:, 1]
    pl = lgb().fit(rows[FEATURES], rows.label, sample_weight=w).predict_proba(target[FEATURES])[:, 1]
    return 0.5 * px + 0.5 * pl

def candidates(rows, target):
    w4 = recency(rows.time_step, 4); budget = ess(w4)
    ratio = similarity_ratio(rows, target)
    wS, gS = temper(ratio, np.ones(len(rows)), budget)
    wSH, gSH = temper(ratio, recency(rows.time_step, 8), budget)
    return {'best': w4, 'S': wS, 'SH': wSH}, {'gamma_S': gS, 'gamma_SH': gSH, 'ess_budget': budget}

t0 = time.time()
W, info = candidates(known, test)
print('test weights:', {k: round(v, 3) for k, v in info.items()}, flush=True)
share = pd.DataFrame({k: pd.Series(w, index=known.time_step.to_numpy()).groupby(level=0).sum() / w.sum() for k, w in W.items()})
print('weight share by training step (last 12 steps):'); print((100 * share).round(1).tail(12).T.to_string())
print('weight share on steps 29-35:', {k: round(float(100 * share.loc[29:, k].sum()), 1) for k in W})
illicit = known.label.to_numpy() == 1
print('effective illicit rows:', {k: round(ess(w[illicit])) for k, w in W.items()})
preds = {k: pair(known, test, w) for k, w in W.items()}
for k, name in [('S', 'submission_similarity_weighting.csv'), ('SH', 'submission_similarity_x_recent8.csv')]:
    sub = sample[['index']].copy(); sub['target'] = sub['index'].map(pd.Series(preds[k], index=test['index']))
    assert sub['index'].equals(sample['index']) and sub.target.notna().all() and sub.target.between(0, 1).all()
    sub.to_csv(OUT / name, index=False); print('Saved', OUT / name)
steps = test.time_step.to_numpy(); top = 920; rb = pd.Series(preds['best']).rank(ascending=False)
for k, p in preds.items():
    r = pd.Series(p).rank(ascending=False)
    print(f"{k:5s} corr with best {r.corr(rb):.4f} | top-6% overlap {((r<=top)&(rb<=top)).sum()/top:.1%} | "
          f"top rows from best's bottom half {(rb[r<=top] > len(r)/2).mean():.1%} | top-6% from steps 44-49 {((r<=top)&(steps>=44)).sum()/top:.1%}", flush=True)
print('test part done', round(time.time() - t0))

if '--exams' in sys.argv:
    rows = []
    for k in range(7, 22):
        L = known[known.time_step <= k]; V = known[known.time_step.between(k + 1, k + 14)]
        Wk, ik = candidates(L, V)
        r = {'exam': f'{k+1}-{k+14}', **{n: roc_auc_score(V.label, pair(L, V, w)) for n, w in Wk.items()}, **ik}
        rows.append(r); print(r, round(time.time() - t0), flush=True)
    res = pd.DataFrame(rows).set_index('exam'); res.to_csv(OUT / 'similarity_weighting_validation.csv')
    print(res[['best', 'S', 'SH']].mean().round(5).to_string()); print('later seven:'); print(res[['best', 'S', 'SH']].iloc[-7:].mean().round(5).to_string())
    print('wins vs best:', {c: int((res[c] > res.best).sum()) for c in ['S', 'SH']})
print('done', round(time.time() - t0))
