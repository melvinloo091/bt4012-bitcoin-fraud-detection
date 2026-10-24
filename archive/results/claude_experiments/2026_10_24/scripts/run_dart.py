"""DART variant on selected practice exams (slow: random tree dropout makes every boosting round re-score all trees)."""
import sys, time; sys.path.insert(0, '/home/claude/w/exp')
from common import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
G = graphshape(); GF = list(G.columns)
kn = known.join(G, on='txId')
def xdart(): return XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method='hist', grow_policy='lossguide', max_depth=0, max_leaves=15,
    max_bin=127, min_child_weight=1, reg_lambda=1, objective='binary:logistic', n_jobs=2, random_state=42,
    booster='dart', rate_drop=0.1, skip_drop=0.5, sample_type='uniform', normalize_type='tree')
def ldart(): return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5, min_child_weight=1e-3,
    reg_lambda=1, objective='binary', deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=2, random_state=42,
    boosting_type='dart', drop_rate=0.1, skip_drop=0.5, max_drop=50, drop_seed=42)
KS = [int(a) for a in sys.argv[1].split(',')]
t0 = time.time(); res = []
for k in KS:
    L = kn[kn.time_step <= k]; V = kn[kn.time_step.between(k + 1, k + 14)]
    cal = calendar(L); xf = [c for c in FEATURES if c not in cal]; w = recency(L.time_step, 4); o = {}
    o['l_dart'] = ldart().fit(L[FEATURES], L.label, sample_weight=w).predict_proba(V[FEATURES])[:, 1]
    o['x_dart'] = xdart().fit(L[xf + GF], L.label, sample_weight=w).predict_proba(V[xf + GF])[:, 1]
    res.append(pd.DataFrame({'k': k, 'txId': V.txId.values, 'time_step': V.time_step.values, 'label': V.label.values, **o}))
    pd.concat(res).to_parquet('new_exam_dart.pq'); print(k, round(time.time() - t0), flush=True)
print('done')
