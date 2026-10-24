"""24 Oct: untested model variants vs the 0.95524 model (XGB 300 + network counts, LGB 150, half-life 4).
  dart : DART boosting (random tree dropout) in both models
  rank : every feature replaced by its rank within its own time step (computed among labelled rows of that step)
  loc  : transaction's own features only (1-93), network counts still in XGBoost
Saves exam predictions (15 rolling exams) and test predictions."""
import sys, time; sys.path.insert(0, '/home/claude/w/exp')
from common import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
G = graphshape(); GF = list(G.columns)
kn = known.join(G, on='txId'); te = test.join(G, on='txId')
# within-step ranks (labelled rows only, consistent with the test file which holds only labelled rows)
RF_ = [f'r_{c}' for c in FEATURES]
def add_ranks(df):
    r = df.groupby('time_step')[FEATURES].rank(pct=True); r.columns = RF_
    return pd.concat([df, r], axis=1)
kn = add_ranks(kn); te = add_ranks(te)
LOC = [f'feat_{i}' for i in range(1, 94)]
def xdart(): return XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method='hist', grow_policy='lossguide', max_depth=0, max_leaves=15,
    max_bin=127, min_child_weight=1, reg_lambda=1, objective='binary:logistic', n_jobs=2, random_state=42,
    booster='dart', rate_drop=0.1, skip_drop=0.5, sample_type='uniform', normalize_type='tree')
def ldart(): return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5, min_child_weight=1e-3,
    reg_lambda=1, objective='binary', deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=2, random_state=42,
    boosting_type='dart', drop_rate=0.1, skip_drop=0.5, max_drop=50, drop_seed=42)
def fit(L, V):
    cal = calendar(L); xf = [c for c in FEATURES if c not in cal]; w = recency(L.time_step, 4); o = {}
    o['x_dart'] = xdart().fit(L[xf + GF], L.label, sample_weight=w).predict_proba(V[xf + GF])[:, 1]
    o['l_dart'] = ldart().fit(L[FEATURES], L.label, sample_weight=w).predict_proba(V[FEATURES])[:, 1]
    o['x_rank'] = xgb().fit(L[RF_ + GF], L.label, sample_weight=w).predict_proba(V[RF_ + GF])[:, 1]
    o['l_rank'] = lgb().fit(L[RF_], L.label, sample_weight=w).predict_proba(V[RF_])[:, 1]
    xl = [c for c in LOC if c not in cal]
    o['x_loc'] = xgb().fit(L[xl + GF], L.label, sample_weight=w).predict_proba(V[xl + GF])[:, 1]
    o['l_loc'] = lgb().fit(L[LOC], L.label, sample_weight=w).predict_proba(V[LOC])[:, 1]
    return o
t0 = time.time()
o = fit(kn, te); pd.DataFrame({'index': te['index'].values, 'txId': te.txId.values, 'time_step': te.time_step.values, **o}).to_parquet('new_test.pq')
print('test done', round(time.time() - t0), flush=True)
res = []
for k in range(7, 22):
    L = kn[kn.time_step <= k]; V = kn[kn.time_step.between(k + 1, k + 14)]
    o = fit(L, V); res.append(pd.DataFrame({'k': k, 'txId': V.txId.values, 'time_step': V.time_step.values, 'label': V.label.values, **o}))
    pd.concat(res).to_parquet('new_exam.pq'); print(k, round(time.time() - t0), flush=True)
print('done')
