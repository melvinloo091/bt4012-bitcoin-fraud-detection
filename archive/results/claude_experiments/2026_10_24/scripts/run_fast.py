"""Fast variants on the 15 practice exams: within-step ranks (rank) and own features only (loc)."""
import sys, time; sys.path.insert(0, '/home/claude/w/exp')
from common import *
G = graphshape(); GF = list(G.columns)
kn = known.join(G, on='txId')
RF_ = [f'r_{c}' for c in FEATURES]
r = kn.groupby('time_step')[FEATURES].rank(pct=True); r.columns = RF_; kn = pd.concat([kn, r], axis=1)
LOC = [f'feat_{i}' for i in range(1, 94)]
t0 = time.time(); res = []
for k in range(7, 22):
    L = kn[kn.time_step <= k]; V = kn[kn.time_step.between(k + 1, k + 14)]
    cal = calendar(L); w = recency(L.time_step, 4); o = {}
    o['x_rank'] = xgb().fit(L[RF_ + GF], L.label, sample_weight=w).predict_proba(V[RF_ + GF])[:, 1]
    o['l_rank'] = lgb().fit(L[RF_], L.label, sample_weight=w).predict_proba(V[RF_])[:, 1]
    xl = [c for c in LOC if c not in cal]
    o['x_loc'] = xgb().fit(L[xl + GF], L.label, sample_weight=w).predict_proba(V[xl + GF])[:, 1]
    o['l_loc'] = lgb().fit(L[LOC], L.label, sample_weight=w).predict_proba(V[LOC])[:, 1]
    res.append(pd.DataFrame({'k': k, 'txId': V.txId.values, 'time_step': V.time_step.values, 'label': V.label.values, **o}))
    pd.concat(res).to_parquet('new_exam_fast.pq'); print(k, round(time.time() - t0), flush=True)
print('done')
