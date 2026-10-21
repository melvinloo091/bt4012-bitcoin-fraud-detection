import sys; sys.path.insert(0,'/home/claude/w/exp')
from common import *
import time
t0=time.time(); res=[]
def fit_pair(L,V):
    xf=[c for c in FEATURES if c not in calendar(L)]; w=recency(L.time_step,4)
    px=xgb().fit(L[xf],L.label,sample_weight=w).predict_proba(V[xf])[:,1]
    pl=lgb().fit(L[FEATURES],L.label,sample_weight=w).predict_proba(V[FEATURES])[:,1]
    return px,pl
for k,L,V in exams():
    px,pl=fit_pair(L,V)
    res.append(pd.DataFrame({'k':k,'txId':V.txId.values,'time_step':V.time_step.values,'label':V.label.values,'px':px,'pl':pl}))
    print(k,round(time.time()-t0),flush=True)
pd.concat(res).to_parquet('/home/claude/w/exp/base_exam_preds.pq')
px,pl=fit_pair(known,test)
pd.DataFrame({'index':test['index'].values,'txId':test.txId.values,'time_step':test.time_step.values,'px':px,'pl':pl}).to_parquet('/home/claude/w/exp/base_test_preds.pq')
print('done',round(time.time()-t0))
