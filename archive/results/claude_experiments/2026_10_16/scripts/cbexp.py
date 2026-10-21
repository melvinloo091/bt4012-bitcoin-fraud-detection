import sys; sys.path.insert(0,'/home/claude/w/exp')
from common import *
from catboost import CatBoostClassifier
import time
def cb(): return CatBoostClassifier(iterations=500,learning_rate=0.08,depth=6,l2_leaf_reg=1,border_count=128,thread_count=2,random_seed=42,verbose=0,allow_writing_files=False)
def fit(L,V):
    xf=[c for c in FEATURES if c not in calendar(L)]; w=recency(L.time_step,4)
    return cb().fit(L[xf],L.label,sample_weight=w).predict_proba(V[xf])[:,1]
t0=time.time(); res=[]
for k in range(7,23):
    L=known[known.time_step<=k]; V=known[known.time_step.between(k+1,min(k+14,35))]
    res.append(pd.DataFrame({'k':k,'txId':V.txId.values,'cb':fit(L,V)})); print(k,round(time.time()-t0),flush=True)
    pd.concat(res).to_parquet('cb_exam.pq')
pd.DataFrame({'index':test['index'].values,'cb':fit(known,test)}).to_parquet('cb_test.pq'); print('done',round(time.time()-t0))
