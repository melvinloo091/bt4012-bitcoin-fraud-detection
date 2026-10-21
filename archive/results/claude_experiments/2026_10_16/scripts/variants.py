import sys; sys.path.insert(0,'/home/claude/w/exp')
from common import *
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import time
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
def xcs(seed,cs=0.5): return XGBClassifier(n_estimators=300,learning_rate=0.05,tree_method='hist',grow_policy='lossguide',max_depth=0,max_leaves=15,max_bin=127,
    min_child_weight=1,reg_lambda=1,subsample=0.8,colsample_bytree=cs,objective='binary:logistic',n_jobs=2,random_state=seed)
def lcs(seed,cs=0.5): return LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=15,max_bin=63,min_child_samples=5,min_child_weight=1e-3,reg_lambda=1,
    subsample=0.8,subsample_freq=1,colsample_bytree=cs,objective='binary',force_col_wise=True,verbosity=-1,n_jobs=2,random_state=seed)
def run(L,V):
    xf=[c for c in FEATURES if c not in calendar(L)]; w=recency(L.time_step,4); o={}
    o['xcs']=np.mean([xcs(s).fit(L[xf],L.label,sample_weight=w).predict_proba(V[xf])[:,1] for s in (1,2)],0)
    o['lcs']=np.mean([lcs(s).fit(L[FEATURES],L.label,sample_weight=w).predict_proba(V[FEATURES])[:,1] for s in (1,2)],0)
    o['rf']=RandomForestClassifier(500,max_features='sqrt',n_jobs=2,random_state=42).fit(L[xf],L.label,sample_weight=w).predict_proba(V[xf])[:,1]
    o['et']=ExtraTreesClassifier(500,max_features='sqrt',n_jobs=2,random_state=42).fit(L[xf],L.label,sample_weight=w).predict_proba(V[xf])[:,1]
    return o
t0=time.time(); res=[]
for k,L,V in exams():
    if k>22: continue
    o=run(L,V); res.append(pd.DataFrame({'k':k,'txId':V.txId.values,**o})); print(k,round(time.time()-t0),flush=True)
    pd.concat(res).to_parquet('/home/claude/w/exp/var_exam_preds.pq')
o=run(known,test); pd.DataFrame({'index':test['index'].values,**o}).to_parquet('/home/claude/w/exp/var_test_preds.pq'); print('done',round(time.time()-t0))
