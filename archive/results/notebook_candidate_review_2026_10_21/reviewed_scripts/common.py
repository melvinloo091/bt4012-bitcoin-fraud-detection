import numpy as np, pandas as pd, scipy.sparse as sp
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
D='/home/claude/w/data/'
train=pd.read_parquet(D+'train.pq'); test=pd.read_parquet(D+'test.pq'); sample=pd.read_csv(D+'sample.csv')
FEATURES=[f'feat_{i}' for i in range(1,166)]
lab=pd.to_numeric(train['label'],errors='coerce')
known=train.loc[lab.notna()].copy(); known['label']=lab[lab.notna()].astype(int)
def calendar(rows,thr=0.9):
    c=rows[FEATURES].rank().corrwith(rows['time_step'].rank()); return set(c[c.abs()>thr].index)
def recency(steps,hl):
    w=0.5**((steps.max()-steps)/hl); return (w/w.mean()).to_numpy()
def xgb(n=300,leaves=15,seed=42): return XGBClassifier(n_estimators=n,learning_rate=0.05,tree_method='hist',grow_policy='lossguide',max_depth=0,
    max_leaves=leaves,max_bin=127,min_child_weight=1,reg_lambda=1,objective='binary:logistic',n_jobs=2,random_state=seed)
def lgb(n=150,seed=42): return LGBMClassifier(n_estimators=n,learning_rate=0.05,num_leaves=15,max_bin=63,min_child_samples=5,min_child_weight=1e-3,
    reg_lambda=1,objective='binary',deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=2,random_state=seed)
def graphshape():
    E=pd.read_csv(D+'edges.csv'); E.columns=['a','b']
    ids=pd.Index(pd.unique(np.r_[E.a.values,E.b.values,train.txId.values,test.txId.values])); n=len(ids)
    A=sp.csr_matrix((np.ones(len(E)),(ids.get_indexer(E.a),ids.get_indexer(E.b))),shape=(n,n)); U=((A+A.T)>0).astype(float).tocsr()
    g_out=np.asarray(A.sum(1)).ravel(); g_in=np.asarray(A.sum(0)).ravel(); g_deg=np.asarray(U.sum(1)).ravel()
    two=U@U; two.setdiag(0); two.eliminate_zeros(); g_hop2=np.asarray((two>0).sum(1)).ravel(); safe=np.maximum(g_deg,1)
    nb_max=np.array([g_deg[U.indices[U.indptr[i]:U.indptr[i+1]]].max() if U.indptr[i+1]>U.indptr[i] else 0 for i in range(n)])
    chain=((g_in==1)&(g_out==1)).astype(float)
    return pd.DataFrame({"g_in":g_in,"g_out":g_out,"g_deg":g_deg,"g_hop2":g_hop2,"g_nbdeg_mean":np.asarray(U@g_deg).ravel()/safe,"g_nbdeg_max":nb_max,
        "g_nb_chain":np.asarray(U@chain).ravel()/safe,"g_nb_out":np.asarray(U@g_out).ravel()/safe,"g_nb_in":np.asarray(U@g_in).ravel()/safe},index=pd.Index(ids,name='txId'))
