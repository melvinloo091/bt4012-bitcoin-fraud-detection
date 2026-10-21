import numpy as np, pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
D='/home/claude/w/data/'
train=pd.read_parquet(D+'train.pq'); test=pd.read_parquet(D+'test.pq'); sample=pd.read_csv(D+'sample.csv')
FEATURES=[f'feat_{i}' for i in range(1,166)]
lab=pd.to_numeric(train['label'],errors='coerce')
known=train.loc[lab.notna()].copy(); known['label']=lab[lab.notna()].astype(int)
unknown=train.loc[lab.isna()].copy()
def calendar(rows,thr=0.9):
    c=rows[FEATURES].rank().corrwith(rows['time_step'].rank()); return set(c[c.abs()>thr].index)
def recency(steps,hl):
    w=0.5**((steps.max()-steps)/hl); return (w/w.mean()).to_numpy()
def xgb(n=300,leaves=15,seed=42): return XGBClassifier(n_estimators=n,learning_rate=0.05,tree_method='hist',grow_policy='lossguide',max_depth=0,
    max_leaves=leaves,max_bin=127,min_child_weight=1,reg_lambda=1,objective='binary:logistic',n_jobs=2,random_state=seed)
def lgb(n=150,seed=42): return LGBMClassifier(n_estimators=n,learning_rate=0.05,num_leaves=15,max_bin=63,min_child_samples=5,min_child_weight=1e-3,
    reg_lambda=1,objective='binary',deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=2,random_state=seed)
def exams():
    out=[]
    for k in range(7,22): out.append((k,known[known.time_step<=k],known[known.time_step.between(k+1,k+14)]))
    for k in range(22,33): out.append((k,known[known.time_step<=k],known[known.time_step.between(k+1,35)]))
    return out
