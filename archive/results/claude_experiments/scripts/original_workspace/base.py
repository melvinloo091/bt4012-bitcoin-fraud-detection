import sys, os, pickle, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/work/scripts')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
D='/home/claude/work/data'; P='/home/claude/work/v2/preds'
RANDOM_SEED=42; FEATURE_COLUMNS=[f"feat_{i}" for i in range(1,166)]
train=pd.read_parquet(f'{D}/train.parquet'); test=pd.read_parquet(f'{D}/test.parquet')
labels=pd.to_numeric(train["label"],errors="coerce"); labeled=train.loc[labels.notna()].copy(); labeled["label"]=labels.loc[labels.notna()].astype(int)
unknown=train.loc[labels.isna()].copy()
WINDOWS=[(k,k+1,k+14) for k in range(7,22)]   # 15 rolling 14-step windows
PUBLIC={'hgb255':0.93654,'et':0.91673,'hgb_ens':0.94199,'xgb_tuned':0.94265,'blend_look':0.87805}

def chrono_xgb(learning, cols, make, max_rounds=400, patience=60):
    end=int(learning.time_step.max()); a=learning[learning.time_step<=end-3]; b=learning[learning.time_step>end-3]
    m=make(max_rounds); m.set_params(early_stopping_rounds=patience, eval_metric='auc')
    m.fit(a[cols],a.label,eval_set=[(b[cols],b.label)],verbose=False); r=m.best_iteration+1
    m2=make(r); m2.set_params(early_stopping_rounds=None); m2.fit(learning[cols],learning.label); return m2,r

def xgb_maker(**kw):
    def make(n):
        p=dict(n_estimators=int(n),learning_rate=0.05,tree_method="hist",grow_policy="lossguide",max_depth=0,max_leaves=15,max_bin=127,min_child_weight=3,reg_lambda=10,objective="binary:logistic",n_jobs=2,random_state=RANDOM_SEED); p.update(kw)
        return XGBClassifier(**p)
    return make

def get_preds(name, fn, windows=WINDOWS, final=False):
    """fn(learning_df, target_df, period) -> probabilities for target_df. Cached."""
    path=f'{P}/{name}.pkl'
    cache=pickle.load(open(path,'rb')) if os.path.exists(path) else {}
    changed=False
    for (k,a,b) in windows:
        if k in cache: continue
        L=labeled[labeled.time_step<=k]; V=labeled[labeled.time_step.between(a,b)]
        with threadpool_limits(limits=2): cache[k]=pd.Series(fn(L,V),index=V.index)
        changed=True
    if final and 'test' not in cache:
        with threadpool_limits(limits=2): cache['test']=pd.Series(fn(labeled,test),index=test.index); changed=True
    if changed: pickle.dump(cache,open(path,'wb'))
    return cache

def scores(cache, windows=WINDOWS):
    out={}
    for (k,a,b) in windows:
        V=labeled.loc[cache[k].index]; out[f'{a}-{b}']=roc_auc_score(V.label,cache[k])
    return pd.Series(out)
