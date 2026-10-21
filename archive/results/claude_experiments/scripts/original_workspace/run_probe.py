from base import *
import sys
from lightgbm import LGBMClassifier
F=FEATURE_COLUMNS; LOC=F[:93]
TF_DROP=['feat_100','feat_101','feat_103','feat_136','feat_137','feat_139']
ALLW=WINDOWS+[(k,k+1,35) for k in range(22,33)]
def lgbm(): return LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=15,max_bin=63,min_child_samples=5,min_child_weight=1e-3,reg_lambda=1,objective='binary',deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=2,random_state=42)
def pair(cols_x, cols_l, weight=None):
    def f(L,V):
        w=None
        if weight: end=L.time_step.max(); w=0.5**((end-L.time_step)/weight); w=w/w.mean()
        x=xgb_maker(reg_lambda=1,min_child_weight=1)(200); x.fit(L[cols_x],L.label,sample_weight=w)
        l=lgbm().fit(L[cols_l],L.label,sample_weight=w)
        return 0.5*x.predict_proba(V[cols_x])[:,1]+0.5*l.predict_proba(V[cols_l])[:,1]
    return f
todo={'pair_local':pair([c for c in LOC if c not in TF_DROP],LOC),
      'pair_recent14':pair([c for c in F if c not in TF_DROP],F,weight=14),
      'pair_noagg_drift':pair([c for c in F if c not in TF_DROP],F)}  # control: same as pair, rebuilt
for n in sys.argv[1:]: get_preds(n,todo[n],windows=ALLW,final=True); print(n,'done',flush=True)
