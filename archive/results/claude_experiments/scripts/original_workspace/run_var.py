from base import *
import sys
from lightgbm import LGBMClassifier
F=FEATURE_COLUMNS
TF=[c for c in F if c not in ['feat_100','feat_101','feat_103','feat_136','feat_137','feat_139']]
ALLW=WINDOWS+[(k,k+1,35) for k in range(22,33)]
def lgbm(**kw):
    p=dict(n_estimators=150,learning_rate=0.05,num_leaves=15,max_bin=63,min_child_samples=5,min_child_weight=1e-3,reg_lambda=1,objective='binary',deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=2,random_state=42); p.update(kw); return LGBMClassifier(**p)
def mk(cx=TF, cl=F, hl=None):
    def f(L,V):
        w=None
        if hl: end=L.time_step.max(); w=(0.5**((end-L.time_step)/hl)).to_numpy(); w=w/w.mean()
        x=xgb_maker(reg_lambda=1,min_child_weight=1)(200); x.fit(L[cx],L.label,sample_weight=w)
        l=lgbm().fit(L[cl],L.label,sample_weight=w)
        return 0.5*x.predict_proba(V[cx])[:,1]+0.5*l.predict_proba(V[cl])[:,1]
    return f
todo={'pair_tf_both':mk(TF,TF),'pair_recent7':mk(hl=7),'pair_recent14_tfboth':mk(TF,TF,14)}
for n in sys.argv[1:]: get_preds(n,todo[n],windows=ALLW,final=True); print(n,'done',flush=True)
