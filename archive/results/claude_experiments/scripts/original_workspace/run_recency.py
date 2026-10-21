from base import *
import sys
from lightgbm import LGBMClassifier
F=FEATURE_COLUMNS
TF=[c for c in F if c not in ['feat_100','feat_101','feat_103','feat_136','feat_137','feat_139']]
ALLW=WINDOWS+[(k,k+1,35) for k in range(22,33)]
FIX=dict(n_estimators=150,learning_rate=0.05,num_leaves=15,max_bin=63,min_child_weight=1e-3,reg_lambda=1,objective='binary',deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=2,random_state=42)
def W(L,hl):
    if hl is None: return None
    w=(0.5**((L.time_step.max()-L.time_step)/hl)).to_numpy(); return w/w.mean()
def mk(hl, leaves=(5,), xgb=True):
    def f(L,V):
        w=W(L,hl); ps=[]
        if xgb:
            x=xgb_maker(reg_lambda=1,min_child_weight=1)(200); x.fit(L[TF],L.label,sample_weight=w); ps.append(x.predict_proba(V[TF])[:,1])
        lg=[LGBMClassifier(min_child_samples=m,**FIX).fit(L[F],L.label,sample_weight=w).predict_proba(V[F])[:,1] for m in leaves]
        if xgb: return 0.5*ps[0]+0.5*np.mean(lg,0)
        return np.mean(lg,0)
    return f
todo={'pair_recent4':mk(4),'pair_recent3':mk(3),'pair_recent10':mk(10),
      'recent7_xgb_leaf5_50':mk(7,(5,50)),'recent7_leaf5_50':mk(7,(5,50),xgb=False),'recent4_xgb_leaf5_50':mk(4,(5,50))}
for n in sys.argv[1:]: get_preds(n,todo[n],windows=ALLW,final=True); print(n,'done',flush=True)
