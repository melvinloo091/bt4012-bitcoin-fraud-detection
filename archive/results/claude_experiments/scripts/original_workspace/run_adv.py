from base import *
import sys
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
F=FEATURE_COLUMNS
TF=[c for c in F if c not in ['feat_100','feat_101','feat_103','feat_136','feat_137','feat_139']]
ALLW=WINDOWS+[(k,k+1,35) for k in range(22,33)]
FIX=dict(n_estimators=150,learning_rate=0.05,num_leaves=15,max_bin=63,min_child_weight=1e-3,reg_lambda=1,objective='binary',deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=2,random_state=42)
def adv_weights(L, V, alpha=0.5, clip=(0.05,20)):
    """How test-like is each training row? Out-of-fold probability from a train-vs-target classifier (features only)."""
    X=pd.concat([L[TF],V[TF]]).to_numpy(); y=np.r_[np.zeros(len(L)),np.ones(len(V))]
    p=np.zeros(len(X))
    for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(X,y):
        m=LGBMClassifier(n_estimators=100,learning_rate=0.1,num_leaves=15,min_child_samples=50,verbosity=-1,n_jobs=2,random_state=0).fit(X[tr],y[tr])
        p[te]=m.predict_proba(X[te])[:,1]
    pL=np.clip(p[:len(L)],1e-4,1-1e-4); odds=pL/(1-pL)*len(L)/len(V)
    w=np.clip(odds**alpha,*clip); return w/w.mean()
def recency(L,hl=7):
    w=(0.5**((L.time_step.max()-L.time_step)/hl)).to_numpy(); return w/w.mean()
def pair(L,V,w):
    x=xgb_maker(reg_lambda=1,min_child_weight=1)(200); x.fit(L[TF],L.label,sample_weight=w)
    l=LGBMClassifier(min_child_samples=5,**FIX).fit(L[F],L.label,sample_weight=w)
    return 0.5*x.predict_proba(V[TF])[:,1]+0.5*l.predict_proba(V[F])[:,1]
def f_adv(L,V): return pair(L,V,adv_weights(L,V))
def f_adv_x_recent(L,V):
    w=adv_weights(L,V)*recency(L); return pair(L,V,w/w.mean())
todo={'adv_pair':f_adv,'adv_x_recent7':f_adv_x_recent}
for n in sys.argv[1:]: get_preds(n,todo[n],windows=ALLW,final=True); print(n,'done',flush=True)
