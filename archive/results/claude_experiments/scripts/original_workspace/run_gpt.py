from base import *
import sys
from lightgbm import LGBMClassifier
F=FEATURE_COLUMNS
ALLW=WINDOWS+[(k,k+1,35) for k in range(22,33)]
FIX=dict(n_estimators=150,learning_rate=0.05,num_leaves=15,max_bin=63,min_child_weight=0.001,reg_lambda=1,objective='binary',deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=2,random_state=42)
def lg(m): return LGBMClassifier(min_child_samples=m,**FIX)
def f_leaf50(L,V): return lg(50).fit(L[F],L.label).predict_proba(V[F])[:,1]
def f_leaf5(L,V): return lg(5).fit(L[F],L.label).predict_proba(V[F])[:,1]
todo={'gpt_leaf50':f_leaf50,'gpt_leaf5':f_leaf5}
for n in sys.argv[1:]: get_preds(n,todo[n],windows=ALLW,final=True); print(n,'done',flush=True)
C5=pickle.load(open(f'{P}/gpt_leaf5.pkl','rb')); C50=pickle.load(open(f'{P}/gpt_leaf50.pkl','rb'))
pickle.dump({k:0.5*C5[k]+0.5*C50[k] for k in C5},open(f'{P}/gpt_leaf_blend.pkl','wb'))
