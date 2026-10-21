import sys; sys.path.insert(0,'/home/claude/w/exp')
from common import *
import time
G=pd.read_parquet('gfeat.pq').drop(columns=['g_comp','g_chainrun'])
GF=list(G.columns)
kn=known.join(G,on='txId'); te=test.join(G,on='txId')
print('rank corr with time:',kn[GF].rank().corrwith(kn.time_step.rank()).round(2).to_dict(),flush=True)
def fit(L,V,extra):
    xf=[c for c in FEATURES if c not in calendar(L)]+extra; lf=FEATURES+extra; w=recency(L.time_step,4)
    px=xgb().fit(L[xf],L.label,sample_weight=w).predict_proba(V[xf])[:,1]
    pl=lgb().fit(L[lf],L.label,sample_weight=w).predict_proba(V[lf])[:,1]
    return px,pl
mode=sys.argv[1]
t0=time.time(); res=[]
for k in range(7,23):
    L=kn[kn.time_step<=k]; V=kn[kn.time_step.between(k+1,min(k+14,35))]
    px,pl=fit(L,V,GF); res.append(pd.DataFrame({'k':k,'txId':V.txId.values,'gx':px,'gl':pl})); print(k,round(time.time()-t0),flush=True)
    pd.concat(res).to_parquet(f'g_exam_{mode}.pq')
px,pl=fit(kn,te,GF); pd.DataFrame({'index':te['index'].values,'gx':px,'gl':pl}).to_parquet(f'g_test_{mode}.pq'); print('done')
