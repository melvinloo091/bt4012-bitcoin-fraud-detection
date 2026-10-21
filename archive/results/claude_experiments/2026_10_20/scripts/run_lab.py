import sys,time; sys.path.insert(0,'/home/claude/w/exp')
from common import *
G=graphshape(); GF=list(G.columns); L=pd.read_parquet('lfeat.pq'); LF=list(L.columns)
kn=known.join(G,on='txId').join(L,on='txId'); te=test.join(G,on='txId').join(L,on='txId'); kn[LF]=kn[LF].fillna(0); te[LF]=te[LF].fillna(0)
def fit(Ltr,V):
    cal=calendar(Ltr); xf=[c for c in FEATURES if c not in cal]; w=recency(Ltr.time_step,4); out={}
    for name,cols in [('px',xf),('gx',xf+GF),('lx',xf+LF),('glx',xf+GF+LF)]:
        out[name]=xgb().fit(Ltr[cols],Ltr.label,sample_weight=w).predict_proba(V[cols])[:,1]
    for name,cols in [('pl',FEATURES),('ll',FEATURES+LF)]:
        out[name]=lgb().fit(Ltr[cols],Ltr.label,sample_weight=w).predict_proba(V[cols])[:,1]
    return out
t0=time.time()
o=fit(kn,te); pd.DataFrame({'index':te['index'].values,'txId':te.txId.values,'time_step':te.time_step.values,**o}).to_parquet('lab_test.pq'); print('test done',round(time.time()-t0),flush=True)
res=[]
for k in range(7,22):
    Ltr=kn[kn.time_step<=k]; V=kn[kn.time_step.between(k+1,k+14)]
    o=fit(Ltr,V); res.append(pd.DataFrame({'k':k,'txId':V.txId.values,'time_step':V.time_step.values,'label':V.label.values,**o}))
    pd.concat(res).to_parquet('lab_exam.pq'); print(k,round(time.time()-t0),flush=True)
print('done')
