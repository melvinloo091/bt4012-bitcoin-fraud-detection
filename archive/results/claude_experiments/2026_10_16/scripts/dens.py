import pandas as pd, numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from common import known, test, FEATURES
LOC=[f'feat_{i}' for i in range(1,94)]
allr=pd.concat([known.assign(src='k'),test.assign(src='t',label=np.nan)])
X=allr[LOC].copy(); X=(X-X.mean())/X.std().replace(0,1); X=X.clip(-5,5).fillna(0)
out=[]
for s,idx in allr.groupby('time_step').indices.items():
    Z=X.values[idx]; nn=NearestNeighbors(n_neighbors=min(21,len(idx))).fit(Z); d,_=nn.kneighbors(Z)
    out.append(pd.DataFrame({'txId':allr.txId.values[idx],'d1':d[:,1],'d5':d[:,min(5,d.shape[1]-1)],'d20':d[:,-1]}))
D=pd.concat(out).set_index('txId'); D['n_close']=0
D.to_parquet('dens.pq')
B=pd.read_parquet('base_exam_preds.pq'); B['p']=.5*B.px+.5*B.pl; g=B[B.k==21].join(D,on='txId')
thr=g[g.label==0].p.quantile(0.95); miss=(g.label==1)&(g.p<thr); lic=g.label==0
for c in ['d1','d5','d20']:
    sub=g[miss|lic]; print(c,'AUC missed-illicit vs licit (low dist=illicit):',round(roc_auc_score(miss[miss|lic],-sub[c]),3),' all illicit vs licit:',round(roc_auc_score(g.label,-g[c]),3))
print(g[miss].groupby('time_step')[['d1','d5']].median().round(3)); print('licit median',g[lic][['d1','d5']].median().round(3).to_dict())
