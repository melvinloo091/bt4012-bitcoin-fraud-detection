import pandas as pd, numpy as np
from common import train, known, test
E=pd.read_csv('/home/claude/w/data/edges.csv'); print(E.columns.tolist(), len(E))
E.columns=['a','b']
print('unknown steps range', train.loc[train.label.isna(),'time_step'].agg(['min','max']).tolist())
B=pd.read_parquet('base_exam_preds.pq'); B['p']=.5*B.px+.5*B.pl
lab=dict(zip(known.txId,known.label))
und=pd.concat([E,E.rename(columns={'a':'b','b':'a'})])
for k in [15,17,19,21]:
    g=B[B.k==k].copy()
    score=dict(zip(g.txId,g.p))
    nb=und[und.a.isin(g.txId)].copy(); nb['nlab']=nb.b.map(lab); nb['nscore']=nb.b.map(score)
    agg=nb.groupby('a').agg(deg=('b','size'),n_illicit=('nlab',lambda s:(s==1).sum()),max_nscore=('nscore','max'))
    g=g.join(agg,on='txId'); g[['deg','n_illicit']]=g[['deg','n_illicit']].fillna(0)
    ill=g[g.label==1]; miss=ill[ill.p<g[g.label==0].p.quantile(0.95)]
    print(f'k={k} illicit={len(ill)} missed={len(miss)} | ill w/ illicit nbr {(ill.n_illicit>0).mean():.2f} | missed w/ illicit nbr {(miss.n_illicit>0).mean():.2f} | missed nbrscore>0.5 {(miss.max_nscore>0.5).mean():.2f} | licit nbrscore>0.5 {(g[g.label==0].max_nscore>0.5).mean():.3f} | missed deg0 {(miss.deg==0).mean():.2f}')
    print('   missed by step', miss.time_step.value_counts().sort_index().to_dict())
