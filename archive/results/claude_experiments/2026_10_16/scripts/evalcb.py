import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B=pd.read_parquet('base_exam_preds.pq'); C=pd.read_parquet('cb_exam.pq'); M=B.merge(C,on=['k','txId'])
f={'best':lambda g:.5*g.px+.5*g.pl,'cb':lambda g:g.cb,'cb+lgb':lambda g:.5*g.cb+.5*g.pl,'cb+xgb':lambda g:.5*g.cb+.5*g.px,
   'trio 1/3':lambda g:(g.px+g.pl+g.cb)/3,'best 75 + cb 25':lambda g:.75*(.5*g.px+.5*g.pl)+.25*g.cb}
R=pd.DataFrame([{'k':k,**{n:roc_auc_score(g.label,fn(g)) for n,fn in f.items()}} for k,g in M.groupby('k')]).set_index('k')
pd.set_option('display.width',200); print(R.round(4).to_string()); o=R.loc[7:21]
print(pd.DataFrame({'orig15':o.mean(),'later7':R.loc[15:21].mean(),'wins15':o.gt(o.best,axis=0).sum(),'k22':R.loc[22] if 22 in R.index else np.nan}).round(5))
