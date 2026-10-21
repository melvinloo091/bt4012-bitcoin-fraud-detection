import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B=pd.read_parquet('base_exam_preds.pq'); G=pd.read_parquet(sys.argv[1]); M=B.merge(G,on=['k','txId'])
rows=[]
for k,g in M.groupby('k'):
    b=.5*g.px+.5*g.pl; n=.5*g.gx+.5*g.gl
    rows.append({'k':k,'best':roc_auc_score(g.label,b),'graph':roc_auc_score(g.label,n),'g_xgb_only':roc_auc_score(g.label,.5*g.gx+.5*g.pl),
                 'g_lgb_only':roc_auc_score(g.label,.5*g.px+.5*g.gl),'avg_best_graph':roc_auc_score(g.label,.5*b+.5*n)})
R=pd.DataFrame(rows).set_index('k'); print(R.round(4).to_string()); o=R.loc[7:21]
print(pd.DataFrame({'orig15':o.mean(),'later7':R.loc[15:21].mean(),'wins':o.gt(o.best,axis=0).sum(),'k22':R.loc[22] if 22 in R.index else np.nan}).round(5))
