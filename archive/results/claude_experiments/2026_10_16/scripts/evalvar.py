import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B=pd.read_parquet('/home/claude/w/exp/base_exam_preds.pq'); V=pd.read_parquet('/home/claude/w/exp/var_exam_preds.pq')
M=B.merge(V,on=['k','txId'])
def rk(s): return s.rank(pct=True)
combos={
 'best':lambda g:.5*g.px+.5*g.pl,
 'xcs+lcs':lambda g:.5*g.xcs+.5*g.lcs,
 'xcs+pl':lambda g:.5*g.xcs+.5*g.pl,
 'px+lcs':lambda g:.5*g.px+.5*g.lcs,
 'rf':lambda g:g.rf,'et':lambda g:g.et,
 'best+rf 80/20':lambda g:.8*(.5*g.px+.5*g.pl)+.2*g.rf,
 'best+rf rank 70/30':lambda g:.7*rk(.5*g.px+.5*g.pl)+.3*rk(g.rf),
 'best+et 80/20':lambda g:.8*(.5*g.px+.5*g.pl)+.2*g.et,
 'cs4':lambda g:.25*(g.px+g.pl+g.xcs+g.lcs),
 'cs4+rf':lambda g:.8*.25*(g.px+g.pl+g.xcs+g.lcs)+.2*g.rf,
 'cs+rf+et':lambda g:.7*(.5*g.xcs+.5*g.lcs)+.15*g.rf+.15*g.et,
}
rows=[]
for k,g in M.groupby('k'): rows.append({'k':k,**{n:roc_auc_score(g.label,f(g)) for n,f in combos.items()}})
R=pd.DataFrame(rows).set_index('k'); pd.set_option('display.width',250)
print(R.round(4).to_string())
o=R.loc[7:21]
print(pd.DataFrame({'orig15':o.mean(),'later7':R.loc[15:21].mean(),'wins':(o.gt(o.best,axis=0)).sum(),'k22':R.loc[22] if 22 in R.index else np.nan}).round(5).to_string())
