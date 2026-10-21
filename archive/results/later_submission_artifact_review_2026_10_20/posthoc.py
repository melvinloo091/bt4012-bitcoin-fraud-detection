import sys; sys.path.insert(0,'/home/claude/w/exp')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
P=pd.read_parquet('/home/claude/w/exp/base_exam_preds.pq')
from common import known, recency
def em_adjust(p, pi_tr, iters=100):
    # Saerens et al. 2002: estimate new prior, return adjusted posteriors
    p=np.clip(p,1e-6,1-1e-6); pi=pi_tr
    for _ in range(iters):
        a=(pi/pi_tr)*p; b=((1-pi)/(1-pi_tr))*(1-p); q=a/(a+b); new=q.mean()
        if abs(new-pi)<1e-7: break
        pi=new
    return q, pi
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
rows=[]
for k,g in P.groupby('k'):
    L=known[known.time_step<=k]; w=recency(L.time_step,4); pi_tr=float((w*L.label).sum()/w.sum())
    p=0.5*g.px+0.5*g.pl; r={'k':k,'n_steps':g.time_step.nunique(),'base':roc_auc_score(g.label,p)}
    for name,fn in [('em',lambda s: em_adjust(s.values,pi_tr)[0]),('em_px_pl',None)]:
        if fn is None: continue
        q=g.assign(p=p).groupby('time_step')['p'].transform(fn); r[name]=roc_auc_score(g.label,q)
    # EM on each model separately then average
    qx=g.groupby('time_step')['px'].transform(lambda s: em_adjust(s.values,pi_tr)[0]); ql=g.groupby('time_step')['pl'].transform(lambda s: em_adjust(s.values,pi_tr)[0])
    r['em_sep']=roc_auc_score(g.label,0.5*qx+0.5*ql)
    # half-strength: logit shift halfway
    q=g.assign(p=p).groupby('time_step')['p'].transform(lambda s: em_adjust(s.values,pi_tr)[0]); r['em_half']=roc_auc_score(g.label,1/(1+np.exp(-(0.5*logit(q)+0.5*logit(p)))))
    # oracle (true step prevalence) - upper bound, uses labels, diagnostic only
    tp=g.groupby('time_step').label.transform('mean').clip(1e-3,1-1e-3)
    r['oracle_prior']=roc_auc_score(g.label,1/(1+np.exp(-(logit(p)+logit(tp)-np.log(pi_tr/(1-pi_tr))))))
    rows.append(r)
R=pd.DataFrame(rows).set_index('k'); pd.set_option('display.width',200); print(R.round(4))
o=R.loc[7:21]; print('orig15 mean\n',o.mean().round(5)); print('later7 (15-21)\n',R.loc[15:21].mean().round(5)); print('k22-32\n',R.loc[22:].mean().round(5))
for c in R.columns[2:]: print(c,'wins orig15',int((o[c]>o.base).sum()))
