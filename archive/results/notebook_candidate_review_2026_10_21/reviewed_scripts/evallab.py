import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
M=pd.read_parquet('lab_exam.pq')
combos={'best (0.95484)':lambda g:.5*g.px+.5*g.pl,
        'graphshape (0.95524)':lambda g:.5*g.gx+.5*g.pl,
        'lab in XGB':lambda g:.5*g.lx+.5*g.pl,
        'lab in LGB':lambda g:.5*g.px+.5*g.ll,
        'lab in both':lambda g:.5*g.lx+.5*g.ll,
        'graph+lab in XGB':lambda g:.5*g.glx+.5*g.pl,
        'graph+lab XGB, lab LGB':lambda g:.5*g.glx+.5*g.ll}
R=pd.DataFrame([{'k':k,**{n:roc_auc_score(g.label,f(g)) for n,f in combos.items()}} for k,g in M.groupby('k')]).set_index('k')
pd.set_option('display.width',250); print(R.round(4).to_string())
ref='graphshape (0.95524)'; b='best (0.95484)'
print(pd.DataFrame({'mean15':R.mean(),'later7(15-21)':R.loc[15:21].mean(),'wins_vs_best':R.gt(R[b],axis=0).sum(),'wins_vs_graphshape':R.gt(R[ref],axis=0).sum(),
                    'n_exams':len(R)}).round(5).to_string())
