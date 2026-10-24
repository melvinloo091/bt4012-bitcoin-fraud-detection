import pandas as pd, numpy as np, os
from sklearn.metrics import roc_auc_score
B = pd.read_parquet('lab_exam.pq')[['k','txId','label','gx','pl']]
M = B.merge(pd.read_parquet('new_exam_fast.pq').drop(columns=['time_step','label']), on=['k','txId'])
combos = {'best (0.95524)': lambda g: .5*g.gx+.5*g.pl,
          'ranks in both': lambda g: .5*g.x_rank+.5*g.l_rank,
          'XGB raw + LGB ranks': lambda g: .5*g.gx+.5*g.l_rank,
          'XGB ranks + LGB raw': lambda g: .5*g.x_rank+.5*g.pl,
          'raw pair + rank pair (4-way)': lambda g: .25*(g.gx+g.pl+g.x_rank+g.l_rank),
          'own features 1-93 only': lambda g: .5*g.x_loc+.5*g.l_loc,
          'XGB raw + LGB own-only': lambda g: .5*g.gx+.5*g.l_loc}
if os.path.exists('new_exam_dart.pq'):
    D = pd.read_parquet('new_exam_dart.pq').drop(columns=['time_step','label'])
    M = M.merge(D, on=['k','txId'], how='left')
    combos.update({'DART both': lambda g: .5*g.x_dart+.5*g.l_dart, 'XGB raw + LGB DART': lambda g: .5*g.gx+.5*g.l_dart,
                   'XGB DART + LGB raw': lambda g: .5*g.x_dart+.5*g.pl})
rows = []
for k, g in M.groupby('k'):
    r = {'k': k}
    for n, f in combos.items():
        s = f(g)
        r[n] = roc_auc_score(g.label, s) if s.notna().all() else np.nan
    rows.append(r)
R = pd.DataFrame(rows).set_index('k'); pd.set_option('display.width', 250)
print(R.round(4).to_string())
b = R['best (0.95524)']
S = pd.DataFrame({'mean15': R.mean(), 'later7 (15-21)': R.loc[15:21].mean(), 'wins_vs_best': R.gt(b, axis=0).sum(), 'n': R.notna().sum()})
print(S.round(4).to_string())
