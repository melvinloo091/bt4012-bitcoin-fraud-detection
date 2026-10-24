import sys, pandas as pd, numpy as np
te=pd.read_parquet('/home/claude/w/data/test.pq',columns=['index','time_step']).set_index('index')
def check(cand, ref):
    c=cand.reindex(te.index); r=ref.reindex(te.index)
    top=c.rank(ascending=False)<=0.06*len(c); bottom_ref=r.rank(pct=True)<=0.5
    rtop=r.rank(ascending=False)<=0.06*len(r)
    return {'agreement(spearman)':round(c.corr(r,method='spearman'),4),'top6%_from_ref_bottom_half':round((top&bottom_ref).sum()/top.sum(),4),
            'post_shutdown_share_top6%':round(te.time_step[top].ge(44).mean(),4),'ref_post_shutdown_share':round(te.time_step[rtop].ge(44).mean(),4)}
