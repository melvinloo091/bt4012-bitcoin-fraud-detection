import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
D='/home/claude/work/data'
test=pd.read_parquet(f'{D}/test.parquet')
def load_sub(path): 
    s=pd.read_csv(path).set_index('index').loc[test['index'],'target'].to_numpy(); return s
def compare(cand, ref, top=0.06):
    """No labels. How differently does `cand` rank the test rows compared with the reference?"""
    rc=rankdata(cand)/len(cand); rr=rankdata(ref)/len(ref)
    n=int(top*len(cand)); tc=set(np.argsort(-cand)[:n]); tr=set(np.argsort(-ref)[:n])
    promoted=np.array(sorted(tc-tr))   # rows in the candidate's top group that the reference does not rank highly
    steps=test.time_step.to_numpy()
    out={'spearman':spearmanr(cand,ref)[0],'top_overlap':len(tc&tr)/n,
         'promoted_from_bottom_half':float(np.mean(rr[promoted]<0.5)) if len(promoted) else 0.0,
         'max_step_gap':0.0,'worst_step':None}
    # per step: share of each step's rows that fall in the overall top group
    in_c=np.isin(np.arange(len(cand)),list(tc)); in_r=np.isin(np.arange(len(cand)),list(tr))
    per=pd.DataFrame({'step':steps,'c':in_c,'r':in_r}).groupby('step')[['c','r']].mean()
    gap=(per.c-per.r).abs(); out['max_step_gap']=gap.max(); out['worst_step']=int(gap.idxmax())
    return out, per
if __name__=='__main__':
    S='/home/claude/work/subs/'
    ref=load_sub(S+'submission_xgboost_tuned.csv')
    rows=[]
    for name,f,pub in [('HGB bin ensemble','submission_hist_bin_ensemble.csv',0.94199),('HGB 255','submission_hist_gradient_boosting.csv',0.93654),
                       ('Extra Trees','submission_extra_trees.csv',0.91673),('Look-alike blend','/home/claude/work/out/submissions/submission_xgboost_lookalike_blend.csv',0.87805),
                       ('XGB time-filtered','submission_xgboost_time_filtered.csv',None),('LightGBM','submission_lightgbm_tuned.csv',None),('Random forest','submission_random_forest.csv',None)]:
        o,_=compare(load_sub(f if f.startswith('/') else S+f),ref); o['model']=name; o['public']=pub; rows.append(o)
    print(pd.DataFrame(rows).set_index('model').round(3).to_string())
