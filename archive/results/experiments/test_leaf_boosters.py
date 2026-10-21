"""Use leaf-limited trees, matching the successful HGB model's capacity."""
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping

HERE = Path(__file__).resolve().parent
FEATURES = [f'feat_{i}' for i in range(1,166)]
known = pd.read_csv(Path(__file__).resolve().parents[1] / 'Data Set' / 'Train Data.csv').dropna(subset=['label']).copy()
known['label'] = known.label.astype(int)
CONFIGS = {
    f'leaves15_bin{bins}_{name}': dict(max_bin=bins, leaves=15, weight=weight,
                                      regularization=10, sampling=1.)
    for bins in (31,63,127) for name,weight in [('natural',1.),('balanced','ratio')]
}
CONFIGS.update({
    'leaves7_bin63_balanced': dict(max_bin=63,leaves=7,weight='ratio',regularization=10,sampling=1.),
    'leaves15_bin63_smooth': dict(max_bin=63,leaves=15,weight='ratio',regularization=30,sampling=.9),
})

def make(family, cfg, rounds, weight, *, stopping=False):
    common = dict(n_estimators=int(rounds), learning_rate=.05, max_bin=cfg['max_bin'],
                  reg_lambda=cfg['regularization'], subsample=cfg['sampling'],
                  colsample_bytree=cfg['sampling'], scale_pos_weight=weight,
                  n_jobs=4, random_state=42)
    if family=='xgboost':
        return XGBClassifier(**common, grow_policy='lossguide', max_depth=0,
                             max_leaves=cfg['leaves'], min_child_weight=3,
                             objective='binary:logistic', eval_metric='auc', tree_method='hist',
                             early_stopping_rounds=60 if stopping else None)
    return LGBMClassifier(**common, num_leaves=cfg['leaves'], max_depth=-1,
                          min_child_samples=30, subsample_freq=1 if cfg['sampling']<1 else 0,
                          objective='binary', metric='auc', deterministic=True,
                          force_col_wise=True, verbosity=-1)

def fit(family,cfg,learning):
    end=int(learning.time_step.max())
    a=learning.loc[learning.time_step<=end-3];b=learning.loc[learning.time_step>end-3]
    weight=cfg['weight']
    if weight=='ratio':weight=float((a.label==0).sum()/(a.label==1).sum())
    m=make(family,cfg,400,weight,stopping=True)
    with threadpool_limits(limits=4):
        if family=='xgboost':
            m.fit(a[FEATURES],a.label,eval_set=[(b[FEATURES],b.label)],verbose=False)
            rounds=int(m.best_iteration+1)
        else:
            m.fit(a[FEATURES],a.label,eval_X=b[FEATURES],eval_y=b.label,
                  callbacks=[early_stopping(60,verbose=False)])
            rounds=int(m.best_iteration_)
        if cfg['weight']=='ratio':weight=float((learning.label==0).sum()/(learning.label==1).sum())
        fitted=make(family,cfg,rounds,weight)
        fitted.fit(learning[FEATURES],learning.label)
    return fitted,rounds

def main():
    results=[];saved={}
    for family in ['xgboost','lightgbm']:
        for name,cfg in CONFIGS.items():
            for end,start,stop in [(14,15,28),(21,22,35)]:
                learning=known.loc[known.time_step<=end];checking=known.loc[known.time_step.between(start,stop)]
                tic=time.perf_counter();model,rounds=fit(family,cfg,learning)
                with threadpool_limits(limits=4):scores=model.predict_proba(checking[FEATURES])[:,1]
                saved[f'{family}_{name}_{end}']=scores
                record=dict(family=family,configuration=name,train_end=end,auc=float(roc_auc_score(checking.label,scores)),rounds=rounds,
                            seconds=round(time.perf_counter()-tic,1))
                results.append(record);print(json.dumps(record),flush=True)
                pd.DataFrame(results).to_csv(HERE/'leaf_booster_results.csv',index=False)
                np.savez_compressed(HERE/'leaf_booster_predictions.npz',**saved)
    print(pd.DataFrame(results).pivot(index=['family','configuration'],columns='train_end',values='auc').to_string(),flush=True)

if __name__=='__main__':main()
