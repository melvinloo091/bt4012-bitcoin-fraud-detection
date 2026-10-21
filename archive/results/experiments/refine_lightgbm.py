"""Small refinement around the strongest LightGBM coarse-bin result."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
FEATURES=[f'feat_{i}' for i in range(1,166)]
known=pd.read_csv(Path(__file__).resolve().parents[1] / 'Data Set' / 'Train Data.csv').dropna(subset=['label']).copy()
known['label']=known.label.astype(int)
CONFIGS={
    'bin31_leaf7':dict(max_bin=31,num_leaves=7,min_child_samples=30,reg_lambda=10,weight=1),
    'bin31_leaf15_smooth':dict(max_bin=31,num_leaves=15,min_child_samples=60,reg_lambda=20,weight=1),
    'bin31_leaf7_weight2':dict(max_bin=31,num_leaves=7,min_child_samples=30,reg_lambda=10,weight=2),
    'bin31_leaf7_balanced':dict(max_bin=31,num_leaves=7,min_child_samples=30,reg_lambda=10,weight='ratio'),
    'bin31_leaf15_smooth_weight2':dict(max_bin=31,num_leaves=15,min_child_samples=60,reg_lambda=20,weight=2),
    'bin63_leaf7':dict(max_bin=63,num_leaves=7,min_child_samples=30,reg_lambda=10,weight=1),
}

def make(cfg,rounds,weight):
    return LGBMClassifier(n_estimators=rounds,learning_rate=.05,max_bin=cfg['max_bin'],num_leaves=cfg['num_leaves'],
        min_child_samples=cfg['min_child_samples'],reg_lambda=cfg['reg_lambda'],scale_pos_weight=weight,
        objective='binary',metric='auc',subsample=1.,colsample_bytree=1.,deterministic=True,
        force_col_wise=True,n_jobs=4,random_state=42,verbosity=-1)

def fit(cfg,learning):
    end=int(learning.time_step.max());a=learning.loc[learning.time_step<=end-3];b=learning.loc[learning.time_step>end-3]
    weight=cfg['weight']
    if weight=='ratio':weight=float((a.label==0).sum()/(a.label==1).sum())
    m=make(cfg,400,weight)
    with threadpool_limits(limits=4):
        m.fit(a[FEATURES],a.label,eval_X=b[FEATURES],eval_y=b.label,callbacks=[early_stopping(60,verbose=False)])
        rounds=int(m.best_iteration_)
        if cfg['weight']=='ratio':weight=float((learning.label==0).sum()/(learning.label==1).sum())
        final=make(cfg,rounds,weight);final.fit(learning[FEATURES],learning.label)
    return final,rounds

def main():
    results=[];saved={}
    for name,cfg in CONFIGS.items():
        for end,start,stop in [(14,15,28),(21,22,35)]:
            learning=known.loc[known.time_step<=end];checking=known.loc[known.time_step.between(start,stop)]
            model,rounds=fit(cfg,learning)
            with threadpool_limits(limits=4):scores=model.predict_proba(checking[FEATURES])[:,1]
            saved[f'lightgbm_{name}_{end}']=scores
            record=dict(family='lightgbm',configuration=name,train_end=end,rounds=rounds,auc=float(roc_auc_score(checking.label,scores)))
            results.append(record);print(json.dumps(record),flush=True)
            pd.DataFrame(results).to_csv(HERE/'lightgbm_refinement.csv',index=False)
            np.savez_compressed(HERE/'lightgbm_refinement_predictions.npz',**saved)
    print(pd.DataFrame(results).pivot(index='configuration',columns='train_end',values='auc').to_string(),flush=True)

if __name__=='__main__':main()
