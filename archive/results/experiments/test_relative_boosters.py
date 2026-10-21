"""Test within-period feature ranks and a probability-loss stopping criterion."""
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from lightgbm import early_stopping
from tune_new_boosters import make_booster, CONFIGS

HERE = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[1] / 'Data Set'
FEATURES = [f'feat_{i}' for i in range(1, 166)]
train = pd.read_csv(DATA / 'Train Data.csv')
test = pd.read_csv(DATA / 'Test Data.csv')
rows = pd.concat([train.drop(columns='label'), test], ignore_index=True)
rank_features = rows.groupby('time_step')[FEATURES].rank(pct=True, method='average')
rank_features.columns = [f'rank_{i}' for i in range(1, 166)]
features = pd.concat([rows[FEATURES], rank_features], axis=1)
known_mask = train.label.notna()
known_positions = np.flatnonzero(known_mask.to_numpy())
known = train.loc[known_mask].copy()
known['row_position'] = known_positions
known['label'] = known.label.astype(int)
FEATURE_OPTIONS = {'raw': FEATURES, 'rank': rank_features.columns.tolist(), 'combined': features.columns.tolist()}

def select_and_fit(family, representation, criterion, learning, configuration_name='balanced63'):
    configuration = CONFIGS[configuration_name]
    columns = FEATURE_OPTIONS[representation]
    end = int(learning.time_step.max())
    fit = learning.loc[learning.time_step <= end - 3]
    stop = learning.loc[learning.time_step > end - 3]
    weight = configuration['weight']
    if weight == 'ratio':
        weight = float((fit.label == 0).sum() / (fit.label == 1).sum())
    model = make_booster(family, configuration, 1000, weight=weight, stopping=True)
    if family == 'xgboost':
        model.set_params(eval_metric='logloss' if criterion == 'loss' else 'auc')
    else:
        model.set_params(metric='binary_logloss' if criterion == 'loss' else 'auc')
    fit_X = features.loc[fit.row_position, columns]
    stop_X = features.loc[stop.row_position, columns]
    with threadpool_limits(limits=4):
        if family == 'xgboost':
            model.fit(fit_X, fit.label, eval_set=[(stop_X, stop.label)], verbose=False)
            rounds = int(model.best_iteration + 1)
        else:
            model.fit(fit_X, fit.label, eval_X=stop_X, eval_y=stop.label,
                      callbacks=[early_stopping(60, verbose=False)])
            rounds = int(model.best_iteration_)
        if configuration['weight'] == 'ratio':
            weight = float((learning.label == 0).sum() / (learning.label == 1).sum())
        refitted = make_booster(family, configuration, rounds, weight=weight)
        refitted.fit(features.loc[learning.row_position, columns], learning.label)
    return refitted, rounds, columns

def main():
    results=[];saved={}
    options=[('raw','loss','balanced63'),('combined','auc','balanced63'),
             ('rank','auc','balanced63'),('combined','loss','balanced63'),
             ('rank','loss','balanced63'),('combined','loss','coarse63'),
             ('combined','loss','smooth127')]
    for family in ['xgboost','lightgbm']:
        for representation, criterion, configuration_name in options:
            name=f'{representation}_{criterion}_{configuration_name}'
            for end,start,stop in [(14,15,28),(21,22,35)]:
                learning=known.loc[known.time_step<=end]
                checking=known.loc[known.time_step.between(start,stop)]
                tic=time.perf_counter()
                model,rounds,columns=select_and_fit(family,representation,criterion,learning,configuration_name)
                with threadpool_limits(limits=4):
                    probabilities=model.predict_proba(features.loc[checking.row_position,columns])[:,1]
                saved[f'{family}_{name}_{end}']=probabilities
                record=dict(family=family,configuration=name,train_end=end,rounds=rounds,
                            auc=float(roc_auc_score(checking.label,probabilities)),seconds=round(time.perf_counter()-tic,1))
                results.append(record);print(json.dumps(record),flush=True)
                pd.DataFrame(results).to_csv(HERE/'relative_booster_results.csv',index=False)
                np.savez_compressed(HERE/'relative_booster_predictions.npz',**saved)
    print(pd.DataFrame(results).pivot(index=['family','configuration'],columns='train_end',values='auc').to_string(),flush=True)

if __name__=='__main__':main()
