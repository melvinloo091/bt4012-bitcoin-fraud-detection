"""Targeted XGBoost / LightGBM search with an internal chronological stop set."""
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
DATA = Path(__file__).resolve().parents[1] / 'Data Set'
FEATURES = [f'feat_{i}' for i in range(1, 166)]
known = pd.read_csv(DATA / 'Train Data.csv').dropna(subset=['label']).copy()
known['label'] = known['label'].astype(int)
CHECKS = [(14, 15, 28), (21, 22, 35), (28, 29, 35)]

# Restrained variants target the transfer problem observed in previous trials.
CONFIGS = {
    'coarse63': dict(max_bin=63, depth=3, minimum=10, regularization=20, sampling=1., weight=1.),
    'coarse127': dict(max_bin=127, depth=3, minimum=10, regularization=20, sampling=1., weight=1.),
    'smooth63': dict(max_bin=63, depth=4, minimum=20, regularization=40, sampling=.85, weight=1.),
    'smooth127': dict(max_bin=127, depth=4, minimum=20, regularization=40, sampling=.85, weight=1.),
    'weighted63': dict(max_bin=63, depth=3, minimum=10, regularization=20, sampling=1., weight=2.),
    'balanced63': dict(max_bin=63, depth=3, minimum=10, regularization=20, sampling=1., weight='ratio'),
    'very_coarse31': dict(max_bin=31, depth=4, minimum=10, regularization=20, sampling=1., weight=1.),
    'shallow63': dict(max_bin=63, depth=2, minimum=10, regularization=20, sampling=1., weight=1.),
}

def make_booster(family, configuration, rounds, *, weight=1., stopping=False):
    common = dict(n_estimators=int(rounds), learning_rate=.03, max_bin=configuration['max_bin'],
                  subsample=configuration['sampling'], colsample_bytree=configuration['sampling'],
                  reg_lambda=configuration['regularization'], scale_pos_weight=weight,
                  n_jobs=4, random_state=42)
    if family == 'xgboost':
        return XGBClassifier(**common, max_depth=configuration['depth'],
                             min_child_weight=configuration['minimum'], objective='binary:logistic',
                             eval_metric='auc', tree_method='hist',
                             early_stopping_rounds=60 if stopping else None)
    # LightGBM's minimum is a row count, unlike XGBoost's Hessian sum.
    return LGBMClassifier(**common, num_leaves=min(15, 2 ** configuration['depth']),
                          max_depth=configuration['depth'],
                          min_child_samples=configuration['minimum'] * 5,
                          subsample_freq=1 if configuration['sampling'] < 1 else 0,
                          objective='binary', metric='auc', deterministic=True,
                          force_col_wise=True, verbosity=-1)

def choose_rounds_and_refit(family, configuration, learning):
    train_end = int(learning.time_step.max())
    stop_start = train_end - 2
    fitting = learning.loc[learning.time_step < stop_start]
    stopping = learning.loc[learning.time_step >= stop_start]
    assert fitting.label.nunique() == stopping.label.nunique() == 2
    assert fitting.time_step.max() < stopping.time_step.min()
    weight = configuration['weight']
    if weight == 'ratio':
        weight = float((fitting.label == 0).sum() / (fitting.label == 1).sum())
    model = make_booster(family, configuration, 1000, weight=weight, stopping=True)
    with threadpool_limits(limits=4):
        if family == 'xgboost':
            model.fit(fitting[FEATURES], fitting.label,
                      eval_set=[(stopping[FEATURES], stopping.label)], verbose=False)
            rounds = model.best_iteration + 1
        else:
            model.fit(fitting[FEATURES], fitting.label,
                      eval_X=stopping[FEATURES], eval_y=stopping.label,
                      callbacks=[early_stopping(60, verbose=False)])
            rounds = model.best_iteration_
        if configuration['weight'] == 'ratio':
            weight = float((learning.label == 0).sum() / (learning.label == 1).sum())
        refitted = make_booster(family, configuration, rounds, weight=weight)
        refitted.fit(learning[FEATURES], learning.label)
    return refitted, int(rounds)

def main():
    results = []
    predictions = {}
    # The first two checks drive selection. The short final check is descriptive.
    for family in ['xgboost', 'lightgbm']:
        for name, configuration in CONFIGS.items():
            for end, start, stop in CHECKS[:2]:
                learning = known.loc[known.time_step <= end]
                checking = known.loc[known.time_step.between(start, stop)]
                tic = time.perf_counter()
                model, rounds = choose_rounds_and_refit(family, configuration, learning)
                with threadpool_limits(limits=4):
                    scores = model.predict_proba(checking[FEATURES])[:, 1]
                predictions[f'{family}_{name}_{end}'] = scores
                record = dict(family=family, configuration=name, train_end=end,
                              auc=float(roc_auc_score(checking.label, scores)), rounds=rounds,
                              seconds=round(time.perf_counter() - tic, 1))
                results.append(record)
                print(json.dumps(record), flush=True)
                pd.DataFrame(results).to_csv(HERE / 'new_booster_search.csv', index=False)
                np.savez_compressed(HERE / 'new_booster_search_predictions.npz', **predictions)
    table = pd.DataFrame(results)
    print(table.pivot(index=['family','configuration'], columns='train_end', values='auc').to_string(), flush=True)

if __name__ == '__main__':
    main()
