"""Proposed fixed comparison. Default execution checks data without fitting models.

After the student's modeling choice, --run --choice both tests the two declared
shorter-training variants. --choice lgb_only tests just the LightGBM change.
There is no automatic candidate selection, follow-on search or Kaggle upload.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost
import lightgbm
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

VERSION = 'fixed-half4-training-length-v1'
FEATURES = [f'feat_{i}' for i in range(1, 166)]
HALF_LIFE = 4
FOLDS = [(k, k + 1, k + 14, 'original_14_period') for k in range(7, 22)]
FOLDS += [(k, k + 1, 35, 'later_cutoff') for k in (24, 28, 31)]
XGB_SETTINGS = dict(learning_rate=.05, tree_method='hist', grow_policy='lossguide',
                    max_depth=0, max_leaves=15, max_bin=127, min_child_weight=1,
                    reg_lambda=1, objective='binary:logistic', n_jobs=2, random_state=42)
LGB_SETTINGS = dict(learning_rate=.05, num_leaves=15, max_bin=63, min_child_samples=5,
                    min_child_weight=.001, reg_lambda=1, objective='binary',
                    deterministic=True, force_col_wise=True, verbosity=-1,
                    n_jobs=2, random_state=42)
RECIPES = {'reference_200_150': (200, 150), 'short_lgb_200_100': (200, 100),
           'short_both_100_100': (100, 100)}
EXPECTED_HASHES = {
    'Train Data.csv': '6f7e56883f0aa27147262a0d71954ad64702ca57f16fbf9ed58fae21a66dd706',
    'Test Data.csv': '344e47c8cd8815f4072d7f6ecbecee8919ab648294a8eb4ed032a026c3eb91db',
    'Sample Submission/BT4012 Competition Submission.csv':
        '3dba74ad6151053aa21f366fb9faf6da7393969a0ba6f8a2ebcec561871f739e',
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_provided_data(data_dir):
    hashes = {name: digest(data_dir / name) for name in EXPECTED_HASHES}
    assert hashes == EXPECTED_HASHES, 'Provided data differs from the reviewed reference.'
    train = pd.read_csv(data_dir / 'Train Data.csv')
    test = pd.read_csv(data_dir / 'Test Data.csv')
    sample = pd.read_csv(data_dir / 'Sample Submission/BT4012 Competition Submission.csv')
    labels = pd.to_numeric(train.label, errors='coerce')
    known = train.loc[labels.notna()].copy()
    known['label'] = labels.loc[labels.notna()].astype(int)
    assert known.label.isin([0, 1]).all() and len(known) == 31235
    assert len(train) == 141772 and len(test) == len(sample) == 15329
    assert known.txId.is_unique and test['index'].is_unique
    assert sample['index'].equals(test['index'])
    assert np.isfinite(train[FEATURES].to_numpy()).all()
    assert np.isfinite(test[FEATURES].to_numpy()).all()
    return train, known, test, sample, hashes

def fitting_features(learning, cutoff):
    # Repeat the original training-only filter inside each fitting period.
    filter_rows = learning.loc[learning.time_step <= cutoff - 3]
    assert len(filter_rows) > 1 and filter_rows.time_step.max() <= cutoff - 3
    correlations = filter_rows[FEATURES].corrwith(filter_rows.time_step, method='spearman')
    removed = correlations.index[correlations.abs() > .9].tolist()
    return [c for c in FEATURES if c not in removed], removed

def component_prediction(kind, trees, learning, checking, columns, cutoff, end,
                         check_ids, hashes, output, is_final=False):
    settings = XGB_SETTINGS if kind == 'xgb' else LGB_SETTINGS
    receipt = dict(version=VERSION, data_hashes=hashes, kind=kind, trees=trees,
                   settings=settings, columns=columns, cutoff=cutoff, check_end=end,
                   half_life=HALF_LIFE, is_final=is_final,
                   versions=dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                                 lightgbm=lightgbm.__version__, numpy=np.__version__))
    key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    cache = output / 'component_predictions'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'{key}.npz'
    receipt_path = cache / f'{key}.json'
    if path.exists() and receipt_path.exists():
        assert json.loads(receipt_path.read_text()) == receipt
        with np.load(path, allow_pickle=False) as saved:
            assert np.array_equal(saved['checking_ids'], check_ids)
            assert np.array_equal(saved['learning_ids'], learning.txId.to_numpy())
            assert np.array_equal(saved['learning_labels'], learning.label.to_numpy())
            probabilities = saved['probabilities'].copy()
        return probabilities
    weights = (.5 ** ((learning.time_step.max() - learning.time_step) / HALF_LIFE)).to_numpy()
    weights /= weights.mean()
    model = (XGBClassifier if kind == 'xgb' else LGBMClassifier)(n_estimators=trees, **settings)
    model.fit(learning[columns], learning.label, sample_weight=weights)
    probabilities = model.predict_proba(checking[columns])[:, 1]
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    if is_final:
        models = output / 'models'
        models.mkdir(exist_ok=True)
        if kind == 'xgb':
            model_path = models / f'xgb_{trees}_half4.json'
            model.save_model(model_path)
            reloaded = XGBClassifier()
            reloaded.load_model(model_path)
            reload_probabilities = reloaded.predict_proba(checking[columns])[:, 1]
        else:
            model_path = models / f'lgb_{trees}_half4.txt'
            model.booster_.save_model(str(model_path))
            reloaded = lightgbm.Booster(model_file=str(model_path))
            reload_probabilities = reloaded.predict(checking[columns])
        assert np.allclose(probabilities, reload_probabilities, atol=1e-7, rtol=0)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, probabilities=probabilities, checking_ids=check_ids,
                        learning_ids=learning.txId.to_numpy(), learning_labels=learning.label.to_numpy())
    temporary.replace(path)
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return probabilities

def run_fixed_comparison(data_dir, output, choice='pending', run=False):
    train, known, test, sample, hashes = load_provided_data(data_dir)
    print(f'Provided rows checked: {len(known):,} confirmed, {len(train)-len(known):,} unknown; {len(test):,} test.')
    print('Reference: 200 XGBoost / 150 LightGBM trees, half-life 4; public AUC 0.95471.')
    print('Proposed variants: 200/100 and 100/100. No hidden-score forecast.')
    if not run:
        print('PREPARED ONLY: no models fitted, no prediction CSVs generated.')
        return dict(status='prepared_not_trained', rows_confirmed=len(known),
                    rows_unknown=len(train)-len(known), test_rows=len(test), choice=choice)
    if choice not in ('both', 'lgb_only'):
        raise ValueError('Record the student modeling choice before fitting this proposal.')
    chosen = RECIPES.copy()
    if choice == 'lgb_only':
        chosen.pop('short_both_100_100')
    output.mkdir(parents=True, exist_ok=True)
    declared = dict(version=VERSION, student_choice=choice, recipes=chosen, folds=FOLDS,
                    xgboost=XGB_SETTINGS, lightgbm=LGB_SETTINGS, half_life=HALF_LIFE,
                    blend_weights=[.5, .5], data_hashes=hashes, automatic_selection=False,
                    external_data=False, kaggle_upload=False,
                    feature_selection='Known fitting rows through cutoff minus three; abs Spearman > .9 removed.')
    (output / 'declared_recipe.json').write_text(json.dumps(declared, indent=2))
    started = time.monotonic()
    rows, per_step, masks = [], [], []
    for cutoff, first, last, group in FOLDS:
        learning = known.loc[known.time_step <= cutoff]
        checking = known.loc[known.time_step.between(first, last)]
        assert learning.time_step.max() < checking.time_step.min()
        xcolumns, removed = fitting_features(learning, cutoff)
        masks.append(dict(cutoff=cutoff, removed=removed, filter_fit_end=cutoff-3))
        components = {}
        for kind, counts, columns in [('xgb', {p[0] for p in chosen.values()}, xcolumns),
                                     ('lgb', {p[1] for p in chosen.values()}, FEATURES)]:
            for trees in sorted(counts):
                components[(kind, trees)] = component_prediction(
                    kind, trees, learning, checking, columns, cutoff, last,
                    checking.txId.to_numpy(), hashes, output)
        for name, (xtrees, ltrees) in chosen.items():
            p = .5*components[('xgb', xtrees)] + .5*components[('lgb', ltrees)]
            rows.append(dict(cutoff=cutoff, check_first=first, check_last=last, group=group,
                             recipe=name, training_rows=len(learning), checking_rows=len(checking),
                             auc=float(roc_auc_score(checking.label, p))))
            for step in checking.time_step.unique():
                selected = checking.time_step.to_numpy() == step
                y = checking.label.to_numpy()[selected]
                per_step.append(dict(cutoff=cutoff, step=int(step), recipe=name,
                                     rows=int(selected.sum()), positives=int(y.sum()),
                                     auc=float(roc_auc_score(y, p[selected])) if len(np.unique(y)) == 2 else None))
        pd.DataFrame(rows).to_csv(output / 'validation_results.csv', index=False)
        (output / 'progress.json').write_text(json.dumps(dict(phase='validation', last_cutoff=cutoff,
                                      completed_checks=len(masks), elapsed_seconds=time.monotonic()-started)))
        print(f'Completed training through {cutoff}, checking {first}-{last}.', flush=True)
    scores = pd.DataFrame(rows)
    summaries = []
    views = {'original_15': scores.group.eq('original_14_period'),
             'original_later_seven': scores.cutoff.between(15, 21),
             'later_cutoffs_24_28_31': scores.group.eq('later_cutoff')}
    for view, selected in views.items():
        frame = scores.loc[selected]
        control = frame.loc[frame.recipe.eq('reference_200_150')].set_index('cutoff').auc
        for name in chosen:
            values = frame.loc[frame.recipe.eq(name)].set_index('cutoff').auc
            delta = values-control
            summaries.append(dict(group=view, recipe=name, checks=len(values), mean_auc=values.mean(),
                                  worst_auc=values.min(), mean_delta_vs_reference=delta.mean(),
                                  wins_vs_reference=int((delta > 0).sum())))
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(per_step).to_csv(output / 'per_step_auc.csv', index=False)
    (output / 'training_only_feature_masks.json').write_text(json.dumps(masks, indent=2))
    xcolumns, removed = fitting_features(known, 35)
    assert set(removed) == {'feat_100','feat_101','feat_103','feat_136','feat_137','feat_139'}
    components = {}
    for kind, counts, columns in [('xgb', {p[0] for p in chosen.values()}, xcolumns),
                                 ('lgb', {p[1] for p in chosen.values()}, FEATURES)]:
        for trees in sorted(counts):
            components[(kind, trees)] = component_prediction(
                kind, trees, known, test, columns, 35, 49, test['index'].to_numpy(), hashes, output, True)
    submission_hashes = {}
    for name, (xtrees, ltrees) in chosen.items():
        submission = sample[['index']].copy()
        p = .5*components[('xgb', xtrees)] + .5*components[('lgb', ltrees)]
        submission['target'] = p
        assert submission['index'].equals(sample['index']) and submission.target.between(0, 1).all()
        path = output / f'submission_half4_{name}_experimental.csv'
        submission.to_csv(path, index=False)
        checked = pd.read_csv(path)
        assert checked['index'].equals(sample['index']) and np.allclose(checked.target, p, atol=1e-15, rtol=0)
        submission_hashes[name] = digest(path)
        if name == 'reference_200_150':
            old_path = data_dir.parent / 'claude_experiments/submissions/submission_lowreg_pair_recent4.csv'
            if old_path.exists():
                old = pd.read_csv(old_path)
                assert old['index'].equals(submission['index'])
                assert np.max(np.abs(old.target-p)) <= 1e-7
    assert {name: digest(data_dir/name) for name in EXPECTED_HASHES} == hashes
    (output / 'progress.json').write_text(json.dumps(dict(phase='complete', checks=len(FOLDS),
                              elapsed_seconds=time.monotonic()-started, submission_hashes=submission_hashes,
                              final_selection='student decision', kaggle_upload=False), indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print('Fixed comparison complete; review results before selecting a submission.')
    return summaries

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path.cwd()/'Data Set')
    parser.add_argument('--output-dir', type=Path, default=Path.cwd()/'half4_training_length_results')
    parser.add_argument('--choice', choices=['pending', 'both', 'lgb_only'], default='pending')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    return run_fixed_comparison(args.data_dir, args.output_dir, args.choice, args.run)

if __name__ == '__main__':
    main()
