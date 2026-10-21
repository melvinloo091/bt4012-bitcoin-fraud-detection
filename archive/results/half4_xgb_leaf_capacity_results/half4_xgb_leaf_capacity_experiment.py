"""Student-selected, fixed XGBoost leaf-capacity comparison; no parameter search.

Reproduce from supplied train/test/sample files with --run --choice both. The
optional reference directory saves identical completed fits; it is not required.
Only the XGBoost maximum leaf count changes (15 control, 7 and 31 alternatives).
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost
import lightgbm
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

VERSION = 'fixed-half4-xgb-leaf-capacity-v1'
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
RECIPES = {'reference_leaf15': 15, 'xgb_leaf7': 7, 'xgb_leaf31': 31}
EXPECTED_HASHES = {
    'Train Data.csv': '6f7e56883f0aa27147262a0d71954ad64702ca57f16fbf9ed58fae21a66dd706',
    'Test Data.csv': '344e47c8cd8815f4072d7f6ecbecee8919ab648294a8eb4ed032a026c3eb91db',
    'Sample Submission/BT4012 Competition Submission.csv':
        '3dba74ad6151053aa21f366fb9faf6da7393969a0ba6f8a2ebcec561871f739e',
}
PUBLIC_REFERENCE_SHA256 = '7fd1ccf24d89a1ff1be9d21a273f787e1a40f6413d98a324fd92ba8767d95163'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def versions():
    return dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                lightgbm=lightgbm.__version__, numpy=np.__version__)

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
    filter_rows = learning.loc[learning.time_step <= cutoff - 3]
    assert len(filter_rows) > 1 and filter_rows.time_step.max() <= cutoff - 3
    correlations = filter_rows[FEATURES].corrwith(filter_rows.time_step, method='spearman')
    removed = correlations.index[correlations.abs() > .9].tolist()
    return [c for c in FEATURES if c not in removed], removed

def read_reference_sources(reference_dir):
    sources = {}
    if reference_dir is None:
        return sources
    verification = json.loads((reference_dir / 'independent_verification.json').read_text())
    assert verification['all_final_models_reloaded_in_fresh_interpreter']
    assert verification['feature_filter_nested_in_each_fitting_window']
    progress = json.loads((reference_dir / 'progress.json').read_text())
    assert progress['phase'] == 'complete' and progress['checks'] == 18
    reference_csv = reference_dir / 'submission_half4_long_xgb_300_150_experimental.csv'
    assert digest(reference_csv) == PUBLIC_REFERENCE_SHA256
    for path in (reference_dir / 'component_predictions').glob('*.json'):
        receipt = json.loads(path.read_text())
        if (receipt['kind'], receipt['trees']) not in [('xgb', 300), ('lgb', 150)]:
            continue
        assert path.stem == hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        key = (receipt['cutoff'], receipt['is_final'], receipt['kind'])
        assert key not in sources
        sources[key] = (path, receipt)
    assert len(sources) == 38
    return sources

def model_filename(kind, leaves):
    return f'xgb_300_leaf{leaves}_half4.json' if kind == 'xgb' else 'lgb_150_half4.txt'

def component_prediction(kind, leaves, learning, checking, columns, cutoff, end,
                         check_ids, hashes, output, sources, reference_dir, is_final=False):
    settings = dict(XGB_SETTINGS, max_leaves=leaves) if kind == 'xgb' else LGB_SETTINGS.copy()
    trees = 300 if kind == 'xgb' else 150
    receipt = dict(version=VERSION, data_hashes=hashes, kind=kind, trees=trees,
                   settings=settings, columns=columns, cutoff=cutoff, check_end=end,
                   half_life=HALF_LIFE, is_final=is_final, versions=versions())
    key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    cache = output / 'component_predictions'
    cache.mkdir(parents=True, exist_ok=True)
    path, receipt_path = cache / f'{key}.npz', cache / f'{key}.json'
    provenance_path = cache / f'{key}.provenance.json'
    if path.exists() and receipt_path.exists() and provenance_path.exists():
        assert json.loads(receipt_path.read_text()) == receipt
        with np.load(path, allow_pickle=False) as saved:
            assert np.array_equal(saved['checking_ids'], check_ids)
            assert np.array_equal(saved['learning_ids'], learning.txId.to_numpy())
            assert np.array_equal(saved['learning_labels'], learning.label.to_numpy())
            probabilities = saved['probabilities'].copy()
        provenance = json.loads(provenance_path.read_text())
        assert provenance['npz_sha256'] == digest(path)
        return probabilities, provenance

    source_key = (cutoff, is_final, kind)
    reuse = source_key in sources and (kind == 'lgb' or leaves == 15)
    if reuse:
        source_path, old = sources[source_key]
        expected_old = dict(receipt, version='fixed-half4-training-length-v1')
        assert old == expected_old, 'Reference receipt is not the exact requested component.'
        source_npz = source_path.with_suffix('.npz')
        with np.load(source_npz, allow_pickle=False) as saved:
            assert np.array_equal(saved['checking_ids'], check_ids)
            assert np.array_equal(saved['learning_ids'], learning.txId.to_numpy())
            assert np.array_equal(saved['learning_labels'], learning.label.to_numpy())
            probabilities = saved['probabilities'].copy()
        retained = output / 'reference_source_components'
        retained.mkdir(exist_ok=True)
        shutil.copy2(source_path, retained / source_path.name)
        shutil.copy2(source_npz, retained / source_npz.name)
        shutil.copy2(source_npz, path)
        provenance = dict(mode='reused_verified_reference', source_receipt=source_path.name,
                          source_receipt_sha256=digest(source_path), source_npz=source_npz.name,
                          source_npz_sha256=digest(source_npz))
        if is_final:
            source_model = reference_dir / 'models' / ('xgb_300_half4.json' if kind == 'xgb' else 'lgb_150_half4.txt')
            old_metadata = json.loads((reference_dir / 'experiment_metadata.json').read_text())
            assert digest(source_model) == old_metadata['saved_model_hashes'][source_model.name]
            models = output / 'models'
            models.mkdir(exist_ok=True)
            shutil.copy2(source_model, models / model_filename(kind, leaves))
            provenance['source_model_sha256'] = digest(source_model)
    else:
        weights = (.5 ** ((learning.time_step.max() - learning.time_step) / HALF_LIFE)).to_numpy()
        weights /= weights.mean()
        model = (XGBClassifier if kind == 'xgb' else LGBMClassifier)(n_estimators=trees, **settings)
        model.fit(learning[columns], learning.label, sample_weight=weights)
        probabilities = model.predict_proba(checking[columns])[:, 1]
        if is_final:
            models = output / 'models'
            models.mkdir(exist_ok=True)
            model_path = models / model_filename(kind, leaves)
            if kind == 'xgb':
                model.save_model(model_path)
            else:
                model.booster_.save_model(str(model_path))
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, probabilities=probabilities, checking_ids=check_ids,
                            learning_ids=learning.txId.to_numpy(), learning_labels=learning.label.to_numpy())
        temporary.replace(path)
        provenance = dict(mode='fresh_fit')
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert len(probabilities) == len(check_ids)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    provenance.update(component_key=key, kind=kind, leaves=leaves if kind == 'xgb' else None,
                      cutoff=cutoff, is_final=is_final, npz_sha256=digest(path))
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    return probabilities, provenance

def run_fixed_comparison(data_dir, output, reference_dir=None, choice='pending', run=False):
    train, known, test, sample, hashes = load_provided_data(data_dir)
    print(f'Provided rows: {len(known):,} confirmed, {len(train)-len(known):,} unknown; {len(test):,} test.')
    print('Control: XGBoost 300 trees / 15 leaves; LightGBM 150 trees; half-life 4; public AUC 0.95484.')
    print('Selected changes: maximum XGBoost leaves 7 and 31. Every other setting stays fixed.')
    if not run:
        print('Prepared only: no model fitting or prediction CSV creation.')
        return dict(status='prepared_not_trained')
    if choice not in ('both', 'leaf31_only'):
        raise ValueError('The student must choose a declared comparison before fitting.')
    chosen = RECIPES.copy()
    if choice == 'leaf31_only':
        chosen.pop('xgb_leaf7')
    output.mkdir(parents=True, exist_ok=True)
    declared = dict(version=VERSION, student_choice=choice,
        student_choice_verbatim='Test both 7 and 31 leaves (Recommended)' if choice == 'both' else 'Test 31 leaves only',
        recipes=chosen, folds=FOLDS, xgboost=XGB_SETTINGS, lightgbm=LGB_SETTINGS,
        xgboost_trees=300, lightgbm_trees=150, half_life=4, blend_weights=[.5, .5],
        data_hashes=hashes, versions=versions(), pandas_version=pd.__version__,
        current_verified_public_best=.95484, candidate_public_auc=None,
        feature_selection='Known fitting rows through cutoff minus three; abs Spearman > .9 removed.',
        parameter_search=False, blend_weight_search=False, automatic_selection=False,
        external_data=False, unknown_rows_given_labels=False, future_rows_used_for_fitting=False,
        codex_kaggle_upload=False, final_selection='student decision')
    declared_path = output / 'declared_recipe.json'
    if declared_path.exists():
        assert json.loads(declared_path.read_text()) == declared
    else:
        declared_path.write_text(json.dumps(declared, indent=2) + '\n')
    sources = read_reference_sources(reference_dir)
    started = time.monotonic()
    rows, per_step, masks, provenance = [], [], [], []
    for cutoff, first, last, group in FOLDS:
        learning = known.loc[known.time_step <= cutoff]
        checking = known.loc[known.time_step.between(first, last)]
        assert learning.time_step.max() < checking.time_step.min()
        assert not set(learning.txId).intersection(checking.txId)
        xcolumns, removed = fitting_features(learning, cutoff)
        masks.append(dict(cutoff=cutoff, removed=removed, filter_fit_end=cutoff-3))
        lp, receipt = component_prediction('lgb', 15, learning, checking, FEATURES, cutoff, last,
            checking.txId.to_numpy(), hashes, output, sources, reference_dir)
        provenance.append(receipt)
        for name, leaves in chosen.items():
            xp, receipt = component_prediction('xgb', leaves, learning, checking, xcolumns, cutoff, last,
                checking.txId.to_numpy(), hashes, output, sources, reference_dir)
            provenance.append(receipt)
            p = .5*xp + .5*lp
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
            completed_checks=len(masks), elapsed_seconds=time.monotonic()-started), indent=2) + '\n')
        print(f'Completed cutoff {cutoff}, checking steps {first}-{last}.', flush=True)
    scores = pd.DataFrame(rows)
    summaries = []
    groups = {'original_15': list(range(7,22)), 'original_later_seven': list(range(15,22)),
              'later_cutoffs_24_28_31': [24,28,31]}
    for group, cutoffs in groups.items():
        frame = scores.loc[scores.cutoff.isin(cutoffs)]
        control = frame.loc[frame.recipe.eq('reference_leaf15')].set_index('cutoff').auc
        for name in chosen:
            values = frame.loc[frame.recipe.eq(name)].set_index('cutoff').auc
            delta = values-control
            summaries.append(dict(group=group, recipe=name, checks=len(values), mean_auc=values.mean(),
                worst_auc=values.min(), mean_delta_vs_reference=delta.mean(),
                wins_vs_reference=int((delta > 0).sum())))
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(per_step).to_csv(output / 'per_step_auc.csv', index=False)
    (output / 'training_only_feature_masks.json').write_text(json.dumps(masks, indent=2) + '\n')
    xcolumns, removed = fitting_features(known, 35)
    assert set(removed) == {'feat_100','feat_101','feat_103','feat_136','feat_137','feat_139'}
    lp, receipt = component_prediction('lgb',15,known,test,FEATURES,35,49,test['index'].to_numpy(),
        hashes,output,sources,reference_dir,True)
    provenance.append(receipt)
    submission_hashes = {}
    for name, leaves in chosen.items():
        xp, receipt = component_prediction('xgb',leaves,known,test,xcolumns,35,49,test['index'].to_numpy(),
            hashes,output,sources,reference_dir,True)
        provenance.append(receipt)
        submission = sample[['index']].copy()
        p = .5*xp + .5*lp
        submission['target'] = p
        path = output / f'submission_half4_{name}_300_150_experimental.csv'
        submission.to_csv(path, index=False)
        checked = pd.read_csv(path)
        assert checked.columns.tolist() == ['index', 'target']
        assert checked['index'].equals(sample['index']) and checked['index'].is_unique
        assert np.isfinite(checked.target).all() and checked.target.between(0,1).all()
        assert np.max(np.abs(checked.target-p)) <= 1e-15
        submission_hashes[name] = digest(path)
        if name == 'reference_leaf15' and reference_dir is not None:
            assert digest(path) == PUBLIC_REFERENCE_SHA256, 'Control CSV must be byte-identical to 0.95484 submission.'
    assert {name: digest(data_dir/name) for name in EXPECTED_HASHES} == hashes
    fresh = sum(p['mode']=='fresh_fit' for p in provenance)
    reused = sum(p['mode']=='reused_verified_reference' for p in provenance)
    (output / 'reference_reuse_manifest.json').write_text(json.dumps(dict(
        source=str(reference_dir) if reference_dir else None,
        reused_reference_components=reused, fresh_component_fits=fresh,
        components=provenance), indent=2) + '\n')
    metadata = dict(declared, known_training_rows=len(known), unknown_rows_excluded=len(train)-len(known),
        test_rows=len(test), fresh_component_fits=fresh, reused_reference_components=reused,
        elapsed_seconds=time.monotonic()-started,
        saved_model_hashes={p.name:digest(p) for p in (output/'models').iterdir() if p.is_file()},
        validation_limitations=['Overlapping checks and previously inspected development labels.',
            'Later checks have 11, 7 and 4 periods; test spans 14.',
            'Public subset already used to compare multiple recipes; private labels unavailable.'])
    (output / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'progress.json').write_text(json.dumps(dict(phase='complete', checks=len(FOLDS),
        elapsed_seconds=metadata['elapsed_seconds'], fresh_component_fits=fresh,
        reused_reference_components=reused, submission_hashes=submission_hashes,
        final_selection='student decision', codex_kaggle_upload=False), indent=2) + '\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f'Completed fixed comparison: {fresh} new fits; {reused} identical reference components reused.')
    print('Candidate public scores unknown. No automatic winner selection or Kaggle upload.')
    return summaries

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path.cwd()/'Data Set')
    parser.add_argument('--output-dir', type=Path, default=Path.cwd()/'half4_xgb_leaf_capacity_results')
    parser.add_argument('--reference-dir', type=Path, default=None)
    parser.add_argument('--choice', choices=['pending','both','leaf31_only'], default='pending')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    return run_fixed_comparison(args.data_dir,args.output_dir,args.reference_dir,args.choice,args.run)

if __name__ == '__main__':
    main()
