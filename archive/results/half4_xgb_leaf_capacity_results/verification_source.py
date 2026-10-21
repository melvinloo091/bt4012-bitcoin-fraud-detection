"""Independently recalculate selected leaf-capacity results; never fit a model."""
from pathlib import Path
import hashlib
import json

import nbformat
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import Booster

HERE = Path(__file__).resolve().parent
PROJECT = Path('/Users/mlvlo/BT4012/Kaggle')
OUT = HERE/'half4_xgb_leaf_capacity_results'
NOTEBOOK = HERE/'e1156285_half4_xgb_leaf_capacity.ipynb'
SOURCE = HERE/'half4_xgb_leaf_capacity_experiment.py'
REFERENCE = PROJECT/'archive/results/half4_longer_training_results'
FEATURES = [f'feat_{i}' for i in range(1,166)]
RECIPES = {'reference_leaf15':15,'xgb_leaf7':7,'xgb_leaf31':31}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

declared = json.loads((OUT/'declared_recipe.json').read_text())
progress = json.loads((OUT/'progress.json').read_text())
manifest = json.loads((OUT/'execution_manifest.json').read_text())
metadata = json.loads((OUT/'experiment_metadata.json').read_text())
assert declared['recipes']==RECIPES and declared['student_choice']=='both'
assert declared['student_choice_verbatim']=='Test both 7 and 31 leaves (Recommended)'
assert declared['xgboost_trees']==300 and declared['lightgbm_trees']==150
assert declared['half_life']==4 and declared['blend_weights']==[.5,.5]
assert not declared['parameter_search'] and not declared['automatic_selection']
assert not declared['external_data'] and not declared['codex_kaggle_upload']
assert not declared['unknown_rows_given_labels'] and not declared['future_rows_used_for_fitting']
assert declared['xgboost']==dict(learning_rate=.05,tree_method='hist',grow_policy='lossguide',
    max_depth=0,max_leaves=15,max_bin=127,min_child_weight=1,reg_lambda=1,
    objective='binary:logistic',n_jobs=2,random_state=42)
assert declared['lightgbm']==dict(learning_rate=.05,num_leaves=15,max_bin=63,min_child_samples=5,
    min_child_weight=.001,reg_lambda=1,objective='binary',deterministic=True,
    force_col_wise=True,verbosity=-1,n_jobs=2,random_state=42)
assert progress['phase']=='complete' and progress['checks']==18
assert progress['fresh_component_fits']==progress['reused_reference_components']==38
assert manifest['fit_operations_ast_match_previous_recipe']
assert digest(SOURCE)==manifest['source_script_sha256']==digest(OUT/SOURCE.name)
for name,expected in declared['data_hashes'].items():
    assert digest(PROJECT/'Data Set'/name)==expected
train = pd.read_csv(PROJECT/'Data Set/Train Data.csv')
labels = pd.to_numeric(train.label,errors='coerce')
known = train.loc[labels.notna()].copy()
known['label'] = labels.loc[labels.notna()].astype(int)
test = pd.read_csv(PROJECT/'Data Set/Test Data.csv')
sample = pd.read_csv(PROJECT/'Data Set/Sample Submission/BT4012 Competition Submission.csv')
assert len(known)==31235 and len(train)-len(known)==110537
assert len(test)==len(sample)==15329 and known.label.isin([0,1]).all()
assert test['index'].equals(sample['index'])
scores = pd.read_csv(OUT/'validation_results.csv').set_index(['cutoff','recipe'])
summary = pd.read_csv(OUT/'summary.csv').set_index(['group','recipe'])
periods = pd.read_csv(OUT/'per_step_auc.csv').set_index(['cutoff','step','recipe'])
masks = {item['cutoff']:item for item in json.loads((OUT/'training_only_feature_masks.json').read_text())}
components, receipts, provenances = {}, {}, {}
for path in (OUT/'component_predictions').glob('*.json'):
    if path.name.endswith('.provenance.json'):
        continue
    receipt = json.loads(path.read_text())
    assert receipt['version']==declared['version']
    assert receipt['data_hashes']==declared['data_hashes'] and receipt['versions']==declared['versions']
    key_hash = hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    assert path.stem==key_hash
    leaves = receipt['settings']['max_leaves'] if receipt['kind']=='xgb' else 15
    key = (receipt['cutoff'],receipt['is_final'],receipt['kind'],leaves)
    assert key not in components
    assert receipt['trees']==(300 if receipt['kind']=='xgb' else 150)
    assert receipt['settings']==(dict(declared['xgboost'],max_leaves=leaves) if receipt['kind']=='xgb' else declared['lightgbm'])
    assert receipt['half_life']==4
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
        components[key]={name:saved[name].copy() for name in saved.files}
    provenance = json.loads(path.with_suffix('.provenance.json').read_text())
    assert provenance['npz_sha256']==digest(path.with_suffix('.npz'))
    if provenance['mode']=='reused_verified_reference':
        assert receipt['kind']=='lgb' or leaves==15
        original_receipt = REFERENCE/'component_predictions'/provenance['source_receipt']
        original_npz = REFERENCE/'component_predictions'/provenance['source_npz']
        retained_receipt = OUT/'reference_source_components'/provenance['source_receipt']
        retained_npz = OUT/'reference_source_components'/provenance['source_npz']
        assert digest(original_receipt)==digest(retained_receipt)==provenance['source_receipt_sha256']
        assert json.loads(original_receipt.read_text())==dict(receipt,version='fixed-half4-training-length-v1')
        assert digest(original_npz)==digest(retained_npz)==digest(path.with_suffix('.npz'))==provenance['source_npz_sha256']
    else:
        assert provenance['mode']=='fresh_fit' and receipt['kind']=='xgb' and leaves in [7,31]
    receipts[key], provenances[key] = receipt, provenance
assert len(components)==76
assert sum(p['mode']=='fresh_fit' for p in provenances.values())==38
assert sum(p['mode']=='reused_verified_reference' for p in provenances.values())==38
reference_scores = pd.read_csv(REFERENCE/'validation_results.csv').set_index(['cutoff','recipe'])
appearances, component_scores, diagnostics = [], [], []
for cutoff,first,last,group in declared['folds']:
    learning = known.loc[known.time_step<=cutoff]
    checking = known.loc[known.time_step.between(first,last)]
    assert learning.time_step.max()<checking.time_step.min()
    assert not set(learning.txId).intersection(checking.txId)
    correlation = learning.loc[learning.time_step<=cutoff-3,FEATURES].corrwith(
        learning.loc[learning.time_step<=cutoff-3,'time_step'],method='spearman')
    removed = correlation.index[correlation.abs()>.9].tolist()
    assert masks[cutoff]['removed']==removed and masks[cutoff]['filter_fit_end']==cutoff-3
    columns = [c for c in FEATURES if c not in removed]
    appearances.extend(checking.txId.tolist())
    weights = (.5**((learning.time_step.max()-learning.time_step)/4)).to_numpy()
    weights /= weights.mean()
    diagnostics.append(dict(cutoff=cutoff,check_first=first,check_last=last,fit_rows=len(learning),
        fit_positives=int(learning.label.sum()),check_rows=len(checking),check_positives=int(checking.label.sum()),
        weighted_fit_illicit_share=float(np.average(learning.label,weights=weights)),
        weight_concentration_effective_count=float(weights.sum()**2/np.square(weights).sum())))
    for kind,leaves in [('xgb',7),('xgb',15),('xgb',31),('lgb',15)]:
        key = (cutoff,False,kind,leaves)
        saved = components[key]
        assert np.array_equal(saved['learning_ids'],learning.txId.to_numpy())
        assert np.array_equal(saved['learning_labels'],learning.label.to_numpy())
        assert np.array_equal(saved['checking_ids'],checking.txId.to_numpy())
        assert receipts[key]['columns']==(columns if kind=='xgb' else FEATURES)
        assert receipts[key]['check_end']==last
        p = saved['probabilities']
        assert len(p)==len(checking) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        component_scores.append(dict(cutoff=cutoff,component=f'{kind}_leaf{leaves}',auc=roc_auc_score(checking.label,p)))
    lp = components[(cutoff,False,'lgb',15)]['probabilities']
    for name,leaves in RECIPES.items():
        p = .5*components[(cutoff,False,'xgb',leaves)]['probabilities']+.5*lp
        row = scores.loc[(cutoff,name)]
        assert row.training_rows==len(learning) and row.checking_rows==len(checking)
        assert abs(row.auc-roc_auc_score(checking.label,p))<1e-14
        if name=='reference_leaf15':
            assert row.auc==reference_scores.loc[(cutoff,'long_xgb_300_150'),'auc']
        for step in checking.time_step.unique():
            selected = checking.time_step.to_numpy()==step
            y = checking.label.to_numpy()[selected]
            r = periods.loc[(cutoff,int(step),name)]
            assert r.rows==selected.sum() and r.positives==y.sum()
            if len(np.unique(y))==2:
                assert abs(r.auc-roc_auc_score(y,p[selected]))<1e-14
            else:
                assert pd.isna(r.auc)
groups = {'original_15':list(range(7,22)),'original_later_seven':list(range(15,22)),
          'later_cutoffs_24_28_31':[24,28,31]}
for group,cutoffs in groups.items():
    control = scores.loc[[(k,'reference_leaf15') for k in cutoffs],'auc'].to_numpy()
    for name in RECIPES:
        values = scores.loc[[(k,name) for k in cutoffs],'auc'].to_numpy()
        row = summary.loc[(group,name)]
        assert row.checks==len(cutoffs) and abs(row.mean_auc-values.mean())<1e-14
        assert abs(row.worst_auc-values.min())<1e-14
        assert abs(row.mean_delta_vs_reference-(values-control).mean())<1e-14
        assert row.wins_vs_reference==(values>control).sum()
columns = receipts[(35,True,'xgb',15)]['columns']
assert set(FEATURES)-set(columns)=={'feat_100','feat_101','feat_103','feat_136','feat_137','feat_139'}
actual_models = {}
def leaf_count(tree):
    return 1 if 'leaf' in tree else sum(leaf_count(child) for child in tree['children'])
for kind,leaves in [('xgb',7),('xgb',15),('xgb',31),('lgb',15)]:
    key = (35,True,kind,leaves)
    saved = components[key]
    assert np.array_equal(saved['learning_ids'],known.txId.to_numpy())
    assert np.array_equal(saved['learning_labels'],known.label.to_numpy())
    assert np.array_equal(saved['checking_ids'],test['index'].to_numpy())
    assert receipts[key]['columns']==(columns if kind=='xgb' else FEATURES)
    if kind=='xgb':
        path = OUT/'models'/f'xgb_300_leaf{leaves}_half4.json'
        model = XGBClassifier(n_jobs=2)
        model.load_model(path)
        p = model.predict_proba(test[columns])[:,1]
        count = model.get_booster().num_boosted_rounds()
        counts = [leaf_count(json.loads(tree)) for tree in model.get_booster().get_dump(dump_format='json')]
        assert count==300 and len(counts)==300 and max(counts)<=leaves
        details = dict(trees=count,max_actual_leaves=max(counts),min_actual_leaves=min(counts),mean_actual_leaves=float(np.mean(counts)))
    else:
        path = OUT/'models'/'lgb_150_half4.txt'
        model = Booster(model_file=str(path))
        p = model.predict(test[FEATURES],num_threads=2)
        assert model.current_iteration()==150
        details = dict(trees=150)
    error = float(np.max(np.abs(p-saved['probabilities'])))
    assert error<=1e-7
    assert digest(path)==metadata['saved_model_hashes'][path.name]
    actual_models[path.name]=dict(details,reloaded_probability_max_error=error,sha256=digest(path))
reference_path = REFERENCE/'submission_half4_long_xgb_300_150_experimental.csv'
assert digest(reference_path)=='7fd1ccf24d89a1ff1be9d21a273f787e1a40f6413d98a324fd92ba8767d95163'
old = pd.read_csv(reference_path)
test_diagnostics = {}
lp = components[(35,True,'lgb',15)]['probabilities']
for name,leaves in RECIPES.items():
    p = .5*components[(35,True,'xgb',leaves)]['probabilities']+.5*lp
    path = OUT/f'submission_half4_{name}_300_150_experimental.csv'
    frame = pd.read_csv(path)
    assert frame.columns.tolist()==['index','target'] and len(frame)==15329
    assert frame['index'].is_unique and frame['index'].equals(sample['index'])
    assert frame['index'].equals(test['index']) and frame['index'].equals(old['index'])
    assert np.isfinite(frame.target).all() and frame.target.between(0,1).all()
    assert np.max(np.abs(frame.target-p))<=1e-15 and digest(path)==progress['submission_hashes'][name]
    test_diagnostics[name]=dict(csv_sha256=digest(path),
        rank_correlation_vs_public_reference=float(spearmanr(frame.target,old.target).statistic),
        top1000_overlap_vs_public_reference=len(set(frame.nlargest(1000,'target')['index']) & set(old.nlargest(1000,'target')['index'])),
        mean_absolute_probability_difference=float(np.mean(np.abs(frame.target-old.target))))
    if name=='reference_leaf15':
        assert digest(path)==digest(reference_path) and np.array_equal(frame.target.to_numpy(),old.target.to_numpy())
notebook = nbformat.read(NOTEBOOK,as_version=4)
nbformat.validate(notebook)
assert notebook.cells[2].source==SOURCE.read_text().split('\ndef main():\n')[0]
for c in notebook.cells:
    if c.cell_type=='code':
        compile(c.source,'verified-notebook-cell','exec')
        assert c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs)
assert manifest['all_code_cells_executed']
for name,path in [('final_notebook',PROJECT/'e1156285.ipynb'),('archived_journal',PROJECT/'archive/e1156285_development_log.ipynb'),('public_reference',reference_path)]:
    assert digest(path)==manifest['protected_files_before'][name]
verification = dict(all_declared_settings_verified=True,only_selected_leaf_limits_changed=True,
    every_auc_independently_recalculated=True,every_group_summary_verified=True,per_period_auc_verified=True,
    fitting_and_checking_ids_disjoint=True,feature_filter_nested_in_each_fitting_window=True,
    all_submission_formulas_and_indices_verified=True,all_final_models_reloaded_in_fresh_interpreter=True,
    reference_csv_byte_identical_to_0_95484_submission=True,reused_reference_components_exactly_verified=True,
    original_data_hashes_unchanged=True,protected_final_notebook_unchanged=True,
    chronological_checks=18,saved_component_predictions=76,fresh_component_fits=38,reused_reference_components=38,
    checking_row_appearances=len(appearances),unique_checking_transactions=len(set(appearances)),
    current_verified_public_best=.95484,candidate_public_auc=None,private_auc=None,
    final_model_selection='student decision',codex_kaggle_upload=False,
    actual_final_models=actual_models,test_diagnostics=test_diagnostics,
    notebook_code_validated_and_executed=True,presentation_cell_repaired_without_retraining=True)
(OUT/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
pd.DataFrame(diagnostics).to_csv(OUT/'fold_counts_and_weight_diagnostics.csv',index=False)
pd.DataFrame(component_scores).to_csv(OUT/'component_auc_diagnostics.csv',index=False)
component_frame = pd.DataFrame(component_scores)
component_summary = []
for group,cutoffs in groups.items():
    for name,frame in component_frame.loc[component_frame.cutoff.isin(cutoffs)].groupby('component'):
        component_summary.append(dict(group=group,component=name,mean_auc=frame.auc.mean(),worst_auc=frame.auc.min()))
pd.DataFrame(component_summary).to_csv(OUT/'component_auc_diagnostic_summary.csv',index=False)
print(pd.read_csv(OUT/'summary.csv').to_string(index=False))
print(json.dumps(verification,indent=2))
print('Independent verification complete; no model fitting occurred.')
