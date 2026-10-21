"""Review supplied prediction artifacts and public scores; never fit a model."""
from pathlib import Path
from decimal import Decimal
import copy
import hashlib
import json
import shutil
import sys

import nbformat
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT=Path('/Users/mlvlo/BT4012/Kaggle')
HERE=Path(__file__).resolve().parent
OUT=HERE/'recency300_public_and_final_attempt_review_2026_10_14'
NATIVE=PROJECT/'archive/results/recency300_public_and_final_attempt_review_2026_10_14'
JOURNAL=PROJECT/'archive/e1156285_development_log.ipynb'
FINAL=PROJECT/'e1156285.ipynb'
STAGED_JOURNAL=HERE/'e1156285_updated_recency300_public_review_development_log.ipynb'
REFERENCE=PROJECT/'archive/results/half4_longer_training_results/submission_half4_long_xgb_300_150_experimental.csv'
AVERAGE=PROJECT/'submissions/submission_lowreg_pair_recent345_avg_xgb300.csv'
FIVE_HUNDRED=PROJECT/'archive/results/claude_experiments/submissions/submission_lowreg_pair_recent4_xgb500.csv'
TABLE=FIVE_HUNDRED.parent/'xgb_tree_count_validation.csv'
EVIDENCE=Path('/var/folders/vj/6q4l2jyn58xdb1kn9llb4r6w0000gn/T/codex-clipboard-c3bf54d7-1161-47d5-8b00-51e5c436a6a8.png')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def folder_hashes(folder):return {str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}

if sys.argv[1]=='stage':
    OUT.mkdir(exist_ok=True)
    reference=pd.read_csv(REFERENCE)
    sample=pd.read_csv(PROJECT/'Data Set/Sample Submission/BT4012 Competition Submission.csv')
    assert digest(REFERENCE)=='7fd1ccf24d89a1ff1be9d21a273f787e1a40f6413d98a324fd92ba8767d95163'
    audits={}
    for name,path in [('submitted_average',AVERAGE),('prepared_xgb500',FIVE_HUNDRED)]:
        frame=pd.read_csv(path)
        assert frame.columns.tolist()==['index','target'] and len(frame)==15329
        assert frame['index'].is_unique and frame['index'].equals(sample['index'])
        assert frame['index'].equals(reference['index'])
        assert np.isfinite(frame.target).all() and frame.target.between(0,1).all()
        audits[name]=dict(path=str(path),sha256=digest(path),rows=len(frame),columns=frame.columns.tolist(),
            exact_sample_index_alignment=True,unique_indices=True,finite_probabilities_in_range=True,
            rank_correlation_vs_best=float(spearmanr(frame.target,reference.target).statistic),
            top1000_overlap_vs_best=len(set(frame.nlargest(1000,'target')['index'])&set(reference.nlargest(1000,'target')['index'])),
            mean_absolute_probability_difference=float(np.abs(frame.target-reference.target).mean()),
            generation_code_located=False,model_reproduction_performed=False)
    table=pd.read_csv(TABLE)
    recorded=[]
    for group,frame in [('original_15',table),('later_seven',table.iloc[8:])]:
        for col in ['xgb200','xgb300','xgb400','xgb500']:
            delta=frame[col]-frame.xgb300
            recorded.append(dict(group=group,column=col,checks=len(frame),mean_auc=float(frame[col].mean()),
                mean_delta_vs_table_xgb300=float(delta.mean()),wins_vs_table_xgb300=int((delta>0).sum()),
                worst_auc=float(frame[col].min())))
    pd.DataFrame(recorded).to_csv(OUT/'recorded_table_summary.csv',index=False)
    old_scores=pd.read_csv(REFERENCE.parent/'validation_results.csv')
    old=old_scores.loc[old_scores.recipe.eq('long_xgb_300_150')].set_index('cutoff')
    differences=[]
    for _,row in table.iterrows():
        cutoff=int(row.exam.split('-')[0])-1
        differences.append(dict(cutoff=cutoff,recorded_control300=row.xgb300,
            verified_nested_control300=float(old.loc[cutoff,'auc']),
            difference=float(row.xgb300-old.loc[cutoff,'auc'])))
    pd.DataFrame(differences).to_csv(OUT/'control_protocol_differences.csv',index=False)
    score,best=Decimal('0.95307'),Decimal('0.95484')
    latest=table.iloc[-1]
    main_notebook=nbformat.read(FINAL,as_version=4)
    assert any('n_estimators=200' in c.source and 'def make_xgb' in c.source for c in main_notebook.cells if c.cell_type=='code')
    summary=dict(date='2026-10-14',observed_public_score=float(score),current_verified_public_best=float(best),
        public_delta_vs_best=float(score-best),public_delta_vs_leaf7=float(score-Decimal('0.95368')),
        submitted_filename=AVERAGE.name,score_evidence='Student-supplied Kaggle screenshot',
        prepared_xgb500_public_score=None,basic_file_audits=audits,
        source_generation_code_missing=True,recorded_table_summary=recorded,
        table_control_protocol_matches_verified_checks=False,
        largest_absolute_control_auc_difference=max(abs(r['difference']) for r in differences),
        latest_check_recorded_gain_500_vs_300=float(latest.xgb500-latest.xgb300),
        README_declares_xgb_trees=300,protected_final_notebook_uses_xgb_trees=200,
        final_notebook_sha256=digest(FINAL),journal_before_sha256=digest(JOURNAL),
        no_model_fitting=True,no_new_candidate=True,codex_kaggle_upload=False,
        final_selection='student decision',next_experiment='not selected')
    (OUT/'review_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copy2(EVIDENCE,OUT/'reported_public_scores_2026_10_14.png')
    shutil.copy2(TABLE,OUT/TABLE.name)
    shutil.copy2(PROJECT/'README.md',OUT/'README_snapshot_for_review.md')
    record=f'''## 48. Reported recency-average decline and review before the last daily attempt

**14 October 2026.** The student supplied a screenshot showing **{score}** for `submission_lowreg_pair_recent345_avg_xgb300.csv`. This is **{score-best}** relative to the current best **{best}**, and **{score-Decimal('0.95368')}** relative to today's seven-leaf submission **0.95368**. The best public recipe remains the fifteen-leaf XGBoost 300 / LightGBM 150 pair at half-life four. This is a review of existing files and recorded results; no models or new candidates are trained.

### Interpretation of the new result

The filename and current README describe an average of half-lives three, four and five with 300 XGBoost trees. Its prediction rankings are less successful on the reported public subset than the 0.95484 reference. If generation code confirms that intended recipe, averaging these recency settings did not preserve the reference's public ranking quality. It could dilute useful learned distinctions, but the hidden labels do not identify the exact pairs responsible. No numeric noise correction, statistical significance claim, private-score prediction or claim that ensembles generally fail is justified.

The reviewed average has rank correlation **{audits['submitted_average']['rank_correlation_vs_best']:.6f}** with the reference and shares **{audits['submitted_average']['top1000_overlap_vs_best']} of the top 1,000** transactions. Similarity measures are not accuracy estimates. Its public AUC cannot be obtained by averaging component AUCs. The current evidence does not establish a route above 0.96 or 0.97.

### Basic CSV audit and missing provenance

The submitted average and the existing `submission_lowreg_pair_recent4_xgb500.csv` each have **15,329 rows**, exact sample indices in order, unique IDs, columns `index,target`, and finite probabilities in [0,1]. Their SHA-256 values are **{audits['submitted_average']['sha256']}** and **{audits['prepared_xgb500']['sha256']}**, respectively. These are format and similarity checks only. Without generation code, they do not independently verify training labels, feature availability, weighting, model settings, target-to-transaction mapping or reproducibility.

A search of the native project scripts and notebook sources did not find generation code naming either new file. The student has been asked for the generator path or script text, and the filename proposed for today's last daily submission. No source path is inferred from a filename alone. The current protected final notebook cannot be their generator: its `make_xgb()` still specifies **200 trees**, writes the older recent4 and recent345-average filenames, and remains unchanged.

The README currently describes **300 trees** and says the notebook writes the newer filenames. This documentation/code inconsistency does not explain the submitted CSV's lower public score, but must be reconciled when the student chooses final assessment models and authorises the final code update. Codex does not edit the protected final notebook or README in this review.

### Existing 500-tree candidate: a modest, conditional case

Claude's recorded `xgb_tree_count_validation.csv` reports the following **within-table** comparison:

| Checking group | Column xgb300 | Column xgb500 | Difference | 500 wins |
|---|---:|---:|---:|---:|
| Original 15 | {table.xgb300.mean():.6f} | {table.xgb500.mean():.6f} | {(table.xgb500-table.xgb300).mean():+.6f} | {int((table.xgb500>table.xgb300).sum())}/15 |
| Later seven | {table.iloc[8:].xgb300.mean():.6f} | {table.iloc[8:].xgb500.mean():.6f} | {(table.iloc[8:].xgb500-table.iloc[8:].xgb300).mean():+.6f} | {int((table.iloc[8:].xgb500>table.iloc[8:].xgb300).sum())}/7 |

The latest fourteen-period check gains only **{float(latest.xgb500-latest.xgb300):+.8f}**. Its test-rank correlation with the public reference is **{audits['prepared_xgb500']['rank_correlation_vs_best']:.6f}**, sharing **{audits['prepared_xgb500']['top1000_overlap_vs_best']} of the top 1,000** rows. This is another close variant, not evidence for a large improvement.

The table's 300-tree control differs from the independently verified nested-filter control by up to **{summary['largest_absolute_control_auc_difference']:.6f} AUC** across original checks. This exceeds simple decimal rounding for some checks. The supplied table alone does not establish why: feature masks, preprocessing, weighting, package versions or other recipe differences require source inspection. Use its 500-versus-300 differences within that table; do not mix its absolute scores with our 18-check comparison. No later-cutoff 24/28/31 results for this candidate have been located.

**Review position before the last daily attempt:** The existing 500-tree candidate has a modest recorded case for an exploratory comparison if its generator confirms the intended fixed change and is reproducible. It is not ready for a fully audited recommendation while that source is missing. Preserving today's last attempt is reasonable if there is no traceable, understood candidate. There is no requirement to consume the maximum daily quota. No final daily candidate or final assessment checkbox is selected by Codex.

### Prior work relevant to choosing the next experiment

The recurring-error review in Section 47 found shared blind spots in difficult illicit groups and different information conditions across fitting cutoffs. Current near-copy variants do not demonstrate that those blind spots are resolved. The earlier Section 19 also records that within-period feature ranks were already tested under earlier recipes and did not give the stronger overall result then. A comparison under the current recency recipe would be a new, separately selected hypothesis, not an unexplored idea or permission to repeat the older search automatically. No such new experiment is run here.

**AI assistance record:** At the student's request, OpenAI Codex read the new screenshot, inspected current scripts/notebooks, checked both existing CSV formats and hashes, calculated label-free similarities and recorded table summaries, compared control protocols, and drafted this review. Underlying generation code and individual predictions for the recorded tree-count table remain missing. Codex did not invent provenance, train models, generate new predictions, change final code, choose final selections or upload to Kaggle. The student remains responsible for the next modeling and submission decisions.
'''
    notebook=nbformat.read(JOURNAL,as_version=4)
    assert not any(c.source.startswith('## 48.') for c in notebook.cells)
    before=copy.deepcopy(notebook.cells)
    position=next(i for i,c in enumerate(notebook.cells) if c.source.startswith('## Pending: Compare prediction thresholds'))
    notebook.cells.insert(position,nbformat.v4.new_markdown_cell(record))
    assert notebook.cells[:position]+notebook.cells[position+1:]==before
    nbformat.validate(notebook)
    nbformat.write(notebook,STAGED_JOURNAL)
    (OUT/'journal_backup').mkdir(exist_ok=True)
    shutil.copy2(JOURNAL,OUT/'journal_backup/e1156285_development_log_before_section48.ipynb')
    shutil.copy2(HERE/'review_recency300_public_and_last_attempt.py',OUT/'review_recency300_public_and_last_attempt.py')
    plan=dict(final_notebook_sha256=digest(FINAL),README_sha256=digest(PROJECT/'README.md'),
        journal_before_sha256=digest(JOURNAL),journal_after_sha256=digest(STAGED_JOURNAL),
        reference_csv_sha256=digest(REFERENCE),average_csv_sha256=digest(AVERAGE),
        candidate500_csv_sha256=digest(FIVE_HUNDRED),result_hashes=folder_hashes(OUT),section_added=48)
    (HERE/'recency300_public_review_publish_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(public_score=float(score),delta_vs_best=float(score-best),
        basic_audits_passed=True,generator_missing=True,recorded_500_vs_300=recorded[-1],
        final_notebook_unchanged=True,section_staged=48),indent=2))
elif sys.argv[1]=='publish':
    plan=json.loads((HERE/'recency300_public_review_publish_plan.json').read_text())
    assert digest(FINAL)==plan['final_notebook_sha256']
    assert digest(PROJECT/'README.md')==plan['README_sha256']
    assert digest(JOURNAL)==plan['journal_before_sha256']
    assert digest(STAGED_JOURNAL)==plan['journal_after_sha256']
    for path,key in [(REFERENCE,'reference_csv_sha256'),(AVERAGE,'average_csv_sha256'),(FIVE_HUNDRED,'candidate500_csv_sha256')]:assert digest(path)==plan[key]
    assert folder_hashes(OUT)==plan['result_hashes']
    assert not NATIVE.exists(), 'Preserve any prior review.'
    shutil.copytree(OUT,NATIVE)
    pending=JOURNAL.with_name('e1156285_development_log.section48_pending.ipynb')
    assert not pending.exists()
    shutil.copy2(STAGED_JOURNAL,pending)
    pending.replace(JOURNAL)
    assert folder_hashes(NATIVE)==plan['result_hashes']
    assert digest(JOURNAL)==plan['journal_after_sha256'] and digest(FINAL)==plan['final_notebook_sha256']
    assert digest(PROJECT/'README.md')==plan['README_sha256']
    nbformat.validate(nbformat.read(JOURNAL,as_version=4))
    (NATIVE/'publication_manifest.json').write_text(json.dumps(dict(plan,publication_complete=True,
        final_notebook_unchanged=True,submitted_csvs_unchanged=True,no_model_fitting=True,codex_kaggle_upload=False),indent=2)+'\n')
    print(json.dumps(dict(results=str(NATIVE),section_added=48,final_notebook_unchanged=True,
        no_training=True,no_new_predictions=True,publication_complete=True),indent=2))
else:raise ValueError('Choose stage or publish')
