"""Validate and save a read-only ranking-error review; no training or submission."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import sys

import nbformat
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PROJECT=Path('/Users/mlvlo/BT4012/Kaggle')
OUT=HERE/'ranking_error_review_2026_10_14'
SOURCE=PROJECT/'archive/results/half4_xgb_leaf_capacity_results'
JOURNAL=PROJECT/'archive/e1156285_development_log.ipynb'
FINAL=PROJECT/'e1156285.ipynb'
STAGED_JOURNAL=HERE/'e1156285_updated_ranking_error_review_development_log.ipynb'
NATIVE=PROJECT/'archive/results/ranking_error_review_2026_10_14'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes(folder):return {str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}

if sys.argv[1]=='stage':
    meta=json.loads((OUT/'review_metadata.json').read_text())
    assert digest(FINAL)==meta['final_notebook_sha256']
    assert digest(JOURNAL)==meta['journal_before_sha256']
    fold=pd.read_csv(OUT/'fold_ranking_errors.csv')
    steps=pd.read_csv(OUT/'per_step_error_contributions.csv')
    matrix=pd.read_csv(OUT/'positive_negative_step_pair_errors.csv')
    tx=pd.read_csv(OUT/'positive_transaction_errors.csv')
    agreement=pd.read_csv(OUT/'component_pair_agreement.csv')
    within=pd.read_csv(OUT/'within_vs_cross_step_errors.csv')
    drift=json.loads((OUT/'feature_drift_summary.json').read_text())
    for (cutoff,model),frame in matrix.groupby(['cutoff','model']):
        row=fold.loc[fold.cutoff.eq(cutoff)&fold.model.eq(model)].iloc[0]
        assert int(frame.pairs.sum())==int(row.total_pairs)
        assert abs(frame.misordered_pairs_including_half_ties.sum()-row.misordered_pairs_including_half_ties)<1e-6
        assert abs(1-frame.misordered_pairs_including_half_ties.sum()/frame.pairs.sum()-row.auc)<1e-12
        sf=steps.loc[steps.cutoff.eq(cutoff)&steps.model.eq(model)]
        assert abs(sf.loss_from_this_positive_step.sum()-row.misordered_pairs_including_half_ties)<1e-6
        assert abs(sf.share_of_fold_loss.sum()-1)<1e-12
        tf=tx.loc[tx.cutoff.eq(cutoff)&tx.model.eq(model)]
        assert len(tf)==row.positives and tf.txId.is_unique
        assert abs(tf.bad_pairs.sum()-row.misordered_pairs_including_half_ties)<1e-6
    # Independently confirm the exact pooled AUC via Mann-Whitney ranks.
    component={}
    for path in (SOURCE/'component_predictions').glob('*.json'):
        if path.name.endswith('.provenance.json'):continue
        receipt=json.loads(path.read_text())
        if receipt['cutoff']==21 and not receipt['is_final'] and (receipt['kind']=='lgb' or receipt['settings']['max_leaves']==15):
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
                component[receipt['kind']]=(saved['probabilities'].copy(),saved['checking_ids'].copy())
    assert np.array_equal(component['xgb'][1],component['lgb'][1])
    labels=pd.read_csv(PROJECT/'Data Set/Train Data.csv',usecols=['txId','label']).set_index('txId')
    y=pd.to_numeric(labels.loc[component['xgb'][1],'label']).to_numpy().astype(int)
    p=.5*component['xgb'][0]+.5*component['lgb'][0]
    npos=int(y.sum());nneg=int((y==0).sum())
    rank_auc=(rankdata(p)[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
    control=fold.loc[fold.cutoff.eq(21)&fold.model.eq('reference_pair')].iloc[0]
    assert abs(rank_auc-control.auc)<1e-14
    focal=steps.loc[steps.cutoff.eq(21)&steps.model.eq('reference_pair')].sort_values('time_step')
    contribution=float(focal.loc[focal.time_step.isin([24,29,31,32]),'share_of_fold_loss'].sum())
    shared=float(agreement.loc[agreement.cutoff.eq(21),'both_wrong_fraction_of_blend_wrong'].iloc[0])
    worst=tx.loc[tx.cutoff.eq(21)&tx.model.eq('reference_pair')].sort_values('bad_pairs',ascending=False)
    top400=float(worst.head(400).bad_pairs.sum()/worst.bad_pairs.sum())
    by29=steps.loc[steps.model.eq('reference_pair')&steps.time_step.eq(29)].sort_values('cutoff')
    before=float(by29.loc[by29.cutoff.eq(21),'within_step_auc'].iloc[0])
    after=float(by29.loc[by29.cutoff.eq(24),'within_step_auc'].iloc[0])
    masks={r['cutoff']:r['removed'] for r in json.loads((SOURCE/'training_only_feature_masks.json').read_text())}
    assert masks[21]==masks[24]
    verification=dict(no_model_fitting=True,no_new_predictions=True,all_pair_matrix_totals_verified=True,
        all_period_contributions_verified=True,transaction_loss_totals_verified=True,
        pooled_auc_independently_verified_by_mann_whitney_ranks=True,
        cutoff21_and_cutoff24_xgb_feature_masks_identical=True,
        latest_fourteen_period_auc=float(control.auc),latest_positive_rows=npos,latest_negative_rows=nneg,
        four_positive_period_share_of_errors=contribution,both_components_wrong_share_of_blend_errors=shared,
        top400_positive_share_of_errors=top400,step29_auc_train_through21=before,
        step29_auc_train_through24=after,hidden_test_labels_available=False,
        final_notebook_unchanged=True,codex_kaggle_upload=False)
    (OUT/'independent_review_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
    colors=['#bd4c39' if s in [24,29,31,32] else '#7395ad' for s in focal.time_step]
    bars=axes[0].bar(focal.time_step.astype(str),100*focal.share_of_fold_loss,color=colors)
    axes[0].set(title='Where ranking errors concentrate',xlabel='Time step of the illicit transaction',
                ylabel='Share of all wrongly ordered pairs (%)',ylim=(0,50))
    for bar,value in zip(bars,100*focal.share_of_fold_loss):
        if value>=8:axes[0].text(bar.get_x()+bar.get_width()/2,value+.8,f'{value:.1f}%',ha='center',fontsize=10)
    axes[0].text(.02,.98,'Train through 21; check 22–35',transform=axes[0].transAxes,va='top',fontsize=10)
    axes[1].plot(by29.cutoff,by29.within_step_auc,marker='o',color='#245a70',linewidth=2)
    axes[1].set(title='Step 29 becomes easier with newer fitting data',xlabel='Last time step included in fitting',
                ylabel='AUC within the same held-out step 29',ylim=(.68,1.025),xticks=[15,17,19,21,24,28])
    axes[1].annotate(f'{before:.3f}',(21,before),xytext=(19.5,.71),arrowprops={'arrowstyle':'-','color':'#666666'})
    axes[1].annotate(f'{after:.3f}',(24,after),xytext=(23.3,.94),arrowprops={'arrowstyle':'-','color':'#666666'})
    fig.suptitle('Recorded validation errors — current 300/150 pair, half-life four',fontsize=14)
    fig.supxlabel('Existing fits only. These are development results; hidden Kaggle labels are unavailable.',fontsize=10)
    fig.savefig(OUT/'ranking_error_diagnosis.png',dpi=180)
    fig.savefig(OUT/'ranking_error_diagnosis.svg')
    plt.close(fig)
    table='\n'.join(['| Illicit transaction period | Illicit rows | AUC within period | Share of all block ranking errors |',
        '|---:|---:|---:|---:|',
        *[f'| {int(row.time_step)} | {int(row.positives)} | {row.within_step_auc:.6f} | {100*row.share_of_fold_loss:.2f}% |'
          for _,row in focal.loc[focal.time_step.isin([24,29,31,32])].iterrows()]])
    record=f'''## 47. Student-requested review of recurring ranking mistakes

**14 October 2026.** The student asked Codex to examine recurring errors and explain why the target **0.97** has not been achieved. This is a descriptive review of the existing 18 chronological prediction sets, provided labels and label-free test features. No model, feature transformation, candidate, blend-weight search or threshold is fitted or selected. The public best remains **0.95484** and the most recent seven-leaf submission scored **0.95368**.

### What a ranking error means

For every held-out illicit transaction, count the held-out licit transactions scored above it, assigning half credit to exact ties. Sum these counts and divide by all illicit–licit pairs. This equals **1 minus ROC-AUC**. It measures ranking, not the number of rows misclassified at threshold 0.5. The review verifies this identity against the existing AUCs and independently against Mann–Whitney ranks. Changing a decision threshold cannot fix ROC-AUC. [Scikit-learn ROC documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics) describes the use of non-thresholded scores.

At public AUC 0.95484, pair loss is **4.516%**, including half ties. Target 0.97 corresponds to **3.000%**. Reaching that target requires a net reduction of approximately **33.6% of the remaining pair loss**, not merely correcting 1.516% of transactions. This is an interpretation of the reported score, not identification of hidden public errors.

### Concrete failures in the latest fourteen-period check

The current fifteen-leaf 300/150 recipe trained through step 21 and checked steps 22–35 has **{control.auc:.6f} AUC**, **{npos:,} illicit rows**, **{nneg:,} licit rows**, and **{int(control.misordered_pairs_including_half_ties):,} wrongly ordered pairs including half ties** out of **{int(control.total_pairs):,} pairs**.

{table}

Together, illicit transactions from steps **24, 29, 31 and 32 contribute {100*contribution:.2f}%** of the ranking loss in this block. Each period's contribution counts its positives against **all negatives in the block**; it is not its within-period AUC. The same four positive periods contribute **86.29%** of block errors at fitting cutoff 17 and **92.76%** at cutoff 19. These repetitions involve overlapping transactions and are not independent evidence.

The **400** most error-contributing illicit transactions account for **{100*top400:.2f}%** of this block's pair loss. This is a concentration diagnostic, not permission to memorise transaction IDs. The recurrent-error inventory retains anonymised transaction identifiers for traceability; identifiers must not be used as prediction clues or matched to external labels.

At step 29, median illicit score is **0.000763**, while median licit score is **0.000448**. Within-period AUC is only **{before:.6f}**. These illicit transactions are not just slightly below an arbitrary 0.5 threshold; they are poorly separated in score order from licit transactions. Raw score magnitudes are not assumed calibrated fraud probabilities.

### Shared errors limit the value of more averaging

For fitting cutoff 21, **{100*shared:.2f}%** of the reference pair's strictly wrong illicit–licit orderings are wrong in **both XGBoost and LightGBM individually**. Corresponding shares at cutoffs 17 and 19 are approximately **82.49%** and **89.60%**. The models share many blind spots. A convex probability average of these fixed two components cannot repair a pair that both strictly rank incorrectly. This observation concerns saved checking scores; it is not a bound on hidden public performance or on newly trained models.

There is no evidence here that GPU availability or choice of library is the main barrier. The existing components are similar enough that adjustments to their mixture may have limited effect on shared misses. Earlier generic graph and semi-supervised prototypes also did not establish stronger candidates; this review does not reverse those measured outcomes or prove that every such method fails.

### The validation regimes are materially different

For the **same held-out step 29**, the reference's within-period AUC changes from **{before:.6f}** when fitting through step **21** to **{after:.6f}** when fitting through step **24**. At step 32 it changes from **0.807190** to **0.999154**. Both fitting windows use the **same six excluded XGBoost features**, fixed settings and half-life-four weighting rule. More recent labelled examples and changed recency-weighted fitting distributions are therefore a plausible explanation; this is not an isolated causal test of which added row or feature produced the gain.

Many original checks stop fitting before step 24 and expose models to these difficult later patterns. The final competition model instead fits through step 35. The added later checks include newer examples but forecast shorter horizons, producing approximately 0.997 AUC. Both are legitimate chronological checks of different situations; neither reproduces the hidden future exactly. The older checks must not be dropped because they are difficult, and the later checks must not be treated as forecasts near 0.997. [Scikit-learn's time-series validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data) explains why future periods should remain separate.

**Review of our selection process:** I should have examined which periods drove the seven-leaf improvement before recommending it. Its larger aggregate validation means favoured it, but the public result reversed that ordering. The review shows why a single overall mean is insufficient: we should report gains on recurring difficult periods, later fitting windows, latest/worst checks and the original fourteen-period horizons separately. These are reused development checks, not newly created independent evidence. We cannot choose validation weights retrospectively to make the public winner look best.

Most block errors cross period boundaries (**88.93%**), but cross-period pairs are also **91.37% of all pairs**. Pair-weighted within-period AUC is **0.875498**, below cross-period AUC **0.905504**. Consequently, the number of cross-period errors alone does **not** establish that incompatible score scales or period calibration is the main problem. Genuine within-period discrimination remains weak in the difficult groups.

### What the unlabelled test features show

The test feature distributions were compared with all known training rows and with known training rows from steps 29–35. Across features, weighting the descriptive KS distances by the average normalised split-gain shares of the two frozen final models gives **{drift['gain_weighted_ks_test_vs_all_known']:.6f}** versus **{drift['gain_weighted_ks_test_vs_recent_known']:.6f}**, respectively. This supports the observation that recent feature distributions are closer under this chosen descriptive measure, consistent with investigating recency. It does not reveal test labels, classify fraud types, explain the public/private gap or prove causal drift; changing class mixtures or sampling can also change marginal distributions.

Both models strongly rely on anonymous feat_53 and feat_5 by split-gain share. Their anonymous meanings are not inferred from an external source. The six features removed from XGBoost account for only **{100*drift['lgb_gain_share_of_six_xgb_excluded_features']:.3f}%** of LightGBM's total split gain. This does not prove they are harmless, but the review does not support blaming them as the dominant observed mechanism. Gain is descriptive and cannot assign causal responsibility for prediction errors.

### Conclusion and next student decision

The strongest measured limitation is **shared failure to distinguish particular illicit groups in some future periods**. A second limitation is **candidate selection from validation summaries whose training information and horizons differ from final fitting**. We have not established the precise cause of the remaining **public** errors, because those labels are hidden. There is no measured ceiling at 0.95484, and no justified promise of 0.97 from a library switch, more trees, a threshold change or more averaging.

Keep the exact 0.95484 recipe as reference. Subsequent comparisons should target a clearly explained source of transferable signal or training coverage, and explicitly inspect the recurring hard periods and later fitting regimes. Label-free contextual features are one possible student-chosen direction; their availability, unknown-row use and validation transforms would need a precise fixed design. No such feature set is chosen, implemented or fitted by this review. No ongoing automatic search is started.

### Saved evidence and AI assistance

All 18 checks, five already-fitted score sets, per-period pair losses, exact positive/negative period matrices, recurrent positive errors, shared-component error counts, label-free feature statistics, source scripts and independent arithmetic verification are preserved in `archive/results/ranking_error_review_2026_10_14/`. The figure `ranking_error_diagnosis.png` and editable SVG visualise the concentration and step-29 change. The plot uses existing results only and is not a Kaggle forecast.

**AI assistance record:** At the student's explicit request, OpenAI Codex reviewed the existing predictions and provided labels, calculated ranking-error contributions and recurrence, inspected frozen model gains and label-free test distributions, verified arithmetic independently, plotted the findings and drafted this record. No new model training, guessed labels, external transaction data, participant solutions, new submission file, parameter/weight search, final selection or Kaggle upload occurred. The final notebook and all prior fitting/submission artifacts remain unchanged. Any next material modeling experiment requires a meaningful student choice under the supplied competition rules.
'''
    for name in ['review_ranking_errors.py','review_feature_drift.py','finalize_ranking_error_review.py']:
        shutil.copy2(HERE/name,OUT/name)
    (OUT/'student_review_summary.json').write_text(json.dumps(dict(
        current_verified_public_best=.95484,review_only=True,new_model_fitting=False,
        latest_check_four_period_error_share=contribution,shared_component_error_share=shared,
        step29_auc_cutoff21=before,step29_auc_cutoff24=after,
        required_relative_public_pair_loss_reduction=meta['required_relative_reduction_in_public_pair_loss'],
        conclusion='Shared misses in difficult periods and differing validation information/horizons limit aggregate-based selection; hidden public causes remain unknown.',
        next_experiment='not selected',codex_kaggle_upload=False),indent=2)+'\n')
    notebook=nbformat.read(JOURNAL,as_version=4)
    assert not any(c.source.startswith('## 47.') for c in notebook.cells)
    original=copy.deepcopy(notebook.cells)
    position=next(i for i,c in enumerate(notebook.cells) if c.source.startswith('## Pending: Compare prediction thresholds'))
    notebook.cells.insert(position,nbformat.v4.new_markdown_cell(record))
    assert notebook.cells[:position]+notebook.cells[position+1:]==original
    nbformat.validate(notebook)
    nbformat.write(notebook,STAGED_JOURNAL)
    (OUT/'journal_backup').mkdir(exist_ok=True)
    shutil.copy2(JOURNAL,OUT/'journal_backup/e1156285_development_log_before_section47.ipynb')
    plan=dict(final_notebook_sha256=digest(FINAL),journal_before_sha256=digest(JOURNAL),
        journal_after_sha256=digest(STAGED_JOURNAL),source_verification_sha256=digest(SOURCE/'independent_verification.json'),
        source_reference_csv_sha256=digest(SOURCE/'submission_half4_reference_leaf15_300_150_experimental.csv'),
        result_hashes=hashes(OUT),native_results=str(NATIVE),section_added=47)
    (HERE/'ranking_error_review_publish_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(table)
    print(json.dumps(verification,indent=2))
    print('Read-only review verified and Section 47 staged.')
elif sys.argv[1]=='publish':
    plan=json.loads((HERE/'ranking_error_review_publish_plan.json').read_text())
    assert digest(FINAL)==plan['final_notebook_sha256']
    assert digest(JOURNAL)==plan['journal_before_sha256']
    assert digest(STAGED_JOURNAL)==plan['journal_after_sha256']
    assert digest(SOURCE/'independent_verification.json')==plan['source_verification_sha256']
    assert digest(SOURCE/'submission_half4_reference_leaf15_300_150_experimental.csv')==plan['source_reference_csv_sha256']
    assert hashes(OUT)==plan['result_hashes']
    assert not NATIVE.exists(), 'Preserve an existing review rather than overwrite it.'
    shutil.copytree(OUT,NATIVE)
    pending=JOURNAL.with_name('e1156285_development_log.section47_pending.ipynb')
    assert not pending.exists()
    shutil.copy2(STAGED_JOURNAL,pending)
    pending.replace(JOURNAL)
    assert hashes(NATIVE)==plan['result_hashes']
    assert digest(JOURNAL)==plan['journal_after_sha256']
    assert digest(FINAL)==plan['final_notebook_sha256']
    assert digest(SOURCE/'independent_verification.json')==plan['source_verification_sha256']
    assert digest(SOURCE/'submission_half4_reference_leaf15_300_150_experimental.csv')==plan['source_reference_csv_sha256']
    nbformat.validate(nbformat.read(JOURNAL,as_version=4))
    (NATIVE/'publication_manifest.json').write_text(json.dumps(dict(plan,
        publication_complete=True,protected_final_notebook_unchanged=True,
        no_model_fitting=True,no_new_submission=True,codex_kaggle_upload=False),indent=2)+'\n')
    print(json.dumps(dict(results=str(NATIVE),section_added=47,final_notebook_unchanged=True,
        no_training=True,no_new_submission=True,publication_complete=True),indent=2))
else:raise ValueError('Choose stage or publish')
