"""Read-only error review of existing chronological predictions; no fitting."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT = Path('/Users/mlvlo/BT4012/Kaggle')
SOURCE = PROJECT/'archive/results/half4_xgb_leaf_capacity_results'
OUT = HERE/'ranking_error_review_2026_10_14'
OUT.mkdir(exist_ok=True)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

declared=json.loads((SOURCE/'declared_recipe.json').read_text())
train=pd.read_csv(PROJECT/'Data Set/Train Data.csv',usecols=['txId','time_step','label'])
target=pd.to_numeric(train.label,errors='coerce')
known=train.loc[target.notna()].copy()
known['label']=target.loc[target.notna()].astype(int)
known=known.set_index('txId',drop=False)
assert len(known)==31235 and known.index.is_unique
assert digest(PROJECT/'Data Set/Train Data.csv')==declared['data_hashes']['Train Data.csv']
components={}
for path in (SOURCE/'component_predictions').glob('*.json'):
    if path.name.endswith('.provenance.json'):continue
    receipt=json.loads(path.read_text())
    if receipt['is_final']:continue
    leaves=receipt['settings'].get('max_leaves',15)
    key=(receipt['cutoff'],receipt['kind'],leaves)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
        components[key]={k:saved[k].copy() for k in saved.files}
    assert receipt['data_hashes']==declared['data_hashes']
    provenance=json.loads(path.with_suffix('.provenance.json').read_text())
    assert digest(path.with_suffix('.npz'))==provenance['npz_sha256']

def loss_per_positive(positive,negative):
    neg=np.sort(negative)
    lower=np.searchsorted(neg,positive,side='left')
    upper=np.searchsorted(neg,positive,side='right')
    return len(neg)-upper+.5*(upper-lower)

all_rows,fold_rows,step_rows,matrix_rows,agreements=[],[],[],[],[]
original_scores=pd.read_csv(SOURCE/'validation_results.csv').set_index(['cutoff','recipe'])
for cutoff,first,last,group in declared['folds']:
    ids=components[(cutoff,'xgb',15)]['checking_ids']
    checked=known.loc[ids]
    assert checked.time_step.between(first,last).all()
    learning=known.loc[components[(cutoff,'xgb',15)]['learning_ids']]
    assert learning.time_step.max()<=cutoff and not set(ids).intersection(learning.index)
    y=checked.label.to_numpy()
    times=checked.time_step.to_numpy()
    xp=components[(cutoff,'xgb',15)]['probabilities']
    lp=components[(cutoff,'lgb',15)]['probabilities']
    x7=components[(cutoff,'xgb',7)]['probabilities']
    x31=components[(cutoff,'xgb',31)]['probabilities']
    predictions={'reference_pair':.5*xp+.5*lp,'xgb15':xp,'lightgbm':lp,
                 'leaf7_pair':.5*x7+.5*lp,'leaf31_pair':.5*x31+.5*lp}
    for name,p in predictions.items():
        positives=y==1;negatives=~positives
        positive_loss=loss_per_positive(p[positives],p[negatives])
        rate=positive_loss/negatives.sum()
        auc=roc_auc_score(y,p)
        assert abs(1-auc-rate.mean())<1e-12
        if name=='reference_pair':assert abs(auc-original_scores.loc[(cutoff,'reference_leaf15'),'auc'])<1e-14
        total_pairs=int(positives.sum()*negatives.sum())
        total_loss=float(positive_loss.sum())
        fold_rows.append(dict(cutoff=cutoff,first=first,last=last,model=name,rows=len(y),
            positives=int(positives.sum()),negatives=int(negatives.sum()),auc=auc,
            misordered_pairs_including_half_ties=total_loss,total_pairs=total_pairs,
            severe_positive_rate_over_0_5=float((rate>.5).mean()),
            positives_below_0_01=int((p[positives]<.01).sum())))
        neg_sorted=np.sort(p[negatives])
        for tx,step,score,r in zip(ids[positives],times[positives],p[positives],rate):
            all_rows.append(dict(cutoff=cutoff,txId=int(tx),time_step=int(step),model=name,
                probability=float(score),negative_outranking_fraction=float(r),
                bad_pairs=float(r*negatives.sum()),validation_negatives=int(negatives.sum())))
        for step in np.unique(times):
            selected=times==step
            yp=y[selected];sp=p[selected]
            count_positive=int(yp.sum());count_negative=int((yp==0).sum())
            loss=float(loss_per_positive(p[selected & positives],p[negatives]).sum())
            step_rows.append(dict(cutoff=cutoff,time_step=int(step),model=name,rows=int(selected.sum()),
                positives=count_positive,negatives=count_negative,positive_share=count_positive/selected.sum(),
                within_step_auc=float(roc_auc_score(yp,sp)) if len(np.unique(yp))==2 else None,
                median_positive_score=float(np.median(sp[yp==1])) if count_positive else None,
                median_negative_score=float(np.median(sp[yp==0])) if count_negative else None,
                loss_from_this_positive_step=loss,share_of_fold_loss=loss/total_loss if total_loss else 0,
                positive_mean_outranking_fraction=loss/(count_positive*negatives.sum()) if count_positive else None))
        for ps in np.unique(times[positives]):
            positive=p[positives & (times==ps)]
            for ns in np.unique(times[negatives]):
                negative=p[negatives & (times==ns)]
                loss=float(loss_per_positive(positive,negative).sum())
                pair_count=len(positive)*len(negative)
                matrix_rows.append(dict(cutoff=cutoff,model=name,positive_step=int(ps),negative_step=int(ns),
                    positive_rows=len(positive),negative_rows=len(negative),pairs=pair_count,
                    misordered_pairs_including_half_ties=loss,pair_auc=1-loss/pair_count,
                    share_of_fold_loss=loss/total_loss if total_loss else 0))
    if cutoff in [17,19,21,24,28,31]:
        bp=.5*xp+.5*lp
        positives=np.flatnonzero(y==1);negatives=np.flatnonzero(y==0)
        both_wrong=blend_wrong=disagreement_wrong=both_wrong_blend_wrong=0
        for start in range(0,len(positives),128):
            pos=positives[start:start+128]
            xwrong=xp[pos,None]<xp[negatives][None,:]
            lwrong=lp[pos,None]<lp[negatives][None,:]
            bwrong=bp[pos,None]<bp[negatives][None,:]
            both=xwrong & lwrong
            both_wrong+=int(both.sum());blend_wrong+=int(bwrong.sum())
            disagreement_wrong+=int(((xwrong ^ lwrong)&bwrong).sum())
            both_wrong_blend_wrong+=int((both&bwrong).sum())
        agreements.append(dict(cutoff=cutoff,total_positive_negative_pairs=len(positives)*len(negatives),
            blend_strict_wrong_pairs=blend_wrong,both_components_strict_wrong_pairs=both_wrong,
            both_wrong_fraction_of_blend_wrong=both_wrong_blend_wrong/blend_wrong,
            component_disagreement_fraction_of_blend_wrong=disagreement_wrong/blend_wrong))

fold=pd.DataFrame(fold_rows);steps=pd.DataFrame(step_rows);matrix=pd.DataFrame(matrix_rows)
transactions=pd.DataFrame(all_rows)
for name,frame in [('fold_ranking_errors.csv',fold),('per_step_error_contributions.csv',steps),
                   ('positive_negative_step_pair_errors.csv',matrix),('positive_transaction_errors.csv',transactions),
                   ('component_pair_agreement.csv',pd.DataFrame(agreements))]:
    frame.to_csv(OUT/name,index=False)
recurrence=transactions.groupby(['txId','time_step','model']).agg(
    checking_appearances=('cutoff','count'),mean_negative_outranking_fraction=('negative_outranking_fraction','mean'),
    worst_negative_outranking_fraction=('negative_outranking_fraction','max'),
    severe_appearances=('negative_outranking_fraction',lambda r:int((r>.5).sum())),
    mean_probability=('probability','mean')).reset_index()
recurrence.to_csv(OUT/'recurring_positive_errors.csv',index=False)
aggregates=[]
for (cutoff,name),frame in matrix.groupby(['cutoff','model']):
    within=frame.positive_step.eq(frame.negative_step)
    total_pairs=int(frame.pairs.sum());total_loss=float(frame.misordered_pairs_including_half_ties.sum())
    within_pairs=int(frame.loc[within,'pairs'].sum());within_loss=float(frame.loc[within,'misordered_pairs_including_half_ties'].sum())
    aggregates.append(dict(cutoff=cutoff,model=name,pooled_auc=1-total_loss/total_pairs,
        within_step_pair_weighted_auc=1-within_loss/within_pairs,
        cross_step_pair_auc=1-(total_loss-within_loss)/(total_pairs-within_pairs),
        cross_step_share_of_errors=(total_loss-within_loss)/total_loss,
        cross_step_share_of_all_pairs=(total_pairs-within_pairs)/total_pairs))
pd.DataFrame(aggregates).to_csv(OUT/'within_vs_cross_step_errors.csv',index=False)
summary=dict(current_verified_public_best=.95484,last_leaf7_public_score=.95368,
    target=.97,required_auc_gain=.97-.95484,public_pair_loss=1-.95484,target_pair_loss=1-.97,
    required_relative_reduction_in_public_pair_loss=((1-.95484)-(1-.97))/(1-.95484),
    original_folds=15,total_folds=18,unique_positive_transactions_in_checks=int(transactions.txId.nunique()),
    analysis_no_model_fitting=True,new_candidate_generated=False,hidden_public_labels_available=False,
    external_data=False,codex_kaggle_upload=False,final_notebook_sha256=digest(PROJECT/'e1156285.ipynb'),
    journal_before_sha256=digest(PROJECT/'archive/e1156285_development_log.ipynb'),
    existing_verification_sha256=digest(SOURCE/'independent_verification.json'),
    existing_reference_csv_sha256=digest(SOURCE/'submission_half4_reference_leaf15_300_150_experimental.csv'))
(OUT/'review_metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
print('LATEST FOURTEEN-PERIOD CHECK (train through 21)')
print(fold.loc[fold.cutoff.eq(21)].to_string(index=False))
print('REFERENCE ERROR CONTRIBUTIONS BY POSITIVE STEP, cutoff 21')
print(steps.loc[steps.cutoff.eq(21)&steps.model.eq('reference_pair')].sort_values('share_of_fold_loss',ascending=False).to_string(index=False))
print('COMPONENT AGREEMENT')
print(pd.DataFrame(agreements).to_string(index=False))
print('WITHIN VS CROSS STEP, cutoff 21')
print(pd.DataFrame(aggregates).loc[lambda f:f.cutoff.eq(21)].to_string(index=False))
print('TOP RECURRING POSITIVE ERRORS, reference, >=3 appearances')
print(recurrence.loc[recurrence.model.eq('reference_pair')&recurrence.checking_appearances.ge(3)].sort_values('mean_negative_outranking_fraction',ascending=False).head(12).to_string(index=False))
print(json.dumps(summary,indent=2))
