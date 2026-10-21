"""Descriptive, label-free test-feature comparisons; no model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
from lightgbm import Booster

HERE=Path(__file__).resolve().parent
PROJECT=Path('/Users/mlvlo/BT4012/Kaggle')
OUT=HERE/'ranking_error_review_2026_10_14'
SOURCE=PROJECT/'archive/results/half4_xgb_leaf_capacity_results'
FEATURES=[f'feat_{i}' for i in range(1,166)]
train=pd.read_csv(PROJECT/'Data Set/Train Data.csv')
test=pd.read_csv(PROJECT/'Data Set/Test Data.csv')
labels=pd.to_numeric(train.label,errors='coerce')
known=train.loc[labels.notna()].copy()
known['label']=labels.loc[labels.notna()].astype(int)
recent=known.loc[known.time_step.between(29,35)]
xmodel=XGBClassifier(n_jobs=2)
xmodel.load_model(SOURCE/'models/xgb_300_leaf15_half4.json')
lmodel=Booster(model_file=str(SOURCE/'models/lgb_150_half4.txt'))
xgain=xmodel.get_booster().get_score(importance_type='total_gain')
lgain=dict(zip(lmodel.feature_name(),lmodel.feature_importance(importance_type='gain')))
xsum=sum(xgain.values());lsum=sum(lgain.values())
rows=[]
for feature in FEATURES:
    a=known[feature].to_numpy();r=recent[feature].to_numpy();b=test[feature].to_numpy()
    rows.append(dict(feature=feature,
        ks_test_vs_all_known=float(ks_2samp(a,b,method='asymp').statistic),
        ks_test_vs_recent_known=float(ks_2samp(r,b,method='asymp').statistic),
        test_fraction_outside_all_known_minmax=float(((b<a.min())|(b>a.max())).mean()),
        test_fraction_outside_recent_known_minmax=float(((b<r.min())|(b>r.max())).mean()),
        xgb_total_gain_share=float(xgain.get(feature,0)/xsum),
        lgb_total_gain_share=float(lgain.get(feature,0)/lsum)))
frame=pd.DataFrame(rows)
frame['mean_gain_share']=.5*(frame.xgb_total_gain_share+frame.lgb_total_gain_share)
frame.to_csv(OUT/'label_free_test_feature_drift.csv',index=False)
removed=json.loads((SOURCE/'training_only_feature_masks.json').read_text())[-1]['removed']
for path in (SOURCE/'component_predictions').glob('*.json'):
    if path.name.endswith('.provenance.json'):continue
    receipt=json.loads(path.read_text())
    if receipt['is_final'] and receipt['kind']=='xgb' and receipt['settings']['max_leaves']==15:
        final_removed=[c for c in FEATURES if c not in receipt['columns']]
        break
top=frame.nlargest(20,'mean_gain_share')
summary=dict(known_rows=len(known),unknown_training_rows=int(labels.isna().sum()),recent_known_rows=len(recent),
    test_rows=len(test),
    gain_weighted_ks_test_vs_all_known=float(np.average(frame.ks_test_vs_all_known,weights=frame.mean_gain_share)),
    gain_weighted_ks_test_vs_recent_known=float(np.average(frame.ks_test_vs_recent_known,weights=frame.mean_gain_share)),
    important_20_features_with_recent_ks_above_0_2=int((top.ks_test_vs_recent_known>.2).sum()),
    lgb_gain_share_of_six_xgb_excluded_features=float(frame.loc[frame.feature.isin(final_removed),'lgb_total_gain_share'].sum()),
    interpretation_limits=['KS measures marginal distribution differences, not accuracy or causality.',
        'Differences may reflect class mixture, within-class drift, selection or several effects.',
        'Gain is a fitted-model descriptive statistic; it does not establish causal importance.',
        'No test labels are available and no future features fit or select a new model.'],
    no_model_fitting=True,no_new_candidate=True)
(OUT/'feature_drift_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(frame.nlargest(12,'mean_gain_share').to_string(index=False))
print(json.dumps(summary,indent=2))
