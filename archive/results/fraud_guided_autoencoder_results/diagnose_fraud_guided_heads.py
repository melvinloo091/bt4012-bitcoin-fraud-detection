"""Inspect trained classification heads without refitting or changing a model."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RESULTS = HERE if HERE.name == 'fraud_guided_autoencoder_results' else HERE / 'fraud_guided_autoencoder_results'
locations = [RESULTS.parent / 'Data Set', RESULTS.parent.parent / 'Data Set',
             Path('/Users/mlvlo/BT4012/Kaggle/Data Set')]
DATA = next(p for p in locations if (p / 'Train Data.csv').is_file())
FEATURES = [f'feat_{i}' for i in range(1,166)]
families = ['joint_known_only','joint_all_rows']
ends = [14,17,19,21]
paths = [str(RESULTS/'encoders'/f'{family}_{end}_refit.pt') for end in ends for family in families]
assert all(Path(p).is_file() for p in paths)
extract = '''import json, sys, torch
torch.set_num_threads(2)
result = {}
for path in json.loads(sys.argv[1]):
    state = torch.load(path, map_location='cpu', weights_only=True)
    result[path] = dict(weight=state['fraud_head.weight'].tolist()[0],
                        bias=state['fraud_head.bias'].tolist()[0])
print(json.dumps(result))'''
result = subprocess.run([sys.executable,'-c',extract,json.dumps(paths)],capture_output=True,text=True,timeout=60)
assert result.returncode == 0, result.stderr
heads = json.loads(result.stdout)
train = pd.read_csv(DATA/'Train Data.csv',usecols=['txId','time_step','label']+FEATURES)
known = train.loc[train.label.notna()].copy()
comparison = pd.read_csv(RESULTS/'comparison.csv',index_col='train_end')
records=[]
for end in ends:
    rows=known.loc[known.time_step <= end+14]
    learning=(rows.time_step <= end).to_numpy()
    checking=~learning
    labels=rows.label.to_numpy()
    for family in families:
        prefix=f'{family}_{end}_refit'
        with np.load(RESULTS/'encoders'/f'{prefix}_scaler.npz') as saved:
            scaler=StandardScaler()
            scaler.mean_,scaler.scale_,scaler.var_=saved['mean'],saved['scale'],saved['var']
            scaler.n_features_in_=165
        clean=np.clip(scaler.transform(rows[FEATURES].to_numpy(dtype=np.float32)),-10,10).astype(np.float32)
        input_path=RESULTS/f'.diagnostic_input_{family}_{end}.npy'
        output_path=RESULTS/f'.diagnostic_output_{family}_{end}.npy'
        weights_path=str(RESULTS/'encoders'/f'{prefix}.pt')
        np.save(input_path,clean,allow_pickle=False)
        config=dict(action='encode',input_path=str(input_path),output_path=str(output_path),
                    weights_path=weights_path,threads=2,batch_size=512)
        result=subprocess.run([sys.executable,str(RESULTS/'encoder_worker.py'),json.dumps(config)],
                              capture_output=True,text=True,timeout=60)
        assert result.returncode == 0,result.stderr
        summaries=np.load(output_path,allow_pickle=False)
        head=heads[weights_path]
        logits=summaries[:,:32] @ np.asarray(head['weight'],dtype=np.float32) + np.float32(head['bias'])
        assert np.isfinite(logits).all()
        train_auc=float(roc_auc_score(labels[learning],logits[learning]))
        check_auc=float(roc_auc_score(labels[checking],logits[checking]))
        bce=np.logaddexp(0,logits.astype(np.float64))-labels*logits
        records.append(dict(model=family,train_end=end,training_head_auc=train_auc,
                            checking_head_auc=check_auc,training_head_bce=float(bce[learning].mean()),
                            checking_head_bce=float(bce[checking].mean()),
                            checking_xgboost_auc=float(comparison.loc[end,family]),
                            original_xgboost_auc=float(comparison.loc[end,'original_xgboost'])))
        input_path.unlink();output_path.unlink()
results=pd.DataFrame(records)
results.to_csv(RESULTS/'head_diagnostic.csv',index=False)
print(results.round(6).to_string(index=False))
print('Diagnostic only: no training, tuning, prediction blending or model selection occurred.')
