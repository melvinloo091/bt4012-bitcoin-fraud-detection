from base import *
import sys
from preupload import compare
BIGW=[(k,k+1,35) for k in range(22,33)]
ref=pickle.load(open(f'{P}/xgb_tuned.pkl','rb')); pair=pickle.load(open(f'{P}/pair.pkl','rb'))
def m(c,W): return np.mean([roc_auc_score(labeled.loc[c[k].index].label,c[k]) for (k,a,b) in W])
rows=[]
for n in ['xgb_tuned','pair']+sys.argv[1:]:
    p=f'{P}/{n}.pkl'
    if not os.path.exists(p): continue
    c=pickle.load(open(p,'rb'))
    if any(k not in c for (k,a,b) in WINDOWS+BIGW): continue
    r={'model':n,'rolling15':m(c,WINDOWS),'big11':m(c,BIGW),'hard5(k17-21)':m(c,[w for w in WINDOWS if w[0]>=17])}
    if 'test' in c:
        o,_=compare(c['test'].to_numpy(),ref['test'].to_numpy()); r['spearman_vs_tuned']=o['spearman']; r['promoted']=o['promoted_from_bottom_half']
        o2,_=compare(c['test'].to_numpy(),pair['test'].to_numpy()); r['spearman_vs_pair']=o2['spearman']
    rows.append(r)
print(pd.DataFrame(rows).set_index('model').round(4).to_string())
