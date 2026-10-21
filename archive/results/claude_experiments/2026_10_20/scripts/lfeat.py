"""Labelled-neighbour features: for every transaction, how many of its payment partners carry a label.
Train rows: a partner is 'labelled' if it is a known (licit/illicit) training row.
Test rows:  a partner is 'labelled' if it is in the test file (the test file holds exactly the labelled steps 36-49 rows;
            transactions missing from all files are the unlabelled steps 36-49 rows - verified: 46,668 missing = 203,769 - 157,101,
            and missing transactions only ever link to test-period transactions).
Only label PRESENCE is used, never label values."""
import pandas as pd, numpy as np
tr=pd.read_parquet('/home/claude/w/data/train.pq',columns=['txId','label']); te=pd.read_parquet('/home/claude/w/data/test.pq',columns=['txId'])
E=pd.read_csv('/home/claude/w/data/edges.csv'); E.columns=['a','b']
labelled=set(tr.txId[tr.label.notna()])|set(te.txId)
E['la']=E.a.isin(labelled).astype(int); E['lb']=E.b.isin(labelled).astype(int)
# out-links of a: partner b; in-links of b: partner a
out=E.groupby('a').agg(out_n=('lb','size'),out_lab=('lb','sum'))
inn=E.groupby('b').agg(in_n=('la','size'),in_lab=('la','sum'))
F=out.join(inn,how='outer').fillna(0); F.index.name='txId'
F['out_unl']=F.out_n-F.out_lab; F['in_unl']=F.in_n-F.in_lab
F['nb_lab']=F.out_lab+F.in_lab; F['nb_unl']=F.out_unl+F.in_unl
F['nb_lab_frac']=F.nb_lab/np.maximum(F.nb_lab+F.nb_unl,1)
F=F[['in_lab','in_unl','out_lab','out_unl','nb_lab','nb_unl','nb_lab_frac']].astype(float)
F.columns=['l_'+c for c in F.columns]
F.to_parquet('lfeat.pq'); print(F.describe().T.round(3))
