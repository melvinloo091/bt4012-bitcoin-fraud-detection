import pandas as pd, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from common import train, test
E=pd.read_csv('/home/claude/w/data/edges.csv'); E.columns=['a','b']
ids=pd.Index(pd.unique(np.r_[E.a.values,E.b.values,train.txId.values,test.txId.values])); n=len(ids)
ia=ids.get_indexer(E.a); ib=ids.get_indexer(E.b)
A=sp.csr_matrix((np.ones(len(E)),(ia,ib)),shape=(n,n))   # directed a->b
U=((A+A.T)>0).astype(float).tocsr()
outd=np.asarray(A.sum(1)).ravel(); ind=np.asarray(A.sum(0)).ravel(); deg=np.asarray(U.sum(1)).ravel()
ncomp,lab=connected_components(U,directed=False); csize=np.bincount(lab)[lab]
two=(U@U); two.setdiag(0); two.eliminate_zeros(); hop2=np.asarray((two>0).sum(1)).ravel()
nb_deg_mean=np.asarray(U@deg).ravel()/np.maximum(deg,1)
nb_deg_max=np.array([U.indices[U.indptr[i]:U.indptr[i+1]].size and deg[U.indices[U.indptr[i]:U.indptr[i+1]]].max() or 0 for i in range(n)])
chain=((ind==1)&(outd==1)).astype(float); nb_chain=np.asarray(U@chain).ravel()/np.maximum(deg,1)
# chain run length: size of component in subgraph of nodes with in<=1,out<=1 (peel-chain-ish)
m=(ind<=1)&(outd<=2); S=U.multiply(m[:,None]).multiply(m[None,:]).tocsr(); _,cl=connected_components(S,directed=False); csz=np.bincount(cl)[cl]; csz[~m]=0
# fan-in / fan-out of neighbours
nb_out_mean=np.asarray(U@outd).ravel()/np.maximum(deg,1); nb_in_mean=np.asarray(U@ind).ravel()/np.maximum(deg,1)
G=pd.DataFrame({'g_in':ind,'g_out':outd,'g_deg':deg,'g_comp':np.log1p(csize),'g_hop2':hop2,'g_nbdeg_mean':nb_deg_mean,'g_nbdeg_max':nb_deg_max,
                'g_nb_chain':nb_chain,'g_chainrun':np.log1p(csz),'g_nb_out':nb_out_mean,'g_nb_in':nb_in_mean},index=ids)
G.index.name='txId'; G.to_parquet('/home/claude/w/exp/gfeat.pq'); print(G.describe().T.round(2))
