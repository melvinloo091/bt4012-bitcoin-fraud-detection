"""Best model (half-life 4, XGBoost 300 + LightGBM 150, 50/50) with network-shape features added to XGBoost only.

Network-shape features are counted from the full edge list ('TXS Edgelist.csv'), which also contains
payment links to transactions that are not in our train/test files. For every transaction:
  g_in, g_out        incoming / outgoing payment links
  g_deg              all distinct linked transactions
  g_hop2             transactions reachable in two steps
  g_nbdeg_mean/max   average / largest link count among linked transactions
  g_nb_chain         share of linked transactions that are 'chain links' (exactly 1 in, 1 out)
  g_nb_out, g_nb_in  average outgoing / incoming links of linked transactions
No labels are used to build them, and none of them tracks time (rank correlation with time step <= 0.12).
LightGBM, recency weights, tree counts and the 50/50 average are unchanged from the 0.95484 model.

Practice exams (15 rolling 14-step exams): 0.9505 vs 0.9507 for the best model (better in 5 of 15) - a tie.
Run from the Kaggle project folder:
    python archive/results/claude_experiments/2026_10_16/scripts/make_graphshape_xgb.py
Output: archive/results/claude_experiments/submissions/submission_graphshape_xgb.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT = Path.cwd(); DATA = PROJECT / "Data Set"
OUT = PROJECT / "archive" / "results" / "claude_experiments" / "submissions"; OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(DATA / "Train Data.csv"); test = pd.read_csv(DATA / "Test Data.csv")
edges = pd.read_csv(DATA / "TXS Edgelist.csv"); edges.columns = ["a", "b"]
sample = pd.read_csv(DATA / "Sample Submission" / "BT4012 Competition Submission.csv")
FEATURES = [f"feat_{i}" for i in range(1, 166)]
labels = pd.to_numeric(train["label"], errors="coerce")
known = train.loc[labels.notna()].copy(); known["label"] = labels.loc[labels.notna()].astype(int)

# ---- network-shape features ----
ids = pd.Index(pd.unique(np.r_[edges.a.values, edges.b.values, train.txId.values, test.txId.values])); n = len(ids)
A = sp.csr_matrix((np.ones(len(edges)), (ids.get_indexer(edges.a), ids.get_indexer(edges.b))), shape=(n, n))
U = ((A + A.T) > 0).astype(float).tocsr()
g_out = np.asarray(A.sum(1)).ravel(); g_in = np.asarray(A.sum(0)).ravel(); g_deg = np.asarray(U.sum(1)).ravel()
two = U @ U; two.setdiag(0); two.eliminate_zeros(); g_hop2 = np.asarray((two > 0).sum(1)).ravel()
safe = np.maximum(g_deg, 1)
nb_max = np.array([g_deg[U.indices[U.indptr[i]:U.indptr[i + 1]]].max() if U.indptr[i + 1] > U.indptr[i] else 0 for i in range(n)])
chain = ((g_in == 1) & (g_out == 1)).astype(float)
G = pd.DataFrame({"g_in": g_in, "g_out": g_out, "g_deg": g_deg, "g_hop2": g_hop2,
                  "g_nbdeg_mean": np.asarray(U @ g_deg).ravel() / safe, "g_nbdeg_max": nb_max,
                  "g_nb_chain": np.asarray(U @ chain).ravel() / safe,
                  "g_nb_out": np.asarray(U @ g_out).ravel() / safe, "g_nb_in": np.asarray(U @ g_in).ravel() / safe}, index=ids)
GRAPH = list(G.columns)
known = known.join(G, on="txId"); test = test.join(G, on="txId")

# ---- best model recipe, graph features in XGBoost only ----
corr = known[FEATURES].rank().corrwith(known["time_step"].rank())
calendar = set(corr[corr.abs() > 0.9].index)
XF = [c for c in FEATURES if c not in calendar] + GRAPH
age = known["time_step"].max() - known["time_step"]; w = (0.5 ** (age / 4)).to_numpy(); w = w / w.mean()
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method="hist", grow_policy="lossguide", max_depth=0, max_leaves=15,
                    max_bin=127, min_child_weight=1, reg_lambda=1, objective="binary:logistic", n_jobs=2, random_state=42)
lgb = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5, min_child_weight=1e-3,
                     reg_lambda=1, objective="binary", deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=2, random_state=42)
px = xgb.fit(known[XF], known["label"], sample_weight=w).predict_proba(test[XF])[:, 1]
pl = lgb.fit(known[FEATURES], known["label"], sample_weight=w).predict_proba(test[FEATURES])[:, 1]
sub = sample[["index"]].copy(); sub["target"] = sub["index"].map(pd.Series(0.5 * px + 0.5 * pl, index=test["index"]))
assert sub["index"].equals(sample["index"]) and sub["target"].notna().all() and sub["target"].between(0, 1).all()
sub.to_csv(OUT / "submission_graphshape_xgb.csv", index=False); print("Saved", OUT / "submission_graphshape_xgb.csv", len(sub), "rows")
