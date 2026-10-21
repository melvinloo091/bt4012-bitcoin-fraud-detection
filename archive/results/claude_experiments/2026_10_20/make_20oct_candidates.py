"""The three Kaggle candidates of 20 Oct 2026, built from the raw competition files.

All three start from the 0.95524 model: XGBoost (300 trees, no calendar features, + 9 network-shape counts) and
LightGBM (150 trees, 165 features), averaged 50/50, recency half-life 4.

  submission_labnb_xgb.csv            XGBoost gets 7 labelled-neighbour counts instead of the 9 network-shape counts.
                                      The transactions missing from the files are exactly the 46,668 unlabelled steps 36-49
                                      transactions, so "how many partners carry a label" means the same in training (known label)
                                      and test (in the test file). Label presence only, never label values.
                                      Practice exams: 0.9507 vs 0.9507 for the 0.95484 model (better in 8 of 15).
  submission_graphshape22_xgb.csv     XGBoost gets 22 network-shape counts (the 9 + 13 more, incl. triangles, k-core, PageRank).
                                      Practice exams: 0.9503 (better in 6 of 15).
  submission_graphshape_hl345_avg.csv The 0.95524 model at half-lives 3, 4 and 5, averaged (a hedge for the private leaderboard).

Run from the Kaggle project folder (needs networkx for the 22-feature version):
    python archive/results/claude_experiments/2026_10_20/make_20oct_candidates.py
Output: archive/results/claude_experiments/submissions/
"""
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, networkx as nx
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

# ---- 9 network-shape counts (same as make_graphshape_xgb.py) ----
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
G9 = list(G.columns)

# ---- 13 more network-shape counts (22 in total) ----
F = {}; inN = A.T.tocsr(); outN = A
F["g_in_nb_out"] = np.asarray(inN @ g_out).ravel() / np.maximum(g_in, 1)
F["g_out_nb_in"] = np.asarray(outN @ g_in).ravel() / np.maximum(g_out, 1)
F["g_in_nb_in"] = np.asarray(inN @ g_in).ravel() / np.maximum(g_in, 1)
F["g_out_nb_out"] = np.asarray(outN @ g_out).ravel() / np.maximum(g_out, 1)
F["g_leaf_nbrs"] = np.asarray(U @ (g_deg == 1).astype(float)).ravel(); F["g_leaf_share"] = F["g_leaf_nbrs"] / safe
F["g_io_ratio"] = np.log1p(g_out) - np.log1p(g_in)
B2 = (two > 0).astype(float); h2 = np.asarray(B2.sum(1)).ravel()
F["g_hop2_meandeg"] = np.asarray(B2 @ g_deg).ravel() / np.maximum(h2, 1); F["g_hop2_per_nbr"] = h2 / safe
Gx = nx.from_scipy_sparse_array(U); tri = nx.triangles(Gx); F["g_tri"] = np.array([tri[i] for i in range(n)])
F["g_clust"] = F["g_tri"] / np.maximum(g_deg * (g_deg - 1) / 2, 1)
core = nx.core_number(Gx); F["g_core"] = np.array([core[i] for i in range(n)])
P = sp.diags(1 / np.maximum(g_out, 1)) @ A; pr = np.ones(n) / n
for _ in range(50): pr = 0.15 / n + 0.85 * (P.T @ pr + pr[g_out == 0].sum() / n)
F["g_pagerank"] = np.log(pr * n)
G = G.join(pd.DataFrame(F, index=ids)); G22 = list(G.columns)

# ---- 7 labelled-neighbour counts ----
has_label = set(known.txId) | set(test.txId)
E = edges.assign(la=edges.a.isin(has_label).astype(int), lb=edges.b.isin(has_label).astype(int))
L = E.groupby("a").agg(out_n=("lb", "size"), out_lab=("lb", "sum")).join(
    E.groupby("b").agg(in_n=("la", "size"), in_lab=("la", "sum")), how="outer").fillna(0)
L["out_unl"] = L.out_n - L.out_lab; L["in_unl"] = L.in_n - L.in_lab
L["nb_lab"] = L.out_lab + L.in_lab; L["nb_unl"] = L.out_unl + L.in_unl
L["nb_lab_frac"] = L.nb_lab / np.maximum(L.nb_lab + L.nb_unl, 1)
L = L[["in_lab", "in_unl", "out_lab", "out_unl", "nb_lab", "nb_unl", "nb_lab_frac"]].astype(float).add_prefix("l_")
LF = list(L.columns)

known = known.join(G, on="txId").join(L, on="txId"); test = test.join(G, on="txId").join(L, on="txId")
known[LF] = known[LF].fillna(0); test[LF] = test[LF].fillna(0)
corr = known[FEATURES].rank().corrwith(known["time_step"].rank()); calendar = set(corr[corr.abs() > 0.9].index)
XF = [c for c in FEATURES if c not in calendar]

def weights(hl):
    w = (0.5 ** ((known.time_step.max() - known.time_step) / hl)).to_numpy(); return w / w.mean()
def xgb_pred(cols, hl):
    m = XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method="hist", grow_policy="lossguide", max_depth=0, max_leaves=15,
                      max_bin=127, min_child_weight=1, reg_lambda=1, objective="binary:logistic", n_jobs=2, random_state=42)
    return m.fit(known[cols], known.label, sample_weight=weights(hl)).predict_proba(test[cols])[:, 1]
def lgb_pred(hl):
    m = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=63, min_child_samples=5, min_child_weight=1e-3,
                       reg_lambda=1, objective="binary", deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=2, random_state=42)
    return m.fit(known[FEATURES], known.label, sample_weight=weights(hl)).predict_proba(test[FEATURES])[:, 1]
def save(pred, name):
    sub = sample[["index"]].copy(); sub["target"] = sub["index"].map(pd.Series(pred, index=test["index"]))
    assert sub["index"].equals(sample["index"]) and sub["target"].notna().all() and sub["target"].between(0, 1).all()
    sub.to_csv(OUT / name, index=False); print("Saved", OUT / name)

pl = {hl: lgb_pred(hl) for hl in (3, 4, 5)}
save(0.5 * xgb_pred(XF + LF, 4) + 0.5 * pl[4], "submission_labnb_xgb.csv")
save(0.5 * xgb_pred(XF + G22, 4) + 0.5 * pl[4], "submission_graphshape22_xgb.csv")
save(np.mean([0.5 * xgb_pred(XF + G9, hl) + 0.5 * pl[hl] for hl in (3, 4, 5)], axis=0), "submission_graphshape_hl345_avg.csv")
