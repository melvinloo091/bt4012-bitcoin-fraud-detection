## 27. A stricter way to test models, and a less-regularised XGBoost

### Why do we need a stricter test?

The look-alike blend in Section 26 looked excellent locally but scored only **0.87805** on Kaggle. Before trying anything new, we need a local test we can trust.

### The new local test: 15 rolling "practice exams"

The real task is to predict **14 future time steps** (36–49). So we create many practice exams with the same shape:

| Exam | Learn from | Predict |
|---|---|---|
| 1 | Steps 1–7 | Steps 8–21 |
| 2 | Steps 1–8 | Steps 9–22 |
| … | … | … |
| 15 | Steps 1–21 | Steps 22–35 |

We report the **average ROC-AUC across all 15 exams**. No single period can dominate the result.

### Can we trust it? We checked it against real Kaggle scores

| Model | Local average (15 exams) | Kaggle public score |
|---|---:|---:|
| Extra Trees | 0.915 | 0.917 |
| Histogram Gradient Boosting (255 bins) | 0.931 | 0.937 |
| HGB bin ensemble | 0.940 | 0.942 |
| XGBoost tuned | 0.941 | 0.943 |

For these ordinary models, the local average lands within about 0.002–0.006 of the real Kaggle score, and it puts them in the same order.

**One important exception:** the look-alike blend scored about 0.971 locally but 0.878 on Kaggle. That method compared each test transaction with the *other test transactions*. The test period behaves differently from the training period (a dark market shut down at step 43), so local exams cannot judge methods like that. For those, we add a second safety check before uploading (Section 27.3).

### What we found: we were regularising XGBoost too heavily

We tested many changes on the 15 exams: different tree sizes, numbers of bins, learning rates, feature groups, Random Forest, bagging, class weights, removing drifting features, and treating unknown labels as licit. Most made little difference or made things worse.

One change helped clearly. Our XGBoost used strong **regularisation**, which stops the model from learning patterns supported by only a few examples:

- `min_child_weight = 3` (a leaf needs enough "support" before it is created)
- `reg_lambda = 10` (strong penalty on large leaf values)

Going back to **XGBoost's own default values** (`min_child_weight = 1`, `reg_lambda = 1`) lets the model learn smaller groups of illicit transactions. This raised the local average from **0.941 to 0.952** and improved **13 of the 15 exams**.

Everything else stays the same: 15 leaves, 127 bins, learning rate 0.05, all 165 features. We use a fixed **200 rounds**, which worked well for both small and large training sets.

---

```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROLLING_EXAMS = [(k, k + 1, k + 14) for k in range(7, 22)]   # learn 1–k, predict the next 14 steps
LOWREG_ROUNDS = 200

def make_lowreg_xgboost(rounds=LOWREG_ROUNDS):
    return XGBClassifier(
        n_estimators=rounds, learning_rate=0.05, tree_method="hist",
        grow_policy="lossguide", max_depth=0, max_leaves=15, max_bin=127,
        min_child_weight=1,   # was 3
        reg_lambda=1,         # was 10
        objective="binary:logistic", n_jobs=4, random_state=RANDOM_SEED,
    )

rolling_records = []
for train_end, valid_start, valid_end in ROLLING_EXAMS:
    learning = labeled.loc[labeled["time_step"] <= train_end]
    checking = labeled.loc[labeled["time_step"].between(valid_start, valid_end)]
    assert learning["time_step"].max() < checking["time_step"].min()

    # Current best: XGBoost tuned (Section 20), with its chronological stopping rule.
    tuned_model, tuned_rounds = fit_with_chronological_stopping(make_selected_xgboost, "xgboost", learning)
    # New candidate: same model with XGBoost's default regularisation.
    lowreg_model = make_lowreg_xgboost()
    with threadpool_limits(limits=4):
        lowreg_model.fit(learning[FEATURE_COLUMNS], learning["label"])
        tuned_probabilities = tuned_model.predict_proba(checking[FEATURE_COLUMNS])[:, 1]
        lowreg_probabilities = lowreg_model.predict_proba(checking[FEATURE_COLUMNS])[:, 1]

    rolling_records.append({
        "exam": f"learn 1–{train_end}, predict {valid_start}–{valid_end}",
        "xgboost_tuned_auc": roc_auc_score(checking["label"], tuned_probabilities),
        "lowreg_xgboost_auc": roc_auc_score(checking["label"], lowreg_probabilities),
    })
    print(rolling_records[-1]["exam"], "done", flush=True)

rolling_results = pd.DataFrame(rolling_records)
rolling_results["change"] = rolling_results["lowreg_xgboost_auc"] - rolling_results["xgboost_tuned_auc"]
display(rolling_results.round(4))
print("Average over 15 exams:")
display(rolling_results[["xgboost_tuned_auc", "lowreg_xgboost_auc"]].mean().round(4))
print("Exams improved:", int((rolling_results["change"] > 0).sum()), "of", len(rolling_results))
```

---

### 27.1. Results of the 15 practice exams

| | XGBoost tuned | Less-regularised XGBoost |
|---|---:|---:|
| Average ROC-AUC (15 exams) | 0.9409 | **0.9521** |
| Exams improved | – | **13 of 15** |

The improvement appears in both easy and hard exams, so it does not depend on one lucky period. The result is also stable for different numbers of rounds: 100, 150, 200 and 300 rounds all average at least 0.950.

We also checked larger training sets (learn 1–22 up to 1–32, predict up to step 35). These periods are easier, but the new settings still scored slightly higher on average (0.9874 vs 0.9865).

If the link between local and Kaggle scores holds, we would expect a public score of roughly **0.95**. This is an estimate, not a guarantee, and it is still below 0.96.

---

### 27.2. Train on all known labels and create the submission

We train the less-regularised XGBoost on all 31,235 known labels from steps 1–35, with a fixed 200 rounds. Then we predict the 15,329 test transactions.

---

```python
final_lowreg_model = make_lowreg_xgboost()
with threadpool_limits(limits=4):
    final_lowreg_model.fit(labeled[FEATURE_COLUMNS], labeled["label"])
    lowreg_test_probabilities = final_lowreg_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

# Claude's submissions are kept in their own folder.
claude_submission_dir = PROJECT_DIR / "claude_experiments" / "submissions"
claude_submission_dir.mkdir(parents=True, exist_ok=True)

by_identifier = pd.Series(lowreg_test_probabilities, index=test["index"])
lowreg_submission = sample_submission[["index"]].copy()
lowreg_submission["target"] = lowreg_submission["index"].map(by_identifier)
assert lowreg_submission.columns.tolist() == ["index", "target"]
assert lowreg_submission["index"].equals(sample_submission["index"])
assert np.isfinite(lowreg_submission["target"].to_numpy()).all()
assert lowreg_submission["target"].between(0, 1).all()

lowreg_submission_path = claude_submission_dir / "submission_xgboost_lowreg.csv"
lowreg_submission.to_csv(lowreg_submission_path, index=False)
rolling_results.to_csv(claude_submission_dir / "xgboost_lowreg_validation.csv", index=False)
display(lowreg_submission.head())
print("Saved to:", lowreg_submission_path)
```

---

### 27.3. Safety check before uploading (no labels needed)

The local exams cannot catch every problem, as Section 26 showed. Before uploading, we compare the new test predictions with our best submitted model (XGBoost tuned, 0.94265):

- **Spearman correlation:** do both models put the test transactions in a similar order? (1 = identical order)
- **Top-group overlap:** of each model's top 6% most suspicious transactions, how many are shared?
- **Promoted from the bottom half:** of the transactions newly placed in the top 6%, what share did XGBoost tuned place in its *least* suspicious half? A large value means the new model is making big, risky changes.

This is how earlier submissions compare with XGBoost tuned:

| Submission | Spearman | Top-group overlap | Promoted from bottom half | Kaggle public |
|---|---:|---:|---:|---:|
| HGB bin ensemble | 0.956 | 0.91 | 0.0% | 0.94199 |
| Extra Trees | 0.608 | 0.76 | 0.9% | 0.91673 |
| Look-alike blend | 0.592 | 0.89 | **14.6%** | **0.87805** |

The failed look-alike blend shows a very low Spearman and many rows promoted from the bottom half. A safe candidate should look like the HGB bin ensemble row.

---

```python
from scipy.stats import spearmanr

def compare_with_reference(candidate, reference, top_share=0.06):
    n = int(top_share * len(candidate))
    candidate_top = set(np.argsort(-candidate)[:n])
    reference_top = set(np.argsort(-reference)[:n])
    newly_promoted = np.array(sorted(candidate_top - reference_top))
    reference_rank = pd.Series(reference).rank(pct=True).to_numpy()
    return {
        "spearman": spearmanr(candidate, reference)[0],
        "top_group_overlap": len(candidate_top & reference_top) / n,
        "promoted_from_bottom_half": float(np.mean(reference_rank[newly_promoted] < 0.5)) if len(newly_promoted) else 0.0,
    }

reference_submission = pd.read_csv(PROJECT_DIR / "submissions" / "submission_xgboost_tuned.csv")
assert reference_submission["index"].equals(lowreg_submission["index"])
safety = compare_with_reference(lowreg_submission["target"].to_numpy(), reference_submission["target"].to_numpy())
display(pd.Series(safety).round(3))
```

---

### 27.4. Interpretation and Kaggle result

The safety check gives Spearman **0.97**, top-group overlap **0.94** and **0%** promoted from the bottom half. The new model ranks the test transactions much like XGBoost tuned, with small adjustments. That is the profile of a normal model improvement, not a risky change like the look-alike blend.

| Submission | Public ROC-AUC |
|---|---:|
| XGBoost tuned | 0.94265 |
| Look-alike blend | 0.87805 |
| **Less-regularised XGBoost** | *Pending* |

**Assistance note:** Claude (Anthropic) designed the rolling evaluation, ran the experiments and drafted this section's code and explanations.