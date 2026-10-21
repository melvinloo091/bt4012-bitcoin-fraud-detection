## 28. Refinement: average two less-regularised models

### What are we doing?

Section 27 showed that lighter regularisation helps. We now test whether **averaging two different less-regularised models** helps further. Different libraries build trees in slightly different ways, so their mistakes are not identical. Averaging can cancel out some of them.

The two models:

1. **Less-regularised XGBoost without the six time-tracking features** (the Section 24 rule plus the Section 27 settings, 200 rounds).
2. **Less-regularised LightGBM** with 63 bins, at least 5 transactions per leaf, `reg_lambda = 1`, 15 leaves and 150 rounds.

Final score = **50% model 1 + 50% model 2**.

### How did we choose this?

We tested 14 settings and several combinations on the same 15 practice exams:

- **No single change** to the less-regularised XGBoost moved its average much above 0.952. The changes tested were bins, tree size, class weights, feature sampling, multiple seeds, and LightGBM and HGB versions.
- **The best combination was this two-model average.** It improved the average by about 0.003 and beat the single less-regularised XGBoost in 10 of 15 exams.

Because we chose the best of several combinations, the true improvement may be a little smaller than measured.

### Do these models use the unknown (unlabelled) transactions?

**No.** Both models learn only from the 31,235 transactions with known labels. We tested two ways of using the 110,537 unknown transactions, and both made the local average worse:

| Way of using unknown transactions | Local average (15 exams) |
|---|---:|
| Not used (XGBoost tuned) | 0.941 |
| Treat unknown as licit, full weight | 0.869 |
| Treat unknown as licit, weight 0.1 | 0.895 |

A separate experiment (the fraud-guided autoencoder) used unknown transactions to learn new features. It also scored lower than the original XGBoost (0.851 vs 0.928 on its four checks).

Unknown transactions still influence the model indirectly. Features 94–165 summarise each transaction's neighbours in the graph, and those neighbours include unknown transactions.

---

```python
from lightgbm import LGBMClassifier

def make_lowreg_lightgbm(rounds=150):
    return LGBMClassifier(
        n_estimators=rounds, learning_rate=0.05, num_leaves=15, max_bin=63,
        min_child_samples=5, min_child_weight=1e-3, reg_lambda=1,
        objective="binary", deterministic=True, force_col_wise=True,
        verbosity=-1, n_jobs=4, random_state=RANDOM_SEED,
    )

def fit_lowreg_pair(learning):
    """Model 1: less-regularised XGBoost without time-tracking features. Model 2: less-regularised LightGBM."""
    xgb_columns, removed, _ = select_time_review_features(learning, filter_time_features=True)
    xgb_model = make_lowreg_xgboost()
    lgb_model = make_lowreg_lightgbm()
    with threadpool_limits(limits=4):
        xgb_model.fit(learning[xgb_columns], learning["label"])
        lgb_model.fit(learning[FEATURE_COLUMNS], learning["label"])
    return xgb_model, xgb_columns, lgb_model

def predict_lowreg_pair(xgb_model, xgb_columns, lgb_model, rows):
    with threadpool_limits(limits=4):
        xgb_probabilities = xgb_model.predict_proba(rows[xgb_columns])[:, 1]
        lgb_probabilities = lgb_model.predict_proba(rows[FEATURE_COLUMNS])[:, 1]
    return 0.5 * xgb_probabilities + 0.5 * lgb_probabilities

pair_records = []
for train_end, valid_start, valid_end in ROLLING_EXAMS:
    learning = labeled.loc[labeled["time_step"] <= train_end]
    checking = labeled.loc[labeled["time_step"].between(valid_start, valid_end)]
    pair_probabilities = predict_lowreg_pair(*fit_lowreg_pair(learning), checking)
    pair_records.append({"exam": f"learn 1–{train_end}, predict {valid_start}–{valid_end}",
                         "two_model_average_auc": roc_auc_score(checking["label"], pair_probabilities)})
    print(pair_records[-1]["exam"], "done", flush=True)

pair_results = rolling_results.merge(pd.DataFrame(pair_records), on="exam")
pair_results["change_vs_lowreg_xgboost"] = pair_results["two_model_average_auc"] - pair_results["lowreg_xgboost_auc"]
display(pair_results.round(4))
print("Average over 15 exams:")
display(pair_results[["xgboost_tuned_auc", "lowreg_xgboost_auc", "two_model_average_auc"]].mean().round(4))
print("Exams improved vs less-regularised XGBoost:",
      int((pair_results["change_vs_lowreg_xgboost"] > 0).sum()), "of", len(pair_results))
```

---

```python
final_pair = fit_lowreg_pair(labeled)
pair_test_probabilities = predict_lowreg_pair(*final_pair, test)
print("Time-tracking features removed:", [c for c in FEATURE_COLUMNS if c not in final_pair[1]])

by_identifier = pd.Series(pair_test_probabilities, index=test["index"])
pair_submission = sample_submission[["index"]].copy()
pair_submission["target"] = pair_submission["index"].map(by_identifier)
assert pair_submission.columns.tolist() == ["index", "target"]
assert pair_submission["index"].equals(sample_submission["index"])
assert np.isfinite(pair_submission["target"].to_numpy()).all()
assert pair_submission["target"].between(0, 1).all()

pair_submission_path = claude_submission_dir / "submission_xgboost_lightgbm_lowreg.csv"
pair_submission.to_csv(pair_submission_path, index=False)
pair_results.to_csv(claude_submission_dir / "xgboost_lightgbm_lowreg_validation.csv", index=False)
print("Saved to:", pair_submission_path)

# Safety check against our best submitted model (Section 27.3).
safety = compare_with_reference(pair_submission["target"].to_numpy(), reference_submission["target"].to_numpy())
display(pd.Series(safety).round(3))
```

---

### 28.1. Results

| | XGBoost tuned | Less-regularised XGBoost (Section 27) | Two-model average |
|---|---:|---:|---:|
| Average ROC-AUC (15 exams) | 0.9409 | 0.9521 | **0.9550** |
| Exams improved vs Section 27 | – | – | 10 of 15 |

The biggest gains are in the hard exams (for example, 16–29: 0.9203 → 0.9390; 19–32: 0.9056 → 0.9168). The easy exams are almost unchanged.

**Safety check vs XGBoost tuned:** Spearman 0.971, top-group overlap 0.92, **0%** promoted from the bottom half. This is a normal, low-risk change.

**What score should we expect?** For ordinary models, the local average has landed close to the Kaggle score (within about 0.002–0.006). A local 0.955 therefore suggests roughly **0.95–0.96**. Reaching 0.96 is possible but **not guaranteed**.

### 28.2. Kaggle results

| Submission | Public ROC-AUC |
|---|---:|
| XGBoost tuned | 0.94265 |
| Less-regularised XGBoost (Section 27) | Not submitted |
| **Two-model average (Section 28)** | **0.94701** (best so far) |

**Assistance note:** Claude (Anthropic) ran the refinement experiments and drafted this section's code and explanations.


### 28.3. What the Kaggle result teaches us

The two-model average improved our best public score from **0.94265 to 0.94701** (+0.0044). Our local exams had predicted a larger gain (+0.0146).

We now have six Kaggle scores to compare with local results:

| Submission | Local average (15 exams) | Kaggle public | Local change vs XGBoost tuned | Kaggle change |
|---|---:|---:|---:|---:|
| Extra Trees | 0.9147 | 0.9167 | −0.0262 | −0.0259 |
| HGB (255 bins) | 0.9310 | 0.9365 | −0.0099 | −0.0061 |
| HGB bin ensemble | 0.9403 | 0.9420 | −0.0006 | −0.0007 |
| XGBoost tuned | 0.9409 | 0.9427 | 0 | 0 |
| **Two-model average** | 0.9555 | **0.9470** | +0.0146 | +0.0044 |
| Look-alike blend | 0.9713 | 0.8780 | +0.0304 | −0.0646 |

- **The local exams put the five ordinary models in the correct order.** They are useful for deciding *whether* a change helps.
- **They overstate the size of improvements that come from the hard exams.** Most of this model's local gain came from periods where new kinds of illicit activity appeared (steps 24, 29, 31). Only about a third of that gain carried over to the test period, where the main change is the dark market shutdown at step 43.
- **Rule of thumb from now on:** expect roughly one third of a local gain to appear on Kaggle, and treat any estimate as uncertain.