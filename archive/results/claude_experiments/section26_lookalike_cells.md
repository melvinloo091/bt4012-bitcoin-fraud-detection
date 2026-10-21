## 26. Candidate: add a "look-alike" model to the time-filtered XGBoost

### What did we notice?

Our hardest check (learn from steps 1–21, predict steps 22–35) stays around **0.92**. To see why, we looked at each time step separately.

Most steps are almost perfect. The problem steps are **24, 29 and 31**. In those steps, many illicit transactions receive a predicted probability close to **0**. They look different from the illicit transactions the model saw during training. They are a *new kind* of illicit activity.

The real test period (steps 36–49) is likely to contain new kinds of illicit activity too. This would explain why our public score (0.94) is lower than our easy local checks (0.99).

### What do these new illicit transactions have in common?

They look **almost identical to each other**.

Within a time step, we measured how far each transaction is from its closest look-alike, using all 165 features. A small distance means "there is another transaction in this period that looks nearly the same".

| | Distance to the closest look-alike (median) |
|---|---:|
| Licit transactions | 1.73 |
| Illicit transactions | 0.35 |

Illicit transactions often come in **batches of near-copies**, which may be produced by the same automated activity. Licit transactions are usually more varied.

This pattern does not depend on *which* kind of illicit activity it is. So it can still help when a new kind appears that the main model has never seen.

### Is this allowed, and is it fair?

- We use **only the supplied features**. No labels are used to measure distances.
- Each transaction is compared only with transactions **from the same time step**.
- We compare each training transaction with other **labelled** training transactions. The test file contains only the transactions Kaggle scores, so this makes training and test measurements comparable.
- The scaling step learns only from training rows.

### What are we doing?

1. **Measure look-alike distances.** For every transaction, we record the distance to its 1st, 2nd, 3rd, 5th, 10th, 20th and 50th closest look-alike in the same time step.
2. **Train a small look-alike model.** A logistic regression learns how these seven distances relate to illicit labels.
3. **Blend.** Final score = **80% time-filtered XGBoost + 20% look-alike model**.

The XGBoost part is exactly the Section 24 model. We do not change it.

---

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

LOOKALIKE_COUNTS = (1, 2, 3, 5, 10, 20, 50)
LOOKALIKE_COLUMNS = [f"lookalike_distance_{k}" for k in LOOKALIKE_COUNTS]

def add_lookalike_distances(rows, scaling_rows):
    """For each transaction, measure how close its nearest look-alikes are.

    Look-alikes are searched only among transactions in the SAME time step.
    No labels are used. The scaler learns only from `scaling_rows`.
    """
    scaler = StandardScaler().fit(scaling_rows[FEATURE_COLUMNS])
    scaled = np.clip(scaler.transform(rows[FEATURE_COLUMNS]), -10, 10)
    steps = rows["time_step"].to_numpy()
    distances = np.zeros((len(rows), len(LOOKALIKE_COUNTS)))

    for step in np.unique(steps):
        positions = np.flatnonzero(steps == step)
        neighbours = min(max(LOOKALIKE_COUNTS) + 1, len(positions))
        # The closest match is always the transaction itself, so position 0 is skipped.
        found, _ = NearestNeighbors(n_neighbors=neighbours).fit(
            scaled[positions]
        ).kneighbors(scaled[positions])
        for j, k in enumerate(LOOKALIKE_COUNTS):
            distances[positions, j] = found[:, min(k, neighbours - 1)]

    # log1p shrinks very large distances so a few extreme values do not dominate.
    return pd.DataFrame(np.log1p(distances), index=rows.index, columns=LOOKALIKE_COLUMNS)

def make_lookalike_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))

# The population matches the hidden test set: labelled training rows plus test rows.
lookalike_population = pd.concat(
    [labeled.drop(columns="label"), test.drop(columns="index")],
    keys=["train", "test"],
)
lookalike_all = add_lookalike_distances(lookalike_population, scaling_rows=labeled)
labeled_lookalike = labeled.join(lookalike_all.loc["train"].set_index(labeled.index))

summary = labeled_lookalike.groupby("label")[LOOKALIKE_COLUMNS].median().T
summary.columns = ["Licit (median)", "Illicit (median)"]
display(summary.round(3))
```

---

### 26.1. Check the blend on later periods

We repeat the three checks used in Section 24. We also add **three fresh checks** that were never used when choosing the 20% weight. Fresh checks give a fairer idea of whether the improvement is real.

For each check:

- XGBoost learns only from the training steps, with the same internal stopping rule as Section 24.
- The look-alike model learns only from the training steps.
- Validation answers are used **only for scoring**.

---

```python
BLEND_WEIGHT = 0.2   # 80% XGBoost + 20% look-alike model

LOOKALIKE_CHECKS = [
    # Original checks
    (14, 15, 28), (21, 22, 35), (28, 29, 35),
    # Fresh checks not used to choose the blend weight
    (10, 11, 24), (17, 18, 31), (19, 20, 33),
]

lookalike_records = []
for train_end, valid_start, valid_end in LOOKALIKE_CHECKS:
    learning = labeled.loc[labeled["time_step"] <= train_end]
    checking = labeled.loc[labeled["time_step"].between(valid_start, valid_end)]
    assert learning["time_step"].max() < checking["time_step"].min()

    # Model 1: the existing time-filtered XGBoost from Section 24.
    xgb_model, xgb_columns, _, xgb_rounds, _ = fit_time_review_model(
        learning, filter_time_features=True
    )
    with threadpool_limits(limits=4):
        xgb_probabilities = xgb_model.predict_proba(checking[xgb_columns])[:, 1]

    # Model 2: look-alike distances. The scaler only learns from the training period.
    period_rows = pd.concat([learning, checking])
    period_distances = add_lookalike_distances(period_rows.drop(columns="label"), scaling_rows=learning)
    # Each row's look-alikes come from its own time step, so the result does not depend
    # on which other periods are included.
    lookalike_model = make_lookalike_model().fit(
        period_distances.loc[learning.index], learning["label"]
    )
    lookalike_probabilities = lookalike_model.predict_proba(
        period_distances.loc[checking.index]
    )[:, 1]

    blended = (1 - BLEND_WEIGHT) * xgb_probabilities + BLEND_WEIGHT * lookalike_probabilities
    lookalike_records.append({
        "validation_steps": f"{valid_start}–{valid_end}",
        "fresh_check": (train_end, valid_start, valid_end) not in TIME_FILTER_CHECKS,
        "xgboost_auc": roc_auc_score(checking["label"], xgb_probabilities),
        "lookalike_only_auc": roc_auc_score(checking["label"], lookalike_probabilities),
        "blend_auc": roc_auc_score(checking["label"], blended),
    })
    print(f"Validate {valid_start}–{valid_end}: XGBoost {lookalike_records[-1]['xgboost_auc']:.4f} "
          f"-> blend {lookalike_records[-1]['blend_auc']:.4f}", flush=True)

lookalike_results = pd.DataFrame(lookalike_records)
lookalike_results["change"] = lookalike_results["blend_auc"] - lookalike_results["xgboost_auc"]
display(lookalike_results.round(4))
print("Average over all six checks:")
display(lookalike_results[["xgboost_auc", "blend_auc"]].mean().round(4))
```

---

### 26.2. What do the results mean?

| Validation steps | Fresh check? | Time-filtered XGBoost | Blend | Change |
|---|---|---:|---:|---:|
| 15–28 | No | 0.9675 | 0.9737 | +0.0062 |
| 22–35 | No | 0.9214 | **0.9631** | **+0.0417** |
| 29–35 | No | 0.9955 | 0.9933 | −0.0021 |
| 11–24 | Yes | 0.9596 | 0.9747 | +0.0150 |
| 18–31 | Yes | 0.9228 | **0.9592** | **+0.0364** |
| 20–33 | Yes | 0.8946 | **0.9571** | **+0.0625** |
| **Average** | | 0.9436 | **0.9702** | +0.0266 |

- The blend improves **five of six** checks.
- The largest gains appear in the **hard periods**, where new kinds of illicit activity appear. That is the situation we expect in the test period.
- The easy recent check (29–35) drops slightly (−0.002), because that period has few new patterns.
- The look-alike model on its own scores only about 0.89–0.90. It is not a replacement for XGBoost. It is a **safety net** for illicit transactions that XGBoost does not recognise.

These are still development checks. They do not guarantee a Kaggle score above 0.96. They are, however, the first change to give a large gain on the hard periods, including fresh ones.

---

### 26.3. Train on all known labels and create the submission

1. Train the time-filtered XGBoost on all 31,235 known labels, using exactly the rule from Section 24 (6 time-related features removed, 305 rounds).
2. Train the look-alike model on all known labels.
3. Compute look-alike distances for the test transactions. Each one is compared only with other test transactions from the same time step.
4. Blend 80% / 20% and save **`submission_xgboost_lookalike_blend.csv`** in `claude_experiments/submissions/`. Claude's files are kept in that separate folder so they do not mix with the existing submissions.

The file contains probabilities, not 0/1 decisions.

---

```python
# 1. Time-filtered XGBoost trained on all known labels (same rule as Section 24).
final_xgb, final_xgb_columns, final_removed, final_rounds, _ = fit_time_review_model(
    labeled, filter_time_features=True
)
with threadpool_limits(limits=4):
    final_xgb_probabilities = final_xgb.predict_proba(test[final_xgb_columns])[:, 1]

# 2. Look-alike model trained on all known labels.
final_lookalike_model = make_lookalike_model().fit(
    labeled_lookalike[LOOKALIKE_COLUMNS], labeled_lookalike["label"]
)
test_lookalike = lookalike_all.loc["test"].set_index(test.index)
final_lookalike_probabilities = final_lookalike_model.predict_proba(
    test_lookalike[LOOKALIKE_COLUMNS]
)[:, 1]

# 3. Blend and save.
final_blend = (1 - BLEND_WEIGHT) * final_xgb_probabilities + BLEND_WEIGHT * final_lookalike_probabilities
# Claude's submissions are kept in their own folder so they do not mix with existing files.
claude_submission_dir = PROJECT_DIR / "claude_experiments" / "submissions"
claude_submission_dir.mkdir(parents=True, exist_ok=True)

by_identifier = pd.Series(final_blend, index=test["index"])
lookalike_submission = sample_submission[["index"]].copy()
lookalike_submission["target"] = lookalike_submission["index"].map(by_identifier)
assert lookalike_submission.columns.tolist() == ["index", "target"]
assert lookalike_submission["index"].equals(sample_submission["index"])
assert np.isfinite(lookalike_submission["target"].to_numpy()).all()
assert lookalike_submission["target"].between(0, 1).all()

lookalike_submission_path = claude_submission_dir / "submission_xgboost_lookalike_blend.csv"
lookalike_submission.to_csv(lookalike_submission_path, index=False)
lookalike_results.to_csv(claude_submission_dir / "xgboost_lookalike_blend_validation.csv", index=False)
display(lookalike_submission.head())
print("Submission rows:", len(lookalike_submission))
print("Saved to:", lookalike_submission_path)

print("Removed time-related features:", final_removed)
print("XGBoost training rounds:", final_rounds)
print("Correlation with the time-filtered XGBoost alone:",
      round(np.corrcoef(final_blend, final_xgb_probabilities)[0, 1], 4))
```

---

### 26.5. How does the blend compare with our previous submissions?

We retrained every submitted model on the same six chronological checks. The **hard checks** are the three periods where new kinds of illicit activity appear (22–35, 18–31, 20–33).

| Model | Average of 6 checks | Average of 3 hard checks | Kaggle public score |
|---|---:|---:|---:|
| Extra Trees | 0.9092 | 0.8471 | 0.91673 |
| Histogram Gradient Boosting (255 bins) | 0.9332 | 0.9031 | 0.93654 |
| LightGBM tuned | 0.9408 | 0.9066 | Not submitted |
| HGB bin ensemble | 0.9429 | 0.9125 | 0.94199 |
| XGBoost time-filtered | 0.9436 | 0.9129 | Not submitted |
| XGBoost tuned | 0.9448 | 0.9142 | **0.94265** |
| **XGBoost time-filtered + look-alike (this section)** | **0.9702** | **0.9598** | *Pending* |

**Can we trust these local checks?** Our local checks put the four submitted models in **exactly the same order** as Kaggle did: Extra Trees < HGB < HGB bin ensemble < XGBoost. This suggests our checks measure what Kaggle measures, especially the hard checks.

**What does this predict?** Every previous change moved the hard-check average by about 0.001–0.06. The look-alike blend moves it by about **+0.047** compared with our best submitted model. That looked like the largest improvement so far. **However, the actual public score was 0.87805. See Section 26.7 for why these checks were misleading.**

**Does the choice of main model matter?** Not much. Adding the same 20% look-alike model to XGBoost tuned, the HGB bin ensemble or an average of the three gave 0.9703–0.9707 on all six checks. The improvement comes from the look-alike model, not from which tree model it is paired with. We keep the time-filtered XGBoost because it was already prepared and reviewed in Section 24.

---

### 26.6. Record the Kaggle result

| Submission | Public ROC-AUC |
|---|---:|
| Original Histogram Gradient Boosting | 0.93654 |
| Extra Trees | 0.91673 |
| Histogram Gradient Boosting bin ensemble | 0.94199 |
| Original tuned XGBoost | 0.94265 |
| Time-filtered XGBoost | Not submitted |
| Time-filtered XGBoost + look-alike blend | **0.87805** (worst so far) |

Remember that the public score uses about 30% of the test rows, and the final ranking uses the other 70%.

**Assistance note:** Claude (Anthropic) helped find the look-alike pattern, ran the validation experiments and drafted this section's code and explanations. This should also be added to the assistance acknowledgement at the end of the notebook and in the report.

---

### 26.7. Why the look-alike blend failed (public score 0.87805)

The blend scored **0.87805**, much worse than XGBoost on its own (0.94265). Our local checks predicted the opposite. We reviewed the method to find out why.

#### Mistake 1: we tested the idea on the same periods where we found it

We discovered the look-alike pattern by studying steps **24, 29 and 31**. Every check that showed a large gain (22–35, 11–24, 18–31, 20–33) **contains at least one of those steps**. We called three of them "fresh checks", but they were not truly fresh. They reused the very periods that inspired the idea.

This is like writing exam questions after seeing the answers. The test only confirms what we already looked at.

When we re-ran the blend on periods that **do not contain** steps 23–33, it was worse every time:

| Validation steps | XGBoost alone | Blend | Change |
|---|---:|---:|---:|
| 5–18 | 0.9408 | 0.9184 | −0.0224 |
| 8–21 | 0.9939 | 0.9938 | −0.0001 |
| 11–22 | 0.9902 | 0.9895 | −0.0007 |
| 15–22 | 0.9922 | 0.9887 | −0.0035 |
| 34–35 | 0.9979 | 0.9942 | −0.0037 |

#### Mistake 2: we assumed "near-copies are illicit" would stay true in the future

In steps 22–35, the look-alike pattern mostly picked out illicit transactions. In the test period it does not. In several test steps, the look-alike model flags many transactions that XGBoost considers licit:

| Test step | Share flagged by look-alike model | Share flagged by XGBoost |
|---|---:|---:|
| 39 | 11.4% | 6.1% |
| 42 | 9.4% | 9.0% |
| **44** | **10.7%** | **0.5%** |

Step 44 alone contains about 170 near-identical transactions that XGBoost scores as licit. Batches of near-copies can also be **licit** automated activity, such as exchange or payment processing. Pushing these up the ranking places many licit transactions above real illicit ones, which lowers ROC-AUC.

#### Mistake 3: we were too confident

We estimated about 0.96 from only four previous Kaggle scores and recommended uploading before running a check that could have caught the problem.

#### What we change from now on

1. **Find an idea on some periods, test it on different periods.** If an idea comes from studying certain steps, every check used to judge it must exclude those steps.
2. **Compare with the test data before uploading** (without labels). If a new model ranks a large group of test transactions very differently from our best model, investigate before submitting.
3. **Keep XGBoost tuned (0.94265) as our best model** until a candidate passes both checks.

This experiment is still useful for the report: it shows how validation that reuses the same periods can give a misleadingly optimistic result.