def fit_with_chronological_stopping(factory, family, learning):
    # The stopping examples belong to the training period, before outer validation.
    training_end = int(learning["time_step"].max())
    fitting = learning.loc[learning["time_step"] <= training_end - 3]
    stopping = learning.loc[learning["time_step"] > training_end - 3]
    assert fitting["time_step"].max() < stopping["time_step"].min()
    assert fitting["label"].nunique() == stopping["label"].nunique() == 2

    initial = factory(NEW_MAXIMUM_ROUNDS, stopping=True)
    with threadpool_limits(limits=4):
        if family == "xgboost":
            initial.fit(
                fitting[FEATURE_COLUMNS], fitting["label"],
                eval_set=[(stopping[FEATURE_COLUMNS], stopping["label"])],
                verbose=False,
            )
            rounds = int(initial.best_iteration + 1)  # XGBoost starts counting at zero.
        else:
            initial.fit(
                fitting[FEATURE_COLUMNS], fitting["label"],
                eval_X=stopping[FEATURE_COLUMNS], eval_y=stopping["label"],
                callbacks=[early_stopping(NEW_EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            rounds = int(initial.best_iteration_)

        assert 1 <= rounds <= NEW_MAXIMUM_ROUNDS
        refitted = factory(rounds, stopping=False)
        refitted.fit(learning[FEATURE_COLUMNS], learning["label"])
    return refitted, rounds

def make_selected_xgboost(rounds, *, stopping=False):
    return XGBClassifier(
        n_estimators=int(rounds),
        learning_rate=0.05,
        tree_method="hist",
        grow_policy="lossguide",
        max_depth=0,             # Overall complexity is limited by the leaf count.
        max_leaves=15,
        max_bin=127,
        min_child_weight=3,
        reg_lambda=10,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=NEW_EARLY_STOPPING_ROUNDS if stopping else None,
        n_jobs=4,
        random_state=RANDOM_SEED,
    )

def select_time_review_features(learning, *, filter_time_features):
    training_end = int(learning["time_step"].max())
    fitting = learning.loc[learning["time_step"] <= training_end - 3]
    correlations = fitting[FEATURE_COLUMNS].corrwith(
        fitting["time_step"], method="spearman"
    )
    removed = (
        correlations.index[correlations.abs() > TIME_CORRELATION_LIMIT].tolist()
        if filter_time_features else []
    )
    columns = [column for column in FEATURE_COLUMNS if column not in removed]
    assert columns and "time_step" not in columns and "label" not in columns
    return columns, removed, correlations

def make_lowreg_xgboost(rounds=LOWREG_ROUNDS):
    return XGBClassifier(
        n_estimators=rounds, learning_rate=0.05, tree_method="hist",
        grow_policy="lossguide", max_depth=0, max_leaves=15, max_bin=127,
        min_child_weight=1,   # was 3
        reg_lambda=1,         # was 10
        objective="binary:logistic", n_jobs=4, random_state=RANDOM_SEED,
    )

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