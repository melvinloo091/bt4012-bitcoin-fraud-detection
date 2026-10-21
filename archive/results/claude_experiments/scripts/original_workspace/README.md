# Original experiment scripts (Claude's cloud workspace, 11–12 Oct 2026)

Kept for the record. They use workspace-specific paths (/home/claude/work/...) and cached predictions,
so they will not run unchanged on your Mac. Use ../make_recent7_and_local_features.py to regenerate the CSVs.

- base.py        – data loading, 15 rolling exams + 11 large-training checks, prediction cache (get_preds)
- preupload.py   – label-free safety check against XGBoost tuned
- report2.py     – local scores + safety check table
- run_var.py     – 'pair_recent7' (recency half-life 7) and other locally-equal variants
- run_probe.py   – 'pair_local' (features 1–93 only) and 'pair_recent14'
- run_gpt.py     – independent reproduction of ChatGPT's LightGBM leaf 5 / leaf 50 models
