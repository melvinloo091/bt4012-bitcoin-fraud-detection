# 24 October 2026: untested model types

All variants start from the 0.95524 model (XGBoost 300 trees with the nine network counts + LightGBM 150 trees, recency half-life 4).

| Variant | 15 practice exams (vs 0.9505) | Kaggle public |
|---|---|---|
| XGBoost with DART boosting (final model) | see `e1156285.ipynb`, Section 6.1 | 0.95527 |
| LightGBM on own features 1-93 only | 0.9470 (better in 3) | 0.94867 |
| XGBoost on features ranked within each time step | 0.9436 (better in 0) | 0.94574 |
| Within-step ranks in both models | 0.9038 | not uploaded |
| LightGBM with DART | much worse (0.80-0.86 on the two hardest exams) | not uploaded |

`scripts/` holds the cloud scripts that produced these numbers (paths point to the cloud workspace, so they are a record, not meant to be re-run here). The final notebook rebuilds the DART submission from the raw files.
