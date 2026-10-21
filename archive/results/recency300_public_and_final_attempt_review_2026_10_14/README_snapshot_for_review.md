# BT4012 Kaggle competition: detecting illicit Bitcoin transactions

Individual Kaggle competition project for NUS student **e1156285**.

**Final model:** XGBoost (300 trees) and LightGBM (150 trees) averaged 50/50, both with light regularisation, trained on all labelled transactions with recency weighting (half-life of 4 time steps).
**Public ROC-AUC:** 0.95484 (first submission 0.93654; course baseline about 0.92).

## What is in this repository

| Path | What it is |
|---|---|
| `e1156285.ipynb` | Final notebook: data loading and checks, feature preparation, validation (15 practice exams), final training and CSV generation |
| `e1156285.pdf` | Report (at most 4 pages) |
| `requirements.txt` | Library versions used |
| `submissions/` | CSV files written by the notebook (`submission_half4_long_xgb_300_150_experimental.csv`, the uploaded 0.95484 file, and the optional `submission_lowreg_pair_recent345_avg_xgb300.csv`) |
| `archive/` | The full record of what was tried: `e1156285_development_log.ipynb` (step-by-step journal), `notebooks/` (individual experiments), `results/` (their outputs and earlier submission files), `requirements-dev.txt` (libraries those notebooks need). These are kept for reference; their file paths were written for the old folder layout, so they are not meant to be re-run from here |

## How to run

Use Python 3.11 or newer. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

On the configured Mac, double-clicking `Start Jupyter.command` opens JupyterLab instead.

Open `e1156285.ipynb` and choose *Kernel → Restart Kernel and Run All Cells* (a few minutes on a laptop). The notebook fixes the random seed and uses 2 CPU threads, so it reproduces the uploaded Kaggle file (to rounding).

## Data

Download the competition data from Kaggle and place it here (the data is not stored in Git):

- `Data Set/Train Data.csv`
- `Data Set/Test Data.csv`
- `Data Set/TXS Edgelist.csv`
- `Data Set/Sample Submission/BT4012 Competition Submission.csv`

Training covers time steps 1–35 (31,235 labelled transactions, 110,537 unknown); the test set covers steps 36–49 (15,329 transactions). Unknown labels are missing, not zero. Kaggle submissions contain `index,target`.

## Acknowledgement of AI assistance

Two AI assistants helped with this project: OpenAI Codex (via ChatGPT) with repository and notebook setup, baselines and early experiments, and Anthropic's Claude with the practice-exam evaluation, the final model's experiments, the final notebook and drafting the report. All experiments used the competition data only. The student reviewed every result and can explain all submitted work.
