# BT4012 Kaggle competition: detecting illicit Bitcoin transactions

Individual Kaggle competition project for NUS student **e1156285**.

**Final model:** XGBoost (300 trees) and LightGBM (150 trees) averaged 50/50, both with light regularisation, trained on all labelled transactions with recency weighting (half-life of 4 time steps). XGBoost also receives nine payment-network counts built from the edge list, and is trained with DART boosting (it randomly switches off some earlier trees while learning).
**Public ROC-AUC:** 0.95527 (first submission 0.93654; course baseline about 0.92).
**Final selections on Kaggle:** this model (0.95527) and the same model with ordinary XGBoost instead of DART (0.95524).

## What is in this repository

| Path | What it is |
|---|---|
| `e1156285.ipynb` | Final notebook: data loading and checks, feature preparation (including the network features), validation (15 practice exams), final training and CSV generation. Settings in Section 1 choose the version: `FINAL_XGB_DART = True` with `XGB_EXTRA = "network"` is the final model |
| `e1156285.pdf` | Report (at most 4 pages) |
| `requirements.txt` | Library versions used |
| `submissions/` | CSV files written by the notebook: `e1156285_submission.csv` (identical to the uploaded 0.95527 file) and the second final selection `e1156285_submission_second.csv` (identical to the uploaded 0.95524 file). Not stored in Git |
| `archive/` | The full record of what was tried: `e1156285_development_log.ipynb` (step-by-step journal), `notebooks/` (individual experiments), `results/` (their scripts, metrics and charts; CSV outputs, raw-data copies and model weights stay local), `requirements-dev.txt` (libraries those notebooks need). These are kept for reference; their file paths were written for the old folder layout, so they are not meant to be re-run from here |

## How to run

Use Python 3.11 or newer. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

On the configured Mac, double-clicking `Start Jupyter.command` opens JupyterLab instead.

Open `e1156285.ipynb` and choose *Kernel → Restart Kernel and Run All Cells* (about 15–20 minutes on a laptop, most of it the practice exams and the DART model). The notebook fixes the random seed and uses 2 CPU threads, so it reproduces the uploaded Kaggle file (to rounding).

## Data

Download the competition data from Kaggle and place it here (the data is not stored in Git):

- `Data Set/Train Data.csv`
- `Data Set/Test Data.csv`
- `Data Set/TXS Edgelist.csv`
- `Data Set/Sample Submission/BT4012 Competition Submission.csv`

Training covers time steps 1–35 (31,235 labelled transactions, 110,537 unknown); the test set covers steps 36–49 (15,329 transactions). Unknown labels are missing, not zero. Kaggle submissions contain `index,target`.

## Acknowledgement of AI assistance

I used two AI assistants. OpenAI Codex (via ChatGPT) set up the repository and notebook and wrote and ran the baselines and early experiments. Anthropic's Claude proposed and coded the practice-exam evaluation and the safety check, wrote and ran most later experiments, built the final notebook and drafted the report. I set the goals, chose which candidates to upload, compared the two assistants' results and chose the final submissions. All experiments used the competition data only.
