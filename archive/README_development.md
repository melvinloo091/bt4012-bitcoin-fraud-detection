# BT4012 Bitcoin Transaction Fraud Detection

Individual Kaggle competition project for student E1156285.

The starter notebook `e1156285.ipynb` loads and checks the supplied files. Modeling, validation, and final submission generation will be added as the project progresses. No model results are claimed yet.

## Run locally

Use Python 3.11. From this project folder:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

On the configured Mac, double-click `Start Jupyter.command` to open JupyterLab, then open `e1156285.ipynb`.

Run notebook cells from top to bottom with Shift + Enter. Before submission, restart the kernel and run all cells to verify reproducibility.

## Data

Download the competition data from Kaggle and place it at these paths relative to the project folder:

- `Data Set/Train Data.csv`
- `Data Set/Test Data.csv`
- `Data Set/TXS Edgelist.csv`
- `Data Set/Sample Submission/BT4012 Competition Submission.csv`

The original data is excluded from Git. Unknown labels are missing, not zero. Training periods are 1–35; test periods are 36–49. Use chronological validation and avoid leaking held-out labels into graph features.

Kaggle submission files must contain `index,target`. The notebook filename follows the student-number convention. The final report must link to this repository; ensure the teaching team can access it by the deadline.

## Acknowledgement

OpenAI Codex assisted with repository setup and the starter notebook. Update this statement as the work develops. The student must understand and justify all submitted work.
