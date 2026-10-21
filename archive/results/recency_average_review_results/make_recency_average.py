"""Build claude_experiments/submissions/submission_lowreg_pair_recent345_avg.csv:
the equal average of the half-life 3, 4 and 5 recency models (run make_recency_sweep.py 3 4 5 first).
Averaging neighbouring recency settings makes the result less dependent on the exact half-life."""
from pathlib import Path
import pandas as pd
S = Path.cwd() / "claude_experiments" / "submissions"
parts = [pd.read_csv(S / f"submission_lowreg_pair_recent{h}.csv") for h in (3, 4, 5)]
assert all(p["index"].equals(parts[0]["index"]) for p in parts)
out = parts[0][["index"]].copy()
out["target"] = sum(p["target"] for p in parts) / 3
out.to_csv(S / "submission_lowreg_pair_recent345_avg.csv", index=False)
print("Saved", S / "submission_lowreg_pair_recent345_avg.csv")
