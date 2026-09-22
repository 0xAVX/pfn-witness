"""Repair control: top-20 H removal vs 3x random-20. Saves figs/repair_ctrl.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from witness.core import audit, fit_and_readout
from witness.data import openml_binary, tabpfn_predict_proba
SEED = 0
X, y = openml_binary("phoneme")
Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000, stratify=y, random_state=1)
idx = np.random.RandomState(SEED).choice(len(Xpool), 2000, replace=False)
Xc, yc = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
rng = np.random.RandomState(5)
flip = rng.choice(len(Xc), int(0.05 * len(Xc)), replace=False)
yp = yc.copy(); yp[flip] = 1 - yp[flip]
_, W, ix = fit_and_readout(Xc, yp, Xval.iloc[:500], seed=SEED)
aud = audit(W, ix, yp, tabpfn_predict_proba(Xc, yp, Xval.iloc[:500], seed=SEED), yval[:500])
rows = []
keep = np.ones(len(Xc), bool); keep[aud["row"].values[:20]] = False
a = roc_auc_score(yval, tabpfn_predict_proba(Xc.iloc[keep], yp[keep], Xval, seed=SEED))
rows.append(("top20-H", a)); print(f"top20-H: {a:.4f}", flush=True)
for s in range(3):
    drop = np.random.RandomState(100 + s).choice(len(Xc), 20, replace=False)
    keep = np.ones(len(Xc), bool); keep[drop] = False
    a = roc_auc_score(yval, tabpfn_predict_proba(Xc.iloc[keep], yp[keep], Xval, seed=SEED))
    rows.append((f"random20-s{s}", a)); print(f"random20-s{s}: {a:.4f}", flush=True)
pd.DataFrame(rows, columns=["setup", "auc"]).to_csv("figs/repair_ctrl.csv", index=False)
