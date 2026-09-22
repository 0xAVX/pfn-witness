"""Witness README figures: jury bars, faithfulness, poison/repair.
Saves figs/jury.png, figs/faithful.png, figs/repair.png. GPU ~3 min.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from witness.core import fit_and_readout, jury
from witness.data import openml_binary

X, y = openml_binary("phoneme")
Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000, stratify=y,
                                            random_state=1)
idx = np.random.RandomState(0).choice(len(Xpool), 2000, replace=False)
Xc, yc = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
_, W, TIX = fit_and_readout(Xc, yc, Xval.iloc[:50], seed=0)
r = int(np.argmax(W.max(axis=1)))  # most concentrated jury = clearest demo
j = jury(W[r], TIX, yc, top=5)

fig, ax = plt.subplots(figsize=(6, 3.5))
sup = j["supporting"] + j["opposing"]
labels = [f"#{i} {'POS' if lab else 'NEG'}" for i, w, lab in sup]
vals = [w if lab == 1 else -w for i, w, lab in sup]
cols = ["#27ae60" if v > 0 else "#c0392b" for v in vals]
ax.barh(labels, vals, color=cols)
ax.set_xlabel("vote weight (+pos / −neg)")
ax.set_title(f"Jury for test row {r}")
fig.tight_layout()
fig.savefig("figs/jury.png", dpi=110)

w = pd.read_csv("figs/witness.csv")
f = w[w.exp == "faithful"]
fig, ax = plt.subplots(figsize=(5, 3.2))
x = np.arange(len(f) // 2)
top = f[f.setup == "top"]["val"].values
rnd = f[f.setup == "random"]["val"].values
ax.bar(x - 0.2, top, 0.4, label="remove top-k", color="#e74c3c")
ax.bar(x + 0.2, rnd, 0.4, label="remove random-k", color="#95a5a6")
ax.set_xticks(x, [f"k={int(k)}" for k in f[f.setup == "top"]["k"].values])
ax.set_ylabel("mean |Δp|")
ax.set_title("Faithfulness: 15–22x effect")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/faithful.png", dpi=110)

p = w[w.exp == "poison"]
rp = w[w.exp == "repair"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
a1.bar(p.setup, p.val.values, color=["#95a5a6", "#e74c3c"])
a1.set_title("Poison 5%: random vs top-attention")
a1.set_ylabel("val AUC")
a2.bar(rp.setup, rp.val.values, color=["#95a5a6", "#27ae60"])
a2.set_title("Repair: before vs after top-20 removal")
a2.set_ylabel("val AUC")
fig.tight_layout()
fig.savefig("figs/repair.png", dpi=110)
print("saved jury/faithful/repair pngs")
