"""Witness experiments (phoneme): faithfulness removal, poison targeting,
repair loop. Saves figs/witness.csv. GPU ~30 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from witness.core import audit, fit_and_readout, removal_effect
from witness.data import openml_binary, tabpfn_predict_proba

SEED = 0


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y = openml_binary("phoneme")
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), 2000, replace=False)
    Xc, yc = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
    rows = []

    print("== faithfulness: top-k vs random-k removal ==", flush=True)
    probes = np.random.RandomState(3).choice(len(Xval), 12, replace=False)
    for k in [1, 5]:
        dt, dr = [], []
        for r in probes:
            _, d = removal_effect(Xc, yc, Xval.iloc[[r]], [k], seed=SEED, top=True)
            dt.append(d[k][0])
            _, d = removal_effect(Xc, yc, Xval.iloc[[r]], [k], seed=SEED,
                                  top=False, n_rand=2)
            dr += d[k]
        rows.append(("faithful", k, "top", float(np.mean(dt))))
        rows.append(("faithful", k, "random", float(np.mean(dr))))
        print(f"  k={k}: top Δp={np.mean(dt):.4f} vs random Δp={np.mean(dr):.4f}",
              flush=True)

    print("== poison: random 5% vs top-attention 5% ==", flush=True)
    clf, W, tix = fit_and_readout(Xc, yc, Xval.iloc[:500], seed=SEED)
    attn_recv = pd.Series(W.sum(axis=0), index=tix).groupby(level=0).sum()
    attn_recv = attn_recv.reindex(range(len(Xc))).fillna(0).values
    rng = np.random.RandomState(5)
    for name, flip in [("random", rng.choice(len(Xc), int(0.05 * len(Xc)),
                                             replace=False)),
                       ("witness", np.argsort(-attn_recv)[:int(0.05 * len(Xc))])]:
        yp = yc.copy()
        yp[flip] = 1 - yp[flip]
        p = tabpfn_predict_proba(Xc, yp, Xval, seed=SEED)
        a = roc_auc_score(yval, p)
        rows.append(("poison", 0.05, name, a))
        print(f"  {name}: AUC={a:.4f}", flush=True)

    print("== repair: audit H -> remove top-20 ==", flush=True)
    flip = rng.choice(len(Xc), int(0.05 * len(Xc)), replace=False)
    yp = yc.copy()
    yp[flip] = 1 - yp[flip]
    p0 = tabpfn_predict_proba(Xc, yp, Xval, seed=SEED)
    a0 = roc_auc_score(yval, p0)
    _, W2, ix2 = fit_and_readout(Xc, yp, Xval.iloc[:500], seed=SEED)
    aud = audit(W2, ix2, yp, tabpfn_predict_proba(Xc, yp, Xval.iloc[:500],
                                                  seed=SEED), yval[:500])
    drop = aud["row"].values[:20]
    keep = np.ones(len(Xc), bool)
    keep[drop] = False
    a1 = roc_auc_score(yval, tabpfn_predict_proba(Xc.iloc[keep], yp[keep], Xval,
                                                  seed=SEED))
    rows.append(("repair", 20, "before", a0))
    rows.append(("repair", 20, "after", a1))
    print(f"  before {a0:.4f} -> after {a1:.4f}", flush=True)

    pd.DataFrame(rows, columns=["exp", "k", "setup", "val"]).to_csv(
        "figs/witness.csv", index=False)
    print(f"saved figs/witness.csv ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
