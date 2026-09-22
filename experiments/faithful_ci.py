"""Faithfulness stats: 30 probes, top/random-k removal, per-probe deltas +
bootstrap CI. Saves figs/faithful_ci.csv. GPU ~15 min.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from witness.core import removal_effect
from witness.data import openml_binary

SEED = 0


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y = openml_binary("phoneme")
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), 2000, replace=False)
    Xc, yc = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
    probes = np.random.RandomState(3).choice(len(Xval), 30, replace=False)
    rows = []
    for r in probes:
        for k in [1, 5]:
            _, d = removal_effect(Xc, yc, Xval.iloc[[r]], [k], seed=SEED,
                                  top=True)
            rows.append((int(r), k, "top", d[k][0]))
            _, d = removal_effect(Xc, yc, Xval.iloc[[r]], [k], seed=SEED,
                                  top=False, n_rand=2)
            for v in d[k]:
                rows.append((int(r), k, "random", v))
        print(f"probe {r} done", flush=True)
    df = pd.DataFrame(rows, columns=["probe", "k", "setup", "delta"])
    df.to_csv("figs/faithful_ci.csv", index=False)
    rng = np.random.RandomState(0)
    for k in [1, 5]:
        for s in ["top", "random"]:
            v = df[(df.k == k) & (df.setup == s)].groupby("probe").delta.mean().values
            bs = [rng.choice(v, len(v), replace=True).mean() for _ in range(2000)]
            print(f"k={k} {s}: mean={v.mean():.4f} "
                  f"95%CI=[{np.percentile(bs, 2.5):.4f},{np.percentile(bs, 97.5):.4f}]",
                  flush=True)
    print(f"done {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
