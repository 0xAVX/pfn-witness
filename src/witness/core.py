"""PFN Witness core: decoder retrieval weights -> jury, audit, faithfulness.

TabPFN-3.5's decoder is an attention retrieval head: prediction for a test
row is a vote over training rows. get_decoder_readout() exposes the votes.
We turn them into: per-row jury (supporting/opposing witnesses), dataset
audit H (rows whose votes systematically oppose truth), and removal-based
faithfulness (top-k vs random-k excision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from tabpfn_extensions.interpretability import get_decoder_readout


def fit_and_readout(Xtr, ytr, Xte, seed=0):
    clf = TabPFNClassifier(random_state=seed)
    clf.fit(Xtr, ytr)
    W, idx = get_decoder_readout(clf, Xte)
    return clf, np.asarray(W), np.asarray(idx)


def jury(W_row, idx, ytr, p_pred=None, top=3):
    """Supporting = highest-attention rows labeled as the predicted class;
    opposing = highest-attention rows labeled otherwise. Needs p_pred
    (predicted P(class1)) to know which class is 'supported'."""
    order = np.argsort(-W_row)
    labs = ytr[idx[order]]
    if p_pred is None:
        p_pred = float(labs[:top].mean())  # fallback, not for display
    pred = int(p_pred > 0.5)
    sup_i = [i for i in order if ytr[idx[i]] == pred][:top]
    opp_i = [i for i in order if ytr[idx[i]] != pred][:top]
    sup = [(int(idx[i]), float(W_row[i]), int(ytr[idx[i]])) for i in sup_i]
    opp = [(int(idx[i]), float(W_row[i]), int(ytr[idx[i]])) for i in opp_i]
    return {"supporting": sup, "opposing": opp, "pred": pred}


def audit(W, idx, ytr, yva_pred, yva_true):
    """H_i = attention received while voting against truth, summed over val rows.
    pred_pos: predicted P(class1) per val row from the same model."""
    n_tr = len(ytr)
    H = np.zeros(n_tr)
    wrong = (yva_pred > 0.5).astype(int) != yva_true
    for r in np.where(wrong)[0]:
        voted_pos = W[r] * ytr[idx]
        voted_neg = W[r] * (1 - ytr[idx])
        truth_pos = yva_true[r] == 1
        harm = voted_neg if truth_pos else voted_pos
        for k, j in enumerate(idx):
            H[j] += harm[k]
    attn = W.sum(axis=0)
    out = pd.DataFrame({"row": idx, "attention": attn,
                        "harmful": np.array([H[j] for j in idx])})
    return out.sort_values("harmful", ascending=False).reset_index(drop=True)


def removal_effect(Xtr, ytr, Xt_row, k_list, seed=0, top=True, n_rand=1):
    """Refit without top-k (or random-k) witnesses for one test row.
    Returns {k: [delta_p...]}. Positive delta = prediction moved."""
    from tabpfn_extensions.interpretability import get_decoder_readout as gdr
    base = TabPFNClassifier(random_state=seed)
    base.fit(Xtr, ytr)
    p0 = float(base.predict_proba(Xt_row)[:, 1][0])
    W, idx = gdr(base, Xt_row)
    order = np.argsort(-W[0])
    out = {}
    rng = np.random.RandomState(0)
    for k in k_list:
        ds = []
        reps = 1 if top else n_rand
        for _ in range(reps):
            drop = order[:k] if top else rng.choice(len(Xtr), k, replace=False)
            keep = np.ones(len(Xtr), bool)
            keep[idx[drop]] = False
            m = TabPFNClassifier(random_state=seed)
            m.fit(Xtr.iloc[keep], ytr[keep])
            ds.append(abs(float(m.predict_proba(Xt_row)[:, 1][0]) - p0))
        out[k] = ds
    return p0, out
