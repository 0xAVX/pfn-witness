"""Self-contained helpers (no sibling repos, no absolute paths)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tabpfn_predict_proba(Xtr, ytr, Xte, chunk=10000, seed=0):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(random_state=seed)
    clf.fit(Xtr, ytr)
    return np.concatenate([clf.predict_proba(Xte[i:i + chunk])[:, 1]
                           for i in range(0, len(Xte), chunk)])


def openml_binary(name):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    d = fetch_openml(name=name, as_frame=True, parser="auto")
    X, y = d.data.copy(), d.target
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    num = [c for c in X.columns if c not in cat]
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat] = enc.fit_transform(X[cat].astype(str))
    X[num] = X[num].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(-1)
    y = LabelEncoder().fit_transform(pd.Series(y).astype(str).values)
    if y.mean() > 0.5:
        y = 1 - y
    return X.reset_index(drop=True), y
