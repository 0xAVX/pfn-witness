import numpy as np
import pandas as pd

from witness.core import audit, fit_and_readout, jury


def test_readout_sums_to_one():
    rng = np.random.RandomState(0)
    Xtr = pd.DataFrame({"a": rng.randn(300), "b": rng.randn(300)})
    ytr = ((Xtr.a + Xtr.b) > 0).astype(int).values
    Xte = pd.DataFrame({"a": [0.5, -1.0], "b": [0.2, 0.3]})
    _, W, idx = fit_and_readout(Xtr, ytr, Xte, seed=0)
    assert W.shape[1] == len(Xtr)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-4)


def test_jury_and_audit():
    rng = np.random.RandomState(1)
    Xtr = pd.DataFrame({"a": rng.randn(200), "b": rng.randn(200)})
    ytr = ((Xtr.a - Xtr.b) > 0).astype(int).values
    Xte = pd.DataFrame({"a": [1.0], "b": [-1.0]})
    _, W, idx = fit_and_readout(Xtr, ytr, Xte, seed=0)
    j = jury(W[0], idx, ytr, top=2)
    assert len(j["supporting"]) == 2 and len(j["opposing"]) == 2
    aud = audit(W, idx, ytr, np.array([0.9]), np.array([1]))
    assert len(aud) == len(Xtr) and (aud["harmful"] >= 0).all()
