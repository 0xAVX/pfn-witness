"""Witness demo: jury for probe rows + audit table. Fits once at startup
(~1 min), then instant. Run: <venv-python> demo/app.py (port 5003)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split

from witness.core import audit, fit_and_readout, jury
from witness.data import openml_binary, tabpfn_predict_proba

print("loading + fitting witness model...", flush=True)
X, y = openml_binary("phoneme")
Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000, stratify=y,
                                            random_state=1)
idx = np.random.RandomState(0).choice(len(Xpool), 2000, replace=False)
Xc, yc = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
Xv = Xval.iloc[:500].reset_index(drop=True)
yv = yval[:500]
_, W, TIX = fit_and_readout(Xc, yc, Xv, seed=0)
P = tabpfn_predict_proba(Xc, yc, Xv, seed=0)
AUD = audit(W, TIX, yc, P, yv)
print("ready.", flush=True)

app = Flask(__name__)
PAGE = """
<h1>PFN Witness — who convinced TabPFN? (phoneme)</h1>
<form method=get>Row (0-499): <input name=i value="{{i}}" size=5>
<input type=submit value="Show jury"></form>
<h2>P(fraud-ish) = {{'%.2f' % p}} (true {{t}})</h2>
<h3>Supporting</h3><ul>{% for r, w, l in sup %}<li>#{{r}} vote {{'%.1f' % (100*w)}}% label {{l}}</li>{% endfor %}</ul>
<h3>Opposing</h3><ul>{% for r, w, l in opp %}<li>#{{r}} vote {{'%.1f' % (100*w)}}% label {{l}}</li>{% endfor %}</ul>
<h3>Audit: top-10 harmful rows</h3>
<table border=1 cellpadding=3><tr><th>row</th><th>attention</th><th>harmful</th></tr>
{% for r in aud %}<tr><td>{{r[0]}}</td><td>{{'%.2f' % r[1]}}</td><td>{{'%.2f' % r[2]}}</td></tr>{% endfor %}</table>
"""


@app.get("/")
def index():
    i = int(request.args.get("i", 0)) % len(Xv)
    j = jury(W[i], TIX, yc, p_pred=float(P[i]), top=3)
    top10 = [(int(r.row), float(r.attention), float(r.harmful))
             for r in AUD.itertuples(index=False)]
    return render_template_string(PAGE, i=i, p=float(P[i]), t=int(yv[i]),
                                  sup=j["supporting"], opp=j["opposing"],
                                  aud=top10)


if __name__ == "__main__":
    app.run(debug=True, port=5003)
