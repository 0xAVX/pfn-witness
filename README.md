# PFN Witness — who convinced TabPFN?

TabPFN-3.5's decoder is an attention retrieval head: each prediction is a
vote over training rows. Witness exposes the votes (via
`tabpfn-extensions`' decoder readout, Aug 2026) and turns them into:

- **Jury**: supporting/opposing training rows per prediction, with values.
- **Audit**: H score = attention received while voting against truth.
- **Faithfulness**: removing top-k witnesses moves predictions more than
  random removal (measured, not assumed).
## Evidence (`figs/witness.csv`, phoneme)

- **Faithfulness**: removing top-1 witness moves P by 0.107 vs 0.007 random
  (15×); top-5: 0.153 vs 0.007 (22×). Votes are causal, not decorative.

![jury](figs/jury.png)

![faithfulness](figs/faithful.png)
- **Poison**: corrupting top-attention 5% → AUC 0.9285 vs random-5% 0.9522.
- **Repair**: audit-H top-20 removal → 0.9511 → 0.9536 on 5%-flipped labels.
  Modest, honest, actionable.

![poison and repair](figs/repair.png)

Not claimed: first data valuation for TabPFN (LOO/Shapley work exists).
Claimed: prediction-local provenance from decoder weights, aggregated into
a label-audit and repair loop.

## Reproduce

```bash
<venv-python> experiments/run.py   # figs/witness.csv
<venv-python> demo/app.py          # jury view (port 5003)
```
