"""Witness GIF: Δp bars grow top-k vs random-k (k=1,5 means + CI whiskers).
Saves figs/faithful.gif."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

df = pd.read_csv("figs/faithful_ci.csv")
stats = df.groupby(["k", "setup"]).delta.agg(["mean", "std"]).reset_index()
cats = [("top", 1), ("random", 1), ("top", 5), ("random", 5)]
vals = [tuple(stats[(stats.k == k) & (stats.setup == s)].iloc[0][["mean", "std"]])
        for s, k in cats]
labels = ["top-1", "rand-1", "top-5", "rand-5"]
COLS = ["#e74c3c", "#95a5a6", "#e74c3c", "#95a5a6"]

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.set_ylim(0, max(v[0] + v[1] for v in vals) * 1.25)
ax.set_ylabel("mean |Δp|")
ax.set_xticks(range(4), labels)


def draw(f):
    n = (f + 1) / 10
    ax.clear()
    ax.set_ylim(0, max(v[0] + v[1] for v in vals) * 1.25)
    ax.set_ylabel("mean |Δp|")
    ax.set_xticks(range(4), labels)
    ax.set_title(f"top-k removal vs random-k (frame {f+1}/10)")
    ax.bar(labels, [m * n for m, s in vals], yerr=[s * n for m, s in vals],
           color=COLS, capsize=5)
    return ax.patches


FuncAnimation(fig, draw, frames=10, interval=400).save(
    "figs/faithful.gif", writer="pillow", dpi=100)
print("saved figs/faithful.gif")
