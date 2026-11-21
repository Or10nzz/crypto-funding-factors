#!/usr/bin/env python3
"""Figure 3 of the v3.x revision draft: paper_v7_figs specification curves plus the ex-ante funding filter (paper_v8 F1:
+0.004%, t 0.22) in panel A. Writes fig_spec_v3; the submitted v2.5 keeps fig_spec."""
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#1baf7a", "#eda100", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

panels = {
 "A. LV 7-day: spot-sample funding component": [
  ("main", -0.244, -4.10), ("floored, last close", -0.266, -4.13), ("without ALPACA week", -0.244, -4.10), ("without 2026", -0.173, -3.35),
  ("w/o top 1% by return", -0.225, -4.10), ("w/o top 5% by return", -0.169, -4.23), ("w/o top 1% by funding", -0.080, -4.52),
  ("w/o top 5% by funding", -0.021, -3.23), ("threshold 20, 317 weeks", -0.274, -4.30), ("NW 26 lags", -0.244, -2.64),
  ("start Tue", -0.232, -4.12), ("start Wed", -0.220, -3.86), ("start Thu", -0.231, -3.72), ("start Fri", -0.240, -3.69),
  ("start Sat", -0.270, -4.11), ("start Sun", -0.246, -3.84),
  ("ex-ante funding filter", 0.004, 0.22)],
 "B. Momentum: funding part": [
  ("main", 0.487, 4.66), ("cross-margined", 0.463, 4.59), ("floored, last close", 0.462, 4.58), ("without ALPACA week", 0.494, 4.71),
  ("without 2026", 0.276, 3.57), ("w/o top 1% by return", 0.408, 4.57), ("w/o top 5% by return", 0.293, 4.30),
  ("w/o top 1% by funding", 0.106, 4.80), ("w/o top 5% by funding", 0.030, 3.27), ("winsorised", 0.463, 4.59), ("NW 26 lags", 0.487, 2.57),
  ("10/10", 0.310, 4.54), ("top 50, 6/6", 0.559, 4.62), ("top 50, 10/10", 0.395, 4.69), ("top 100, 10/10", 0.442, 5.06),
  ("volume-weighted", 0.491, 3.82), ("1h delay", 0.465, 4.62), ("first half", 0.115, 2.55), ("second half", 0.833, 4.78),
  ("90-day universe", 0.219, 3.47)],
 "C. Momentum: net return": [
  ("main", 0.623, 2.33), ("cross-margined", 0.742, 2.64), ("floored, last close", 0.853, 2.88), ("without ALPACA week", 0.585, 2.15),
  ("without 2026", 0.523, 2.11), ("w/o top 1% by return", -0.086, -0.39), ("w/o top 5% by return", -0.750, -3.83),
  ("w/o top 1% by funding", 0.206, 0.81), ("winsorised", 0.428, 1.89), ("NW 26 lags", 0.623, 2.51), ("10/10", 0.324, 1.79),
  ("top 50, 6/6", 0.590, 1.80), ("top 50, 10/10", 0.464, 1.94), ("top 100, 10/10", 0.612, 2.06), ("volume-weighted", 0.989, 3.02),
  ("1h delay", 0.742, 2.67), ("first half", 0.739, 2.18), ("second half", 0.515, 1.27), ("90-day universe", 0.837, 3.32),
  ("tiered costs", 0.484, 1.81), ("margin 2x", 0.543, 1.98), ("margin 3x", 0.599, 2.15), ("maintenance 5%", 0.597, 2.23)]}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, (title, rows) in zip(axes, panels.items()):
    rows = sorted(rows, key=lambda r: r[1])
    est = np.array([r[1] for r in rows]); se = np.abs(est / np.array([r[2] for r in rows]))
    xs = np.arange(len(rows))
    sig = np.abs([r[2] for r in rows]) >= 1.96
    ax.errorbar(xs, est, yerr=1.645 * se, fmt="none", ecolor=GRID, elinewidth=2.2)
    ax.scatter(xs[sig], est[sig], s=14, color=C1, zorder=3, label="|t| ≥ 1.96")
    ax.scatter(xs[~sig], est[~sig], s=14, facecolor="white", edgecolor=C1, zorder=3, label="|t| < 1.96")
    mi = [r[0] for r in rows].index("main"); ax.scatter([mi], [est[mi]], s=40, facecolor="none", edgecolor=C4, zorder=4, label="main")
    ax.axhline(0, color=INK2, linewidth=0.6); ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title(title, loc="left", fontsize=9, color=INK); ax.set_xticks([]); ax.set_xlabel("specifications, sorted by estimate", color=INK2)
axes[0].set_ylabel("% per week (bars: 90% interval)", color=INK)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(FIG / f"fig_spec_v3.{e}", dpi=200)
print("figures written")
