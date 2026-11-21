#!/usr/bin/env python3
"""Figure 4 of the v3.0 revision draft: Bybit vs Binance short-leg excess funding of the 7-day low-volatility
strategy by year (covered legs only), parsed from results/paper/v8.txt (paper_v8, run once)."""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
C1, C4 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

txt = (ROOT / "results/paper/v8.txt").read_text()
line = next(l for l in txt.splitlines() if "by year (Bybit / Binance" in l)
rows = [(int(y), float(a), float(b), int(n)) for y, a, b, n in
        re.findall(r"(\d{4}) ([+-][\d.]+)% / ([+-][\d.]+)% \((\d+) wks\)", line)]
rows = [r for r in rows if r[3] >= 10]                 # 2020 has 2 covered weeks: omitted, stated in the caption
yrs = [r[0] for r in rows]; by = np.array([r[1] for r in rows]); bn = np.array([r[2] for r in rows])

fig, ax = plt.subplots(figsize=(6.6, 3.2))
xx = np.arange(len(yrs)); wd = 0.36
ax.bar(xx - wd / 2 - 0.01, bn, wd, color=C1, label="Binance", zorder=3)
ax.bar(xx + wd / 2 + 0.01, by, wd, color=C4, label="Bybit", zorder=3)
for i in range(len(yrs)):
    for xv, yv in ((xx[i] - wd / 2 - 0.01, bn[i]), (xx[i] + wd / 2 + 0.01, by[i])):
        ax.text(xv, yv + (0.03 if yv >= 0 else -0.03), f"{yv:+.2f}".replace("-", "\u2212"), ha="center", va="bottom" if yv >= 0 else "top",
                fontsize=6.5, color=INK2)
ax.set_xticks(xx, [f"{y}\n({r[3]} wks)" for y, r in zip(yrs, rows)])
ax.axhline(0, color=INK2, linewidth=0.6)
ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
ax.set_ylabel("% of book per week", color=INK)
ax.set_title("Low volatility (7-day), short-leg excess funding on the same legs", loc="left", fontsize=9.5, color=INK)
ax.legend(frameon=False, loc="lower left", fontsize=7.5)
ax.set_ylim(min(by.min(), bn.min()) - 0.25, max(by.max(), bn.max()) + 0.2)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(FIG / f"fig_bybit.{e}", dpi=200)
print("figure written:", yrs)
