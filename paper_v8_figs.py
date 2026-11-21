#!/usr/bin/env python3
"""Figure 4 of the revision draft: Binance vs Bybit and Binance vs OKX short-leg excess funding of the 7-day
low-volatility strategy by year, each on its own covered legs, parsed from results/paper/v8.txt and v8_okx.txt
(paper_v8 A4 and A4b, each run once). Two panels because the covered legs differ by exchange."""
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

def parse(fname, key):
    line = next(l for l in (ROOT / fname).read_text().splitlines() if key in l)
    rows = [(int(y), float(a), float(b), int(n)) for y, a, b, n in
            re.findall(r"(\d{4}) ([+-][\d.]+)% / ([+-][\d.]+)% \((\d+) wks\)", line)]
    return [r for r in rows if r[3] >= 10]          # years with fewer than 10 covered weeks are omitted
panels = [("A. Binance and Bybit, legs covered by Bybit", "Bybit", parse("results/paper/v8.txt", "by year (Bybit / Binance")),
          ("B. Binance and OKX, legs covered by OKX", "OKX", parse("results/paper/v8_okx.txt", "by year (OKX / Binance"))]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
for ax, (title, other, rows) in zip(axes, panels):
    yrs = [r[0] for r in rows]; ot = np.array([r[1] for r in rows]); bn = np.array([r[2] for r in rows])
    xx = np.arange(len(yrs)); wd = 0.36
    ax.bar(xx - wd / 2 - 0.01, bn, wd, color=C1, label="Binance", zorder=3)
    ax.bar(xx + wd / 2 + 0.01, ot, wd, color=C4, label=other, zorder=3)
    for i in range(len(yrs)):
        for xv, yv in ((xx[i] - wd / 2 - 0.01, bn[i]), (xx[i] + wd / 2 + 0.01, ot[i])):
            ax.text(xv, yv + (0.03 if yv >= 0 else -0.03), f"{yv:+.2f}".replace("-", "\u2212"), ha="center",
                    va="bottom" if yv >= 0 else "top", fontsize=6, color=INK2)
    ax.set_xticks(xx, [f"{y}\n({r[3]} wks)" for y, r in zip(yrs, rows)])
    ax.axhline(0, color=INK2, linewidth=0.6); ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_title(title, loc="left", fontsize=9, color=INK); ax.legend(frameon=False, loc="lower left", fontsize=7.5)
axes[0].set_ylabel("% of book per week", color=INK)
lo = min(min(r[1], r[2]) for _, _, rows in panels for r in rows); hi = max(max(r[1], r[2]) for _, _, rows in panels for r in rows)
axes[0].set_ylim(lo - 0.25, hi + 0.2)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(FIG / f"fig_bybit.{e}", dpi=200)
print("figure written")
