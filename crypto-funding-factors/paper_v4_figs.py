#!/usr/bin/env python3
"""Figures for funding_paper v2.x from results/paper/v4_series.json (cumulative sums of weekly % returns).
  python paper_v4_figs.py  -> figures/fig_parts.{pdf,png}, fig_lv_spot.{pdf,png}"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def cum(x):
    return np.cumsum(np.array(x) * 100)


def days(d):
    return [datetime.strptime(x, "%Y-%m-%d") for x in d]


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("Cumulative return, % of book")


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, dy), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.plot([x[-1]], [y[-1]], "o", color=color, markersize=3.5)


def main():
    S = json.loads((ROOT / "results/paper/v5main/v4_series.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for ax, k, title in zip(axes, ("MOM", "LV", "FUND"), ("A. Momentum", "B. Low volatility", "C. FUND")):
        x = days(S[k]["day"])
        p, f = cum(S[k]["price"]), cum(S[k]["fund"])
        ax.plot(x, p, color=BLUE, linewidth=1.4, label="Price part")
        ax.plot(x, f, color=ORANGE, linewidth=1.4, label="Funding part")
        close = abs(p[-1] - f[-1]) < 0.08 * (max(p.max(), f.max()) - min(p.min(), f.min()))
        hi = 1 if p[-1] >= f[-1] else -1
        label_end(ax, x, p, "price", BLUE, 6 * hi if close else 0)
        label_end(ax, x, f, "funding", ORANGE, -6 * hi if close else 0)
        style(ax, title)
        ax.margins(x=0.12)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_parts.{ext}", dpi=200)
    plt.close(fig)

    L = S["LV_spot"]
    x = days(L["day"])
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    s, t = cum(L["spot"]), cum(L["perp_total"])
    ax.fill_between(x, t, s, color=ORANGE, alpha=0.12, linewidth=0)
    ax.plot(x, s, color=BLUE, linewidth=1.4, label="Spot prices")
    ax.plot(x, t, color=ORANGE, linewidth=1.4, label="Perpetual (funding, isolated margin)")
    label_end(ax, x, s, "spot", BLUE); label_end(ax, x, t, "perpetual", ORANGE)
    style(ax, "Low volatility (7-day), same picks, contracts with spot trading")
    ax.margins(x=0.1)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_lv_spot.{ext}", dpi=200)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
