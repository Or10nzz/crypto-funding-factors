#!/usr/bin/env python3
"""Descriptive statistics of funding_paper v2.0 recomputed under the paper_v3 conventions (R0 rules, live universe):
Table 1, funding persistence, predictability of the MOM long leg's funding, turnover and break-even costs, rank ICs,
start weekdays, tails of the MOM long leg. No new tests (registry NOTE paper_v3_extra, 2026-11-20).
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v3_extra.py   -> results/paper/v3_extra.txt"""
import json, statistics
from datetime import datetime, timezone

import numpy as np

import paper_v3 as v
from kit_eval import spearman
from paper_ic_ls import ROOT, OUT, fmt, mean, nw_ols
load = v.load
from paper_funding import spot_opens

year = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).year


def table1(W):
    L = ["Table 1 (contract-weeks; total return and price-weighted funding paid by a long, %)"]
    groups = {}
    for w in W:
        for s, (p, f) in w["ret"].items():
            groups.setdefault(year(w["day"]), []).append((s, p + f, -f))
    groups["All"] = [x for k in list(groups) for x in groups[k]]
    for k, g in groups.items():
        r = np.array([x[1] for x in g]) * 100; f = np.array([x[2] for x in g]) * 100
        z = (r - r.mean()) / r.std()
        wk = len({w["day"] for w in W if k == "All" or year(w["day"]) == k})
        L.append(f"  {k}: weeks {wk} contracts {len({x[0] for x in g})} mean {r.mean():+.2f} median {np.median(r):+.2f} "
                 f"skew {np.mean(z ** 3):.2f} exkurt {np.mean(z ** 4) - 3:.1f} p1 {np.percentile(r, 1):+.1f} "
                 f"p99 {np.percentile(r, 99):+.1f} | funding median {np.median(f):+.3f} p5 {np.percentile(f, 5):+.3f} "
                 f"p95 {np.percentile(f, 95):+.3f}")
    return L


def main():
    data = load()
    hv = v.hour_volume(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
    W, _ = v.build(data, hv, spot, smap)
    L = [f"paper_v3_extra {datetime.now(timezone.utc):%Y-%m-%d}: {len(W)} weeks", ""]
    L += table1(W)
    ever = {s for w in W for s in w["uni"]}
    L.append(f"  contracts ever in universe {len(ever)}; with a Binance spot pair {sum(1 for s in ever if s in smap)}")
    L.append("")
    # persistence
    sl = []
    for w in W:
        ss = [s for s in w["ret"] if s in w["exf"]]
        x = np.array([w["exf"][s] for s in ss]); y = np.array([-w["ret"][s][1] for s in ss])
        if len(ss) >= 12 and x.std() > 0:
            sl.append(np.polyfit(x, y, 1)[0])
    b, t = v.nw(sl)
    L.append(f"Persistence: mean weekly cross-sectional slope of next-week funding on past-7-day funding {b:+.3f} (NW t {t:+.2f}, {len(sl)} weeks)")
    # MOM long leg predictability
    mom = v.ls(W, v.plain("MOM"), 6)
    byday = {w["day"]: w for w in W}
    ex = [mean(byday[r["day"]]["exf"].get(s, 0.0) for s in r["lo"]) for r in mom]
    re_ = [mean(-byday[r["day"]]["ret"][s][1] for s in r["lo"]) for r in mom]
    bb, tt = nw_ols(re_, [ex])
    L.append(f"MOM long leg: ex-ante {v.pct(mean(ex))} realised {v.pct(mean(re_))}; slope of realised on ex-ante {bb[1]:+.2f} (NW t {tt[1]:+.2f})")
    # tails of the MOM long leg
    legs = [byday[r["day"]]["ret"][s][0] + byday[r["day"]]["ret"][s][1] for r in mom for s in r["lo"]]
    legs = np.maximum(np.array(legs), -1)
    big = legs > 0.5
    L.append(f"MOM long leg-weeks {len(legs)}: median {np.median(legs):+.2%} mean {legs.mean():+.2%}; {big.mean():.1%} gain > 50%, "
             f"their share of the sum {legs[big].sum() / legs.sum():.0%}")
    L.append("")
    # turnover, IC
    for k in ("MOM", "LV", "FUND"):
        rows = v.ls(W, v.plain(k), 6)
        ch, held = [], None
        for r in rows:
            now = {s: 1 for s in r["lo"]} | {s: -1 for s in r["sh"]}
            if held is not None:
                ch.append(sum(1 for s, sd in now.items() if held.get(s) != sd) / 12)
            held = now
        g = mean(r["gross"] for r in rows); to = mean(ch)
        ic = []
        for w in W:
            ss = [s for s in w["ret"] if s in w["sig"][k]]
            if len(ss) >= 12:
                ic.append(spearman([w["sig"][k][s] for s in ss], [sum(w["ret"][s]) for s in ss]))
        bi, ti = v.nw(ic)
        L.append(f"{k}: turnover {to:.0%}, gross {v.pct(g)}, break-even cost per side {g / (2 * to) * 100:.2f}%; rank IC {bi:+.3f} (NW t {ti:+.2f})")
    L.append("")
    # weekdays
    L.append("Start weekdays (0 = Monday): net LS and funding part, % per week (NW t)")
    for wd in range(7):
        Wd = W if wd == 0 else v.build(data, hv, spot, smap, weekday=wd)[0]
        parts = []
        for k in ("MOM", "LV", "FUND"):
            r = v.ls(Wd, v.plain(k), 6)
            bn, tn = v.nw([x["net"] for x in r]); bf, tf = v.nw([x["fund"] for x in r])
            parts.append(f"{k} {v.pct(bn)} ({tn:+.2f}) fund {v.pct(bf)} ({tf:+.2f})")
        S, P, T = v.spot_rows(Wd, "LV")
        F = [r["fund"] for r in T]
        bf, tf = v.nw(F)
        L.append(f"  {wd}: " + "  ".join(parts) + f"  | LV spot-test funding {v.pct(bf)} ({tf:+.2f})")
    txt = "\n".join(L)
    print(txt)
    (OUT / "v3_extra.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
