#!/usr/bin/env python3
"""Funding-centred rewrite of the paper (registry PRE-REGISTERED paper_v2, 2026-11-20): items A, B, C, D, E, H and the
NW t of the IC (G). Description on seen data.
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v2.py   -> results/paper/v2.txt"""
import json, statistics
from datetime import datetime, timezone

import numpy as np

import momentum as m
import paper_momentum as pm
from kit_eval import spearman, tstat
from paper_ic_ls import D, N, OUT, ROOT, RAW_FUNDING, analyse, coin_return, fmt, load, mean, nw_ols, signals, weeks
from paper_funding import spot_opens, ls_with_costs, nw_mean

COST = m.SIDE_COST


def universe90(data, day, u):
    """As momentum.universe but ranked by 90-day quote volume (item H)."""
    prev = day - D
    ranked = []
    for sym, c in data.items():
        b = c["bars"]
        if day not in b or prev not in b or prev - (m.MIN_HISTORY - 1) * D not in b:
            continue
        vol = sum(b[prev - i * D][2] for i in range(90) if prev - i * D in b)
        ranked.append((vol, sym))
    ranked.sort(reverse=True)
    return [s for _, s in ranked[:u]]


def vol_sig(data, uni, d, n):
    out = {}
    for s in uni:
        b, r = data[s]["bars"], []
        for i in range(n):
            x, y = b.get(d - (i + 1) * D), b.get(d - (i + 2) * D)
            if x and y and x[1] > 0 and y[1] > 0:
                r.append(x[1] / y[1] - 1)
        if len(r) >= 0.9 * n:
            out[s] = -statistics.stdev(r)
    return out


def spot_test(W, key, spot, label):
    picks, RS, RP, RT = [], [], [], []
    for w in W:
        d = w["day"]
        sig = w["sig"][key]
        ok = [s for s in w["uni"] if s in w["ret"] and s in sig and s in spot and d in spot[s][0]
              and d + 7 * D in spot[s][0] and spot[s][0][d] > 0]
        if len(ok) < 24:
            continue
        v = sorted((sig[s], s) for s in ok)
        picks.append(([s for _, s in v[-N:]], [s for _, s in v[:N]]))
        RS.append({s: spot[s][0][d + 7 * D] / spot[s][0][d] - 1 for s in ok})
        RP.append({s: w["ret"][s][0] for s in ok})
        RT.append({s: w["ret"][s][0] + w["ret"][s][1] for s in ok})
    S, P, T = ls_with_costs(picks, RS), ls_with_costs(picks, RP), ls_with_costs(picks, RT)
    bf, tf = nw_mean([t - p for t, p in zip(T, P)])
    bd, td = nw_mean([s - t for s, t in zip(S, T)])
    return (f"  {label:30s} weeks {len(S)}  spot {mean(S) * 100:+.3f}% (t {tstat(S):+.2f})  perp price {mean(P) * 100:+.3f}% "
            f"(t {tstat(P):+.2f})  perp total {mean(T) * 100:+.3f}% (t {tstat(T):+.2f})  funding {bf * 100:+.3f}% (NW t {tf:+.2f})  "
            f"spot-total {bd * 100:+.3f}% (NW t {td:+.2f})")


def quintiles(W, data, key):
    """A: ex-ante 7-day funding (rate sum, + = longs paid) and realised next-week price-weighted funding by signal quintile."""
    ex = [[] for _ in range(5)]; re_ = [[] for _ in range(5)]
    for w in W:
        d, sig = w["day"], w["sig"][key]
        ss = [s for s in w["uni"] if s in sig and s in w["ret"] and data[s]["times"] and data[s]["times"][0] <= d - 7 * D]
        if len(ss) < 25:
            continue
        o = sorted(ss, key=lambda s: (sig[s], s))
        for q, grp in enumerate(np.array_split(np.array(o), 5)):
            ex[q].append(mean(RAW_FUNDING(data[s], d - 7 * D, d) for s in grp))
            re_[q].append(mean(-w["ret"][s][1] for s in grp))
    out = []
    for q in range(5):
        be, te = nw_mean(ex[q]); br, tr = nw_mean(re_[q])
        out.append(f"Q{q + 1} ex-ante {be * 100:+.3f}% (NW t {te:+.2f}) realised {br * 100:+.3f}% (NW t {tr:+.2f})")
    return out


def decomposition(data, W, key):
    R = analyse(data, W, key)
    ic = [r["ic"] for r in R if r["ic"] is not None]
    net = [r["net"] for r in R]
    b, t = nw_mean(ic)
    parts = "  ".join(f"{lab} {mean(r[f] for r in R) * 100:+.3f}% (t {tstat([r[f] for r in R]):+.2f})"
                      for lab, f in (("long price", "lp"), ("short price", "sp"), ("long funding", "lf"), ("short funding", "sf")))
    return R, f"IC {mean(ic):+.4f} (t {tstat(ic):+.2f}, NW t {t:+.2f})  LS net {mean(net) * 100:+.3f}% (t {tstat(net):+.2f})  {parts}"


def turnover(R, W, key):
    held, ch = {}, []
    for w in W:
        sig = w["sig"][key]
        v = sorted((x, s) for s, x in sig.items())
        if len(v) < 2 * N:
            continue
        now = {s: 1 for _, s in v[-N:]} | {s: -1 for _, s in v[:N]}
        if held:
            ch.append(sum(1 for s, sd in now.items() if held.get(s) != sd) / (2 * N))
        held = now
    gross = mean(r["lp"] + r["sp"] + r["lf"] + r["sf"] for r in R)
    to = mean(ch)
    return f"turnover {to:.0%} of legs per week; gross LS {gross * 100:+.3f}%/wk; break-even cost per side {gross / (2 * to) * 100:.2f}%"


def funding_neutral_mom(data, W):
    """D: within halves split at the median ex-ante funding, long top 3 / short bottom 3 by MOM."""
    picks, RP, RT = [], [], []
    for w in W:
        d, mom = w["day"], w["sig"]["MOM"]
        ss = [s for s in w["uni"] if s in mom and s in w["ret"] and data[s]["times"] and data[s]["times"][0] <= d - 7 * D]
        if len(ss) < 24:
            continue
        f = {s: RAW_FUNDING(data[s], d - 7 * D, d) for s in ss}
        o = sorted(ss, key=lambda s: (f[s], s))
        lo, sh = [], []
        for half in (o[: len(o) // 2], o[len(o) // 2:]):
            v = sorted(half, key=lambda s: (mom[s], s))
            lo += v[-3:]; sh += v[:3]
        picks.append((lo, sh))
        RP.append({s: w["ret"][s][0] for s in ss})
        RT.append({s: w["ret"][s][0] + w["ret"][s][1] for s in ss})
    P, T = ls_with_costs(picks, RP), ls_with_costs(picks, RT)
    bf, tf = nw_mean([t - p for t, p in zip(T, P)])
    word = "largely a funding carry" if tstat(T) < 2 else "partly a funding carry"
    return (f"  funding-neutral MOM ({len(T)} weeks): LS net {mean(T) * 100:+.3f}% (t {tstat(T):+.2f})  price part net "
            f"{mean(P) * 100:+.3f}% (t {tstat(P):+.2f})  funding part {bf * 100:+.3f}% (NW t {tf:+.2f})  -> wording: {word}")


def main():
    data = load()
    W = weeks(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: (spot_opens(s), mult) for p, (s, mult) in smap.items()}
    for w in W:
        for n in (14, 30):
            w["sig"][f"LV{n}"] = vol_sig(data, w["uni"], w["day"], n)
    L = [f"paper_v2 {datetime.now(timezone.utc):%Y-%m-%d}: {len(W)} weeks {fmt(W[0]['day'])}..{fmt(W[-1]['day'])}", ""]
    L.append("A Funding by signal quintile (Q5 = highest signal = long leg)")
    for key in ("MOM", "LV"):
        L += [f"  {key} " + x for x in quintiles(W, data, key)]
    L += ["", "B/C Spot vs perp (same picks, contracts with a spot pair)"]
    for key, lab in (("LV", "LV 7d"), ("LV14", "LV 14d"), ("LV30", "LV 30d"), ("MOM", "MOM 30d"), ("FUND", "FUND")):
        L.append(spot_test(W, key, spot, lab))
    L += ["", "C LV full universe and perp-only contracts"]
    R, line = decomposition(data, W, "LV")
    L.append("  LV full universe: " + line)
    po, sp_ = [], []
    for w, r in zip(W, R):
        v = sorted((x, s) for s, x in w["sig"]["LV"].items())
        for _, s in v[:N]:
            if s in w["ret"]:
                (po if s not in smap else sp_).append(-sum(w["ret"][s]))
    L.append(f"  LV short legs: perp-only {len(po)} leg-weeks ({len(po) / (len(po) + len(sp_)):.0%}), mean {mean(po) * 100:+.2f}% "
             f"median {statistics.median(po) * 100:+.2f}%; with a spot pair mean {mean(sp_) * 100:+.2f}% median {statistics.median(sp_) * 100:+.2f}%  (sign flipped: + = gain for the strategy)")
    L += ["", "D Funding-neutral momentum", funding_neutral_mom(data, W)]
    L += ["", "E Turnover and break-even costs"]
    for key in ("MOM", "LV", "FUND"):
        Rk, line = decomposition(data, W, key)
        L.append(f"  {key}: " + turnover(Rk, W, key))
        L.append(f"  {key} (G, NW t of the IC): " + line)
    L += ["", "H Universe ranked by 90-day quote volume"]
    real_universe = m.universe
    m.universe = universe90
    try:
        W90 = weeks(data)
    finally:
        m.universe = real_universe
    for key in ("MOM", "LV"):
        L.append(f"  {key}: " + decomposition(data, W90, key)[1])
    L.append(spot_test(W90, "LV", spot, "LV 7d (90d universe)"))
    L.append(spot_test(W90, "MOM", spot, "MOM (90d universe)"))
    txt = "\n".join(L)
    print(txt)
    (OUT / "v2.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
