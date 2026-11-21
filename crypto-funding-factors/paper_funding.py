#!/usr/bin/env python3
"""Direction test for the paper (registry PRE-REGISTERED paper_funding, 2026-11-20). Description on seen data.
Test 1: LV long-short on spot prices vs perp price vs perp price + funding (same picks).
Test 2: is momentum a funding carry? persistence of funding, ex-ante funding of the MOM legs, predictability.
  ~/bin/heavy ~/.venvs/quant/bin/python paper_funding.py  -> results/paper/funding_test.txt"""
import json, statistics
from datetime import datetime, timezone

import numpy as np

import momentum as m
from kit_eval import tstat
from paper_ic_ls import D, N, OUT, ROOT, RAW_FUNDING, fmt, load, mean, nw_ols, weeks

COST = m.SIDE_COST


def spot_opens(sym):
    out = {}
    for f in sorted((ROOT / "data/spot/klines" / sym).glob("*.zip")):
        for r in m.rows(f):
            if r and r[0].isdigit():
                t = int(r[0])
                out[t // 1000 if t > 10 ** 14 else t] = float(r[1])
    return out


def nw_mean(x):
    b, t = nw_ols(x, [])
    return b[0], t[0]


def ls_with_costs(picks, rets):
    """picks: list of (longs, shorts); rets: list of dicts sym -> return. Equal weight 6/6, cost on changed legs."""
    out, held = [], {}
    for (lo, sh), R in zip(picks, rets):
        now = {s: 1 for s in lo} | {s: -1 for s in sh}
        changed = sum(1 for s, sd in now.items() if held.get(s) != sd) + sum(1 for s, sd in held.items() if now.get(s) != sd)
        held = now
        g = (mean(R[s] for s in lo) - mean(R[s] for s in sh)) / 2
        out.append(g - changed / (2 * N) * COST)
    return out


def main():
    data = load()
    W = weeks(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: (spot_opens(s), mult) for p, (s, mult) in smap.items()}
    L = [f"paper_funding {datetime.now(timezone.utc):%Y-%m-%d}: {len(W)} weeks {fmt(W[0]['day'])}..{fmt(W[-1]['day'])}", ""]

    # ---------- Test 1
    picks, RS, RP, RT, days = [], [], [], [], []
    for w in W:
        d = w["day"]
        ok = [s for s in w["uni"] if s in w["ret"] and s in w["sig"]["LV"] and s in spot
              and d in spot[s][0] and d + 7 * D in spot[s][0] and spot[s][0][d] > 0]
        if len(ok) < 24:
            continue
        v = sorted((w["sig"]["LV"][s], s) for s in ok)
        lo, sh = [s for _, s in v[-N:]], [s for _, s in v[:N]]
        picks.append((lo, sh)); days.append(d)
        RS.append({s: spot[s][0][d + 7 * D] / spot[s][0][d] - 1 for s in ok})
        RP.append({s: w["ret"][s][0] for s in ok})
        RT.append({s: w["ret"][s][0] + w["ret"][s][1] for s in ok})
    S, P, T = ls_with_costs(picks, RS), ls_with_costs(picks, RP), ls_with_costs(picks, RT)
    F = [t - p for t, p in zip(T, P)]                      # funding component (costs cancel)
    diff = [s - t for s, t in zip(S, T)]
    bf, tf = nw_mean(F); bd, td = nw_mean(diff)
    L += [f"Test 1 (LV, {len(S)} weeks with >= 24 spot pairs, {fmt(days[0])}..{fmt(days[-1])}; net of 0.10%/side)",
          f"  (S) spot price          {mean(S) * 100:+.3f}%/wk  t {tstat(S):+.2f}",
          f"  (P) perp price only      {mean(P) * 100:+.3f}%/wk  t {tstat(P):+.2f}",
          f"  (T) perp price + funding {mean(T) * 100:+.3f}%/wk  t {tstat(T):+.2f}",
          f"  funding component T - P  {bf * 100:+.3f}%/wk  NW t {tf:+.2f}",
          f"  spot minus perp total S - T {bd * 100:+.3f}%/wk  NW t {td:+.2f}",
          f"  spot vs perp price S - P {mean([s - p for s, p in zip(S, P)]) * 100:+.3f}%/wk",
          f"  PASS1 = {tf <= -2 and bf <= 0 and bd > 0 and td >= 2}", ""]

    # ---------- Test 2
    slopes, exl, exs, rel, res_, rows = [], [], [], [], [], []
    for w in W:
        d = w["day"]
        x, y = {}, {}
        for s in w["uni"]:
            c = data[s]
            if s in w["ret"] and c["times"] and c["times"][0] <= d - 7 * D:
                x[s] = RAW_FUNDING(c, d - 7 * D, d)        # past 7 days, rate sum (> 0: longs paid)
                y[s] = -w["ret"][s][1]                       # next week, price-weighted, paid by a long
        ss = list(x)
        if len(ss) >= 12 and statistics.pstdev([x[s] for s in ss]) > 0:
            slopes.append(np.polyfit([x[s] for s in ss], [y[s] for s in ss], 1)[0])
        mom = w["sig"]["MOM"]
        v = sorted((mom[s], s) for s in ss if s in mom)
        if len(v) >= 2 * N:
            lo, sh = [s for _, s in v[-N:]], [s for _, s in v[:N]]
            exl.append(mean(x[s] for s in lo)); exs.append(mean(x[s] for s in sh))
            rel.append(mean(y[s] for s in lo)); res_.append(mean(y[s] for s in sh))
    b_slope, t_slope = mean(slopes), tstat(slopes)
    b_c, t_c = nw_ols(rel, [exl])
    L += [f"Test 2 (MOM and funding, {len(slopes)} weeks)",
          f"  (a) persistence: weekly cross-sectional slope of next-week funding on past-7-day funding: mean {b_slope:+.3f}  t {t_slope:+.2f}",
          f"  (b) MOM long leg  ex-ante funding (past 7d, + = longs paid) {mean(exl) * 100:+.3f}%  t {tstat(exl):+.2f}   realised next week {mean(rel) * 100:+.3f}%  t {tstat(rel):+.2f}",
          f"      MOM short leg ex-ante funding                            {mean(exs) * 100:+.3f}%  t {tstat(exs):+.2f}   realised next week {mean(res_) * 100:+.3f}%  t {tstat(res_):+.2f}",
          f"  (c) long-leg realised funding on its ex-ante funding (weekly, NW): slope {b_c[1]:+.3f}  t {t_c[1]:+.2f}  (intercept {b_c[0] * 100:+.3f}%)",
          f"  PASS2 = {b_slope > 0 and t_slope >= 2 and mean(exl) < 0 and tstat(exl) <= -2 and b_c[1] > 0 and t_c[1] >= 2}"]
    txt = "\n".join(L)
    print(txt)
    (OUT / "funding_test.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
