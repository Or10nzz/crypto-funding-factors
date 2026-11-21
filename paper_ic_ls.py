#!/usr/bin/env python3
"""Main tables for the paper "When Rank IC and Long-Short Returns Disagree: Momentum and Low Volatility in Crypto
Perpetual Futures". Settings fixed in results/registry.txt NOTE paper_ic_ls (2026-11-20).
Description only: every week here has been seen before. No orders, no live code.

  ~/bin/heavy ~/.venvs/quant/bin/python paper_ic_ls.py      -> results/paper/tables.txt, results/paper/weekly.json
"""
import json, math, statistics, sys
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import holdout_momentum as hm          # sets momentum.END to 2026-11-01
import momentum as m
import paper_momentum as pm
from kit_eval import spearman, tstat

ROOT, D, N = Path(__file__).parent, m.DAY, 6
RAW_FUNDING = m.funding                                  # plain sum of rates (used for the FUND signal only)


def _price_at(b, t):
    """Price at time t, linear between the open and the close of its UTC day (archive has daily bars only)."""
    day = t - t % D
    x = b.get(day)
    if not x:
        return None
    o, c = x[0], x[1]
    if c is None:
        return o
    return o + (c - o) * ((t - day) / D)


H = 3600_000
FUNDPX = ROOT / "data/universe/fundpx.pkl"          # cache: {symbol: [hourly open at each funding settlement or None]}


def attach_settlement_prices(data):
    """c["fpx"][k] = open of the 1-hour perp kline that starts at funding settlement c["times"][k] (None if missing).
    Built once from data/universe/klines/<SYM>/1h and cached (registry CLARIFICATION 2026-11-20)."""
    import pickle
    cache = pickle.loads(FUNDPX.read_bytes()) if FUNDPX.exists() else {}
    changed = False
    for sym, c in data.items():
        times = c.get("times") or []
        got = cache.get(sym)
        if got is not None and len(got) == len(times):
            c["fpx"] = got
            continue
        hours = {}
        for f in sorted((ROOT / "data/universe/klines" / sym / "1h").glob("*.zip")):
            for r in m.rows(f):
                if r and r[0].isdigit():
                    t = int(r[0])
                    hours[t // 1000 if t > 10 ** 14 else t] = float(r[1])
        c["fpx"] = [hours.get(t - t % H) for t in times]
        cache[sym] = c["fpx"]
        changed = True
    if changed:
        FUNDPX.write_bytes(pickle.dumps(cache))
    return data


def path_funding(c, t0, t1):
    """Funding paid by a long of notional 1 at t0 (registry CLARIFICATION 2026-11-20): sum over settlements in (t0, t1]
    of rate x P(t) / P(t0). Falls back to the plain sum if the entry price is unknown."""
    b = c["bars"]
    p0 = b[t0][0] if t0 in b and b[t0][0] > 0 else None
    if p0 is None:
        return RAW_FUNDING(c, t0, t1)
    times, pre = c["times"], c["pre"]
    i, j = bisect_right(times, t0), bisect_right(times, t1)
    tot = 0.0
    fpx = c.get("fpx")
    for k in range(i, j):
        rate = pre[k + 1] - pre[k]
        p = fpx[k] if fpx is not None and fpx[k] else _price_at(b, times[k])
        tot += rate * (p / p0 if p else 1.0)
    return tot


m.funding = path_funding                                   # used by momentum.leg_return / run and by this module
OUT = ROOT / "results/paper"
SPLIT = int(datetime(2023, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
fmt = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")
mean = statistics.fmean


def load():
    data = m.load()
    for s in hm.non_coins():
        data.pop(s, None)
    for s in [s for s, c in data.items() if hm.stable_like(c)]:
        data.pop(s)
    return attach_settlement_prices(data)


def signals(data, uni, d, lookback=30):
    prev = d - D
    out = {"MOM": {}, "LV": {}, "FUND": {}}
    for s in uni:
        c = data[s]
        b = c["bars"]
        if prev in b and prev - lookback * D in b and b[prev - lookback * D][1] > 0:
            out["MOM"][s] = b[prev][1] / b[prev - lookback * D][1] - 1
        if (r := pm.retvol(b, d)) is not None:
            out["LV"][s] = -r
        if c["times"] and c["times"][0] <= d - 7 * D:
            out["FUND"][s] = -RAW_FUNDING(c, d - 7 * D, d)
    return out


def coin_return(c, d):
    """(price part, funding part) for a long held open(d) -> open(d+7); None if the coin has no bar at d+7."""
    b = c["bars"]
    if d + 7 * D not in b or b[d][0] <= 0:
        return None
    return b[d + 7 * D][0] / b[d][0] - 1, -m.funding(c, d, d + 7 * D)


def weeks(data, weekday=0, lookback=30):
    """All start days on this weekday (0 = Monday) with a full top-30 universe."""
    last = max(t for c in data.values() for t in c["bars"])
    d = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    while ((d // D) + 3) % 7 != weekday:
        d += D
    out = []
    while d + 7 * D <= last:
        uni = m.universe(data, d, 30)
        if len(uni) == 30:
            out.append(dict(day=d, uni=uni, sig=signals(data, uni, d, lookback),
                            ret={s: r for s in uni if (r := coin_return(data[s], d)) is not None}))
        d += 7 * D
    return out


def picks(sig):
    v = sorted((x, s) for s, x in sig.items())
    return ([s for _, s in v[-N:]], [s for _, s in v[:N]]) if len(v) >= 2 * N else ([], [])


def analyse(data, W, key):
    """Per-week rows for one signal."""
    P = [picks(w["sig"][key]) for w in W]
    net = m.run(data, [w["day"] for w in W], [w["uni"] for w in W], lambda i: P[i], N)
    rows = []
    for w, (lo, sh), r in zip(W, P, net):
        R = w["ret"]
        ss = [s for s in w["sig"][key] if s in R]
        ic = spearman([w["sig"][key][s] for s in ss], [sum(R[s]) for s in ss]) if len(ss) >= 12 else None
        icp = spearman([w["sig"][key][s] for s in ss], [R[s][0] for s in ss]) if len(ss) >= 12 else None
        lo_, sh_ = [s for s in lo if s in R], [s for s in sh if s in R]
        legs = [sum(R[s]) for s in lo_] + [-sum(R[s]) for s in sh_]
        rows.append(dict(day=w["day"], ic=ic, icp=icp, net=r["ret"],
                         lp=mean(R[s][0] for s in lo_) / 2 if lo_ else 0.0, sp=-mean(R[s][0] for s in sh_) / 2 if sh_ else 0.0,
                         lf=mean(R[s][1] for s in lo_) / 2 if lo_ else 0.0, sf=-mean(R[s][1] for s in sh_) / 2 if sh_ else 0.0,
                         long_legs=[sum(R[s]) for s in lo_], short_legs=[-sum(R[s]) for s in sh_], legs=legs,
                         mkt=mean(sum(R[s]) for s in R)))
    return rows


def nw_ols(y, X, lags=4):
    """OLS with Newey-West standard errors. X without constant; returns (coef, t) with the constant first."""
    y = np.asarray(y)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x) for x in X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return b, b / np.sqrt(np.diag(V))


def ts(x):
    return f"{mean(x) * 100:+.2f}% (t {tstat(x):+.2f})" if len(x) > 2 else "—"


def table(data, W, keys=("MOM", "LV", "FUND")):
    L, rows = [], {}
    for k in keys:
        rows[k] = R = analyse(data, W, k)
        ic = [r["ic"] for r in R if r["ic"] is not None]
        icp = [r["icp"] for r in R if r["icp"] is not None]
        gross = [r["lp"] + r["sp"] + r["lf"] + r["sf"] for r in R]
        net = [r["net"] for r in R]
        L.append(f"  {k:5s} IC {mean(ic):+.4f} (t {tstat(ic):+.2f})  price-only IC {mean(icp):+.4f} (t {tstat(icp):+.2f})  "
                 f"LS gross {ts(gross)}  net {ts(net)}  Sharpe {mean(net) / statistics.stdev(net) * math.sqrt(52):.2f}  "
                 f"weeks {len(R)}  log-t {tstat([math.log1p(x) for x in net]):+.2f}")
    return L, rows


def main():
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    W = weeks(data)
    L = [f"paper_ic_ls  {datetime.now(timezone.utc):%Y-%m-%d}  Mondays {fmt(W[0]['day'])} .. {fmt(W[-1]['day'])}  ({len(W)} weeks), "
         f"top 30 Binance USDT-M perps, 6/6, 0.10%/side, funding included", "",
         "Table 2. Two rulers (full sample)"]
    t2, rows = table(data, W)
    L += t2
    for lab, cond in (("before 2023-09", lambda w: w["day"] < SPLIT), ("from 2023-09", lambda w: w["day"] >= SPLIT)):
        L += ["", f"Table 2b. {lab}"] + table(data, [w for w in W if cond(w)])[0]

    L += ["", "Table 3. Decomposition, % of capital per week (each leg = half the book): long price | short price | long funding | short funding"]
    for k, R in rows.items():
        L.append(f"  {k:5s} all   " + "  ".join(ts([r[f] for r in R]) for f in ("lp", "sp", "lf", "sf")))
        for y in sorted({fmt(r["day"])[:4] for r in R}):
            Ry = [r for r in R if fmt(r["day"])[:4] == y]
            L.append(f"        {y}  " + "  ".join(f"{mean(r[f] for r in Ry) * 100 * 52:+6.1f}" for f in ("lp", "sp", "lf", "sf"))
                     + "   (%/yr)")

    L += ["", "Table 4. Tails (leg-week total returns, long legs as is, short legs sign-flipped)"]
    for k, R in rows.items():
        allleg = sorted(x for r in R for x in r["legs"])
        a, z = allleg[int(0.01 * len(allleg))], allleg[int(0.99 * len(allleg)) - 1]
        raw = [mean(r["legs"]) for r in R if r["legs"]]
        trim = [mean(min(max(x, a), z) for x in r["legs"]) for r in R if r["legs"]]
        ll = [x for r in R for x in r["long_legs"]]
        sl = [x for r in R for x in r["short_legs"]]
        big = sum(x for x in ll if x > 0.5) / len(ll) if ll else 0.0
        L.append(f"  {k:5s} mean leg {mean(raw) * 100:+.2f}%  trimmed 1/99 {mean(trim) * 100:+.2f}%  "
                 f"long legs: mean {mean(ll) * 100:+.2f}% median {statistics.median(ll) * 100:+.2f}%  "
                 f"short legs: mean {mean(sl) * 100:+.2f}% median {statistics.median(sl) * 100:+.2f}%  "
                 f"long legs > +50%: {sum(x > 0.5 for x in ll) / len(ll):.1%} of legs, {big * 100:+.2f} pts of the {mean(ll) * 100:+.2f}% mean")

    L += ["", "Table 5. Robustness"]
    for wd in range(7):
        Wd = W if wd == 0 else weeks(data, wd)
        out = []
        for k in ("MOM", "LV"):
            R = analyse(data, Wd, k)
            ic = [r["ic"] for r in R if r["ic"] is not None]
            out.append(f"{k} IC {mean(ic):+.3f} (t {tstat(ic):+.1f}) net {ts([r['net'] for r in R])}")
        L.append(f"  start weekday {wd} ({['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][wd]}): " + "   ".join(out))
    for lb in (7, 14, 60):
        R = analyse(data, weeks(data, 0, lb), "MOM")
        ic = [r["ic"] for r in R if r["ic"] is not None]
        L.append(f"  MOM lookback {lb:2d}d: IC {mean(ic):+.4f} (t {tstat(ic):+.2f})  net {ts([r['net'] for r in R])}")

    mom, fund = rows["MOM"], rows["FUND"]
    y = [r["net"] for r in mom]
    for lab, X in (("on market", [[r["mkt"] for r in mom]]), ("on market + FUND LS", [[r["mkt"] for r in mom], [r["net"] for r in fund]])):
        b, t = nw_ols(y, X)
        L.append(f"  MOM LS {lab}: alpha {b[0] * 100:+.3f}%/wk (NW t {t[0]:+.2f})  slopes " +
                 "  ".join(f"{bb:+.2f} (t {tt:+.2f})" for bb, tt in zip(b[1:], t[1:])))
    L.append(f"  corr(MOM LS, FUND LS) {statistics.correlation(y, [r['net'] for r in fund]):+.2f}")
    txt = "\n".join(L)
    print(txt)
    (OUT / "tables.txt").write_text(txt + "\n")
    (OUT / "weekly.json").write_text(json.dumps({k: [{f: v for f, v in r.items() if f not in ("legs",)} for r in R]
                                                 for k, R in rows.items()}))


if __name__ == "__main__":
    main()
