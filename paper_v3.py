#!/usr/bin/env python3
"""Response to the blind review of funding_paper v1.0 (registry PRE-REGISTERED paper_v3, 2026-11-20).
Description on seen data. Items R0 (trading stop, leg cap, weekly spot flag), A-G.
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v3.py   -> results/paper/v3.txt"""
import json, math, pickle, statistics
from bisect import bisect_left
from datetime import datetime, timezone

import numpy as np

import momentum as m
import paper_momentum as pm
import holdout_momentum as hm
import paper_ic_ls as pil
from paper_ic_ls import D, OUT, ROOT, RAW_FUNDING, fmt, mean, nw_ols, path_funding
from paper_funding import spot_opens
from paper_v2 import universe90, vol_sig

H = 3600_000
COST = m.SIDE_COST
HOURVOL = ROOT / "data/universe/hourvol.pkl"     # cache: {sym: (open times of hours with quote volume > 0, closes)}
ALPACA_WEEK = int(datetime(2025, 4, 28, tzinfo=timezone.utc).timestamp() * 1000)
SPLIT = int(datetime(2023, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- data
def load():
    """paper_ic_ls.load with funding times rounded to the nearest minute (registry CLARIFICATION paper_v3, bug fix)."""
    data = m.load()
    for s in hm.non_coins():
        data.pop(s, None)
    for s in [s for s, c in data.items() if hm.stable_like(c)]:
        data.pop(s)
    for c in data.values():
        c["times"] = [(t + 30_000) // 60_000 * 60_000 for t in c["times"]]
    pil.FUNDPX = ROOT / "data/universe/fundpx_rounded.pkl"
    return pil.attach_settlement_prices(data)


def hour_volume(data):
    cache = pickle.loads(HOURVOL.read_bytes()) if HOURVOL.exists() else {}
    changed = False
    for sym in data:
        if sym in cache:
            continue
        T, C = [], []
        for f in sorted((ROOT / "data/universe/klines" / sym / "1h").glob("*.zip")):
            for r in m.rows(f):
                if r and r[0].isdigit() and float(r[7]) > 0:
                    t = int(r[0])
                    T.append(t // 1000 if t > 10 ** 14 else t); C.append(float(r[4]))
        o = np.argsort(T)
        cache[sym] = (np.asarray(T, dtype=np.int64)[o], np.asarray(C)[o])
        changed = True
    if changed:
        HOURVOL.write_bytes(pickle.dumps(cache))
    return cache


def coin_week(c, hv, d):
    """Long-side (price, funding received) from open(d) to d+7 under R0(a), and under the v1.0 rule, and a stop flag.
    None if no entry price."""
    b = c["bars"]
    if d not in b or b[d][0] <= 0:
        return None
    p0, nxt = b[d][0], d + 7 * D
    # v1.0 rule (momentum.leg_return): next open, else last daily close
    if nxt in b:
        v1 = (b[nxt][0] / p0 - 1, -path_funding(c, d, nxt))
    else:
        dd = d
        while dd + D in b and dd + D < nxt:
            dd += D
        v1 = (b[dd][1] / p0 - 1, -path_funding(c, d, dd + D))
    T, C = hv if hv is not None else (None, None)
    if T is None or len(T) == 0:
        return v1, v1, False, None
    k = bisect_left(T, nxt) - 1
    lastday = b.get(nxt - D)
    # the last 24 h before d+7 are the UTC day nxt-D: the daily bar guards against gaps in the hourly archive
    stopped = nxt not in b or lastday is None or lastday[2] <= 0
    if not stopped:
        return (b[nxt][0] / p0 - 1, -path_funding(c, d, nxt)), v1, False, None
    if k >= 0 and T[k] >= d and T[0] <= d:
        t1 = int(T[k]) + H
        return (C[k] / p0 - 1, -path_funding(c, d, t1)), v1, True, t1
    dd = None                                   # hourly archive does not cover: last daily bar with volume
    for i in range(6, -1, -1):
        x = b.get(d + i * D)
        if x and x[2] > 0:
            dd = d + i * D; break
    if dd is None:
        return (0.0, 0.0), v1, True, d
    return (b[dd][1] / p0 - 1, -path_funding(c, d, dd + D)), v1, True, dd + D


def live(universe):
    """Universe without contracts that had zero quote volume on d-1 (registry CLARIFICATION paper_v3, post-hoc)."""
    def f(data, day, u):
        out = [s for s in universe(data, day, u + 20)
               if (x := data[s]["bars"].get(day - D)) is not None and x[2] > 0]
        return out[:u]
    return f


def build(data, hv, spot, smap, universe=m.universe, u=30, weekday=0):
    universe = live(universe)
    first = {s: min(c["bars"]) for s, c in data.items()}
    last = max(t for c in data.values() for t in c["bars"])
    d = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    while ((d // D) + 3) % 7 != weekday:
        d += D
    W, stops = [], []
    while d + 7 * D <= last:
        uni = universe(data, d, u)
        if len(uni) == u:
            prev = d - D
            w = dict(day=d, uni=uni, ret={}, ret1={}, sig={"MOM": {}, "LV": {}, "FUND": {}}, exf={}, exo={}, surge={},
                     age={}, vol30={}, spot={}, perponly={})
            for s in uni:
                c, b = data[s], data[s]["bars"]
                x = coin_week(c, hv.get(s), d)
                if x is None:
                    continue
                w["ret"][s], w["ret1"][s] = x[0], x[1]
                if x[2]:
                    stops.append((s, d, x[3]))
                if prev in b and prev - 30 * D in b and b[prev - 30 * D][1] > 0:
                    w["sig"]["MOM"][s] = b[prev][1] / b[prev - 30 * D][1] - 1
                if (r := pm.retvol(b, d)) is not None:
                    w["sig"]["LV"][s] = -r
                if c["times"] and c["times"][0] <= d - 14 * D:
                    w["exf"][s] = RAW_FUNDING(c, d - 7 * D, d)
                    w["exo"][s] = RAW_FUNDING(c, d - 14 * D, d - 7 * D)
                    w["sig"]["FUND"][s] = -w["exf"][s]
                v30 = sum(b[prev - i * D][2] for i in range(30) if prev - i * D in b)
                v90 = sum(b[prev - i * D][2] for i in range(90) if prev - i * D in b)
                w["vol30"][s] = v30
                w["surge"][s] = v30 / (v90 / 3) if v90 > 0 else None
                w["age"][s] = math.log1p((d - first[s]) / D)
                sp = spot.get(s)
                has = sp is not None and d in sp and sp[d] > 0
                w["perponly"][s] = not has
                if has and d + 7 * D in sp:
                    w["spot"][s] = sp[d + 7 * D] / sp[d] - 1
            for n in (14, 30):
                w["sig"][f"LV{n}"] = vol_sig(data, uni, d, n)
            W.append(w)
        d += 7 * D
    return W, stops


# ---------------------------------------------------------------- portfolio
def top_bottom(sig, n, ok):
    v = sorted((x, s) for s, x in sig.items() if s in ok)
    return ([s for _, s in v[-n:]], [s for _, s in v[:n]]) if len(v) >= 2 * n else None


def ls(W, pick, n, retkey="ret", parts="pf", weight=None, cost_of=None, cap=True):
    """Weekly long-short (book = 1/2 long, 1/2 short). Returns list of dicts with net, gross, price, fund, legs."""
    out, held = [], {}
    caps = 0
    for w in W:
        R = w[retkey] if isinstance(retkey, str) else retkey(w)
        pk = pick(w, R)
        if pk is None:
            continue
        lo, sh = pk
        now = {s: 1 for s in lo} | {s: -1 for s in sh}
        wt = {}
        for side, grp in ((1, lo), (-1, sh)):
            if weight is None:
                for s in grp:
                    wt[s] = 0.5 / len(grp)
            else:
                tot = sum(weight(w, s) for s in grp)
                for s in grp:
                    wt[s] = 0.5 * weight(w, s) / tot
        pr = fu = 0.0
        lp = sp_ = lf = sf = 0.0
        for s, sd in now.items():
            p, f = R[s]
            p, f = sd * p, sd * f
            if parts == "p":
                f = 0.0
            if cap and p + f < -1:
                k = -1 / (p + f); p, f = p * k, f * k; caps += 1
            pr += wt[s] * p; fu += wt[s] * f
            if sd == 1:
                lp += wt[s] * p; lf += wt[s] * f
            else:
                sp_ += wt[s] * p; sf += wt[s] * f
        cst = 0.0
        for s, sd in now.items():
            if held.get(s) != sd:
                cst += (1 / (2 * n)) * (cost_of(w, s) if cost_of else COST)
        for s, sd in held.items():
            if now.get(s) != sd:
                cst += (1 / (2 * n)) * (cost_of(w, s) if cost_of else COST)
        held = now
        g = pr + fu
        out.append(dict(day=w["day"], gross=g, net=g - cst, price=pr, fund=fu, lp=lp, sp=sp_, lf=lf, sf=sf, lo=lo, sh=sh,
                        mkt=mean(sum(R[s]) for s in R)))
    if out:
        out[0]["caps"] = caps
    return out


def plain(key, n=6):
    return lambda w, R: top_bottom(w["sig"][key], n, R)


def ranks(d):
    ks = sorted(d, key=lambda s: (d[s], s))
    return {s: i / (len(ks) - 1) for i, s in enumerate(ks)} if len(ks) > 1 else {}


def mom_resid(w, R):
    ok = [s for s in w["sig"]["MOM"] if s in w["exf"] and s in R]
    if len(ok) < 12:
        return None
    rm, rf = ranks({s: w["sig"]["MOM"][s] for s in ok}), ranks({s: w["exf"][s] for s in ok})
    x = np.array([rf[s] for s in ok]); y = np.array([rm[s] for s in ok])
    bb = np.polyfit(x, y, 1)
    res = {s: y[i] - (bb[0] * x[i] + bb[1]) for i, s in enumerate(ok)}
    return top_bottom(res, 6, R)


def mom_5x5(w, R):
    ok = [s for s in w["sig"]["MOM"] if s in w["exf"] and s in R]
    if len(ok) < 25:
        return None
    o = sorted(ok, key=lambda s: (w["exf"][s], s))
    lo, sh = [], []
    for grp in np.array_split(np.array(o, dtype=object), 5):
        v = sorted(grp, key=lambda s: (w["sig"]["MOM"][s], s))
        lo.append(v[-1]); sh.append(v[0])
    return lo, sh


# ---------------------------------------------------------------- statistics
def nw(x):
    if len(x) < 10:
        return float("nan"), float("nan")
    b, t = nw_ols(list(x), [])
    return b[0], t[0]


def block_idx(n, block=8):
    starts = RNG.integers(0, n - block + 1, size=n // block + 1)
    return np.concatenate([np.arange(s0, s0 + block) for s0 in starts])[:n]


def boot(fn, n, reps=2000):
    v = np.array([fn(block_idx(n)) for _ in range(reps)])
    return np.nanpercentile(v, 5), np.nanpercentile(v, 95)


def pct(x):
    return f"{x * 100:+.3f}%"


def line(rows, lab):
    g = lambda k: [r[k] for r in rows]
    bn, tn = nw(g("net"))
    bp, tp = nw(g("price"))
    bf, tf = nw(g("fund"))
    return (f"  {lab:34s} wks {len(rows)} net {pct(bn)} (NW t {tn:+.2f})  price {pct(bp)} ({tp:+.2f})  "
            f"funding {pct(bf)} ({tf:+.2f})  [lp {pct(mean(g('lp')))} sp {pct(mean(g('sp')))} lf {pct(mean(g('lf')))} "
            f"sf {pct(mean(g('sf')))}]")


def amounts(rows, lab):
    P, F, G = (np.array([r[k] for r in rows]) for k in ("price", "fund", "gross"))
    n = len(G)
    ci = lambda a: boot(lambda i: a[i].mean(), n)
    sh = boot(lambda i: F[i].mean() / G[i].mean(), n)
    (pl, ph), (fl, fh), (gl, gh) = ci(P), ci(F), ci(G)
    return (f"  {lab:34s} gross {pct(G.mean())} [{pct(gl)},{pct(gh)}]  price {pct(P.mean())} [{pct(pl)},{pct(ph)}]  "
            f"funding {pct(F.mean())} [{pct(fl)},{pct(fh)}]  funding share {F.mean() / G.mean():.0%} [{sh[0]:.0%},{sh[1]:.0%}]")


def spot_rows(W, key, retkey="ret"):
    """Table 3 design: same picks among contracts with a spot return this week; spot / perp price / perp total."""
    def ok_w(w):
        R = w[retkey]
        return {s: R[s] for s in w["spot"] if s in R and s in w["sig"][key]}
    Ws = [w for w in W if len(ok_w(w)) >= 24]
    pick = lambda w, R: top_bottom(w["sig"][key], 6, R)
    S = ls(Ws, pick, 6, retkey=lambda w: {s: (w["spot"][s], 0.0) for s in ok_w(w)})
    P = ls(Ws, pick, 6, retkey=ok_w, parts="p")
    T = ls(Ws, pick, 6, retkey=ok_w)
    return S, P, T


def spot_line(W, key, lab, retkey="ret", boot_ci=False):
    S, P, T = spot_rows(W, key, retkey)
    s, ts_ = nw([r["net"] for r in S]); p, tp = nw([r["net"] for r in P]); t, tt = nw([r["net"] for r in T])
    F = np.array([a["net"] - b["net"] for a, b in zip(T, P)])
    f, tf = nw(F)
    extra = ""
    if boot_ci:
        Sa, Ta = np.array([r["net"] for r in S]), np.array([r["net"] for r in T])
        a, b_ = boot(lambda i: Sa[i].mean(), len(Sa)); c_, d_ = boot(lambda i: Ta[i].mean(), len(Ta))
        e, f_ = boot(lambda i: F[i].mean(), len(F))
        extra = f"  boot5-95 spot [{pct(a)},{pct(b_)}] perp total [{pct(c_)},{pct(d_)}] funding [{pct(e)},{pct(f_)}]"
    return (f"  {lab:34s} wks {len(S)} spot {pct(s)} (NW t {ts_:+.2f})  perp price {pct(p)} ({tp:+.2f})  perp total {pct(t)} "
            f"({tt:+.2f})  funding {pct(f)} (NW t {tf:+.2f}){extra}"), tf


def drop_extreme(W, q):
    tot = np.array([abs(sum(r)) for w in W for r in w["ret"].values()])
    thr = np.percentile(tot, 100 - q)
    out = []
    for w in W:
        bad = {s for s, r in w["ret"].items() if abs(sum(r)) > thr}
        x = dict(w)
        x["ret"] = {s: r for s, r in w["ret"].items() if s not in bad}
        x["spot"] = {s: r for s, r in w["spot"].items() if s not in bad}
        out.append(x)
    return out, thr


def quint(W, sigkey, exkey):
    ex = [[] for _ in range(5)]; re_ = [[] for _ in range(5)]
    for w in W:
        sig = w["sig"][sigkey] if sigkey in w["sig"] else w[sigkey]
        ss = [s for s in w["ret"] if s in w[exkey] and sig.get(s) is not None]
        if len(ss) < 25:
            continue
        o = sorted(ss, key=lambda s: (sig[s], s))
        for q, grp in enumerate(np.array_split(np.array(o, dtype=object), 5)):
            ex[q].append(mean(w[exkey][s] for s in grp))
            re_[q].append(mean(-w["ret"][s][1] for s in grp))
    out = []
    for q in range(5):
        be, te = nw(ex[q]); br, tr = nw(re_[q])
        out.append(f"Q{q + 1} ex-ante {pct(be)} (NW t {te:+.2f}) realised {pct(br)} (NW t {tr:+.2f})")
    return out, ex


def capm(rows):
    b, t = nw_ols([r["net"] for r in rows], [[r["mkt"] for r in rows]])
    return f"alpha {pct(b[0])} (NW t {t[0]:+.2f}) beta {b[1]:+.3f} (NW t {t[1]:+.2f})"


def fama_macbeth(W):
    names = ["MOM", "LV", "FUND", "surge", "age", "perponly"]
    coefs = {k: {n: [] for n in ["const"] + names} for k in ("price", "fund", "total")}
    for w in W:
        ok = [s for s in w["ret"] if all(s in w["sig"][k] for k in ("MOM", "LV", "FUND")) and w["surge"].get(s) is not None]
        if len(ok) < 20:
            continue
        rk = {k: ranks({s: w["sig"][k][s] for s in ok}) for k in ("MOM", "LV", "FUND")}
        rk["surge"] = ranks({s: w["surge"][s] for s in ok}); rk["age"] = ranks({s: w["age"][s] for s in ok})
        X = np.column_stack([np.ones(len(ok))] + [[rk[k][s] for s in ok] for k in names[:5]] +
                            [[1.0 if w["perponly"][s] else 0.0 for s in ok]])
        ys = {"price": [w["ret"][s][0] for s in ok], "fund": [w["ret"][s][1] for s in ok]}
        ys["total"] = [a + b for a, b in zip(ys["price"], ys["fund"])]
        for k, y in ys.items():
            y = np.clip(np.array(y), -1, 10)
            bb = np.linalg.lstsq(X, y, rcond=None)[0]
            for n_, v in zip(["const"] + names, bb):
                coefs[k][n_].append(v)
    out = []
    for k in ("price", "fund", "total"):
        out.append(f"  dep = {k:5s}: " + "  ".join(f"{n_} {pct(nw(v)[0])} (t {nw(v)[1]:+.2f})" for n_, v in coefs[k].items()
                                                  if n_ != "const"))
    return out


# ---------------------------------------------------------------- main
def main():
    data = load()
    hv = hour_volume(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
    W, stops = build(data, hv, spot, smap)
    L = [f"paper_v3 {datetime.now(timezone.utc):%Y-%m-%d}: {len(W)} weeks {fmt(W[0]['day'])}..{fmt(W[-1]['day'])}", ""]

    # R0
    L.append("R0 data rules")
    L.append(f"  trading-stop contract-weeks in the top-30 universe: {len(stops)}")
    for s, d, t1 in stops:
        L.append(f"    {s:16s} week {fmt(d)} last traded hour ends "
                 f"{datetime.fromtimestamp(t1 / 1000, timezone.utc):%Y-%m-%d %H:%M}" if t1 else f"    {s} week {fmt(d)} no trade")
    nspot = sum(1 for w in W for s in w["ret"] if not w["perponly"][s])
    nall = sum(len(w["ret"]) for w in W)
    static = sum(1 for w in W for s in w["ret"] if s in smap)
    L.append(f"  contract-weeks with spot trading at d: {nspot} of {nall} ({nspot / nall:.1%}); static map would say {static}")
    L.append(f"  spot-test weeks (>= 24 contracts with a spot return): {sum(1 for w in W if len(w['spot']) >= 24)}")
    main_rows = {k: ls(W, plain(k), 6) for k in ("MOM", "LV", "FUND")}
    L.append("  capped legs (main rule): " + ", ".join(f"{k} {r[0].get('caps', 0)}" for k, r in main_rows.items()))
    L.append("")

    # Table 3/4 under R0
    L.append("B1 Table 3 under R0 (spot vs perp, same picks, contracts with spot trading that week)")
    tf_main = {}
    for key, lab in (("LV", "LV 7d"), ("LV14", "LV 14d"), ("LV30", "LV 30d"), ("MOM", "MOM 30d"), ("FUND", "FUND")):
        s, tf = spot_line(W, key, lab, boot_ci=True); L.append(s); tf_main[key] = tf
    L.append("B1 Table 4 under R0 (full universe)")
    for k, r in main_rows.items():
        L.append(line(r, k))
    for k in main_rows:
        L.append(line(ls(W, plain(k), 6, cap=False), f"{k} no leg cap (cross-margined)"))
    L.append("")
    L.append("B2 robustness of Tables 3/4")
    W_ex = [w for w in W if w["day"] != ALPACA_WEEK]
    W1, thr1 = drop_extreme(W, 1); W5, thr5 = drop_extreme(W, 5)
    variants = [("v1.0 rule", W, "ret1"), ("ex ALPACA week", W_ex, "ret"),
                (f"ex top 1% |r| (> {thr1:.0%})", W1, "ret"), (f"ex top 5% |r| (> {thr5:.0%})", W5, "ret")]
    lv_ok = tf_main["LV"] <= -2
    for lab, Wv, rk in variants:
        L.append(f" [{lab}]")
        for key in ("LV", "MOM", "FUND"):
            s, tf = spot_line(Wv, key, f"spot test {key}", retkey=rk); L.append(s)
            if key == "LV" and (lab == "ex ALPACA week" or lab.startswith("ex top 1%")):
                lv_ok = lv_ok and tf <= -2
        for key in ("MOM", "LV", "FUND"):
            L.append(line(ls(Wv, plain(key), 6, retkey=rk), f"full {key}"))
    L.append(f"  -> rule B: LV-7d funding NW t <= -2 under R0, ex-ALPACA and ex-top-1%: {lv_ok}")
    L.append("")

    # A momentum
    L.append("A Momentum amounts and funding neutrality")
    W90, _ = build(data, hv, spot, smap, universe=universe90)
    L.append(amounts(main_rows["MOM"], "MOM 30d-volume universe"))
    m90 = ls(W90, plain("MOM"), 6)
    L.append(amounts(m90, "MOM 90d-volume universe"))
    L.append(line(m90, "MOM 90d universe"))
    base_f = abs(mean(r["fund"] for r in main_rows["MOM"]))
    for lab, pk, n in (("MOM residual on funding rank", mom_resid, 6), ("MOM 5x5 dependent sort", mom_5x5, 5)):
        r = ls(W, pk, n)
        bn, tn = nw([x["net"] for x in r]); bf = mean(x["fund"] for x in r)
        neutral = abs(bf) <= base_f / 3
        word = ("keeps a price component after funding is neutralised" if neutral and tn >= 2 else
                "remainder not distinguishable from zero" if neutral else "funding cannot be neutralised within 30 contracts")
        L.append(line(r, lab) + f"  neutral {neutral} -> {word}")
        L.append(amounts(r, lab))
    for lab, Wx in (("90d", W90),):
        r = ls(Wx, mom_resid, 6)
        L.append(line(r, f"MOM residual ({lab} universe)"))
    # spanning
    fund_fp = {r["day"]: r["fund"] for r in main_rows["FUND"]}
    rows = [r for r in main_rows["MOM"] if r["day"] in fund_fp]
    y = np.array([r["net"] for r in rows]); mk = np.array([r["mkt"] for r in rows]); ff = np.array([fund_fp[r["day"]] for r in rows])
    b0, t0 = nw_ols(y, [mk]); b1, t1 = nw_ols(y, [mk, ff])
    def adiff(i):
        X0 = np.column_stack([np.ones(len(i)), mk[i]]); X1 = np.column_stack([np.ones(len(i)), mk[i], ff[i]])
        return np.linalg.lstsq(X0, y[i], rcond=None)[0][0] - np.linalg.lstsq(X1, y[i], rcond=None)[0][0]
    lo_, hi_ = boot(adiff, len(y))
    L.append(f"  spanning on FUND's funding component: alpha (mkt only) {pct(b0[0])} (NW t {t0[0]:+.2f}); with FUND funding "
             f"{pct(b1[0])} (NW t {t1[0]:+.2f}), loading {b1[2]:+.2f} (t {t1[2]:+.2f}); alpha drop {pct(b0[0] - b1[0])} "
             f"boot5-95 [{pct(lo_)},{pct(hi_)}]")
    L.append("")

    # C
    L.append("C Table 2 with the offset ex-ante window (d-14, d-7] (rate sum, + = longs paid)")
    for key in ("MOM", "LV"):
        q7, _ = quint(W, key, "exf"); qo, exo = quint(W, key, "exo")
        L += [f"  {key} same-window  " + x for x in q7] + [f"  {key} offset       " + x for x in qo]
        if key == "LV":
            L.append(f"  -> rule C: LV Q1 offset ex-ante NW t {nw(exo[0])[1]:+.2f}; partly mechanical = {nw(exo[0])[1] > -2}")
    L.append("")

    # D
    L.append("D Overlap of legs and Fama-MacBeth (top 30, R0)")
    pairs = {}
    for w in W:
        P = {k: top_bottom(w["sig"][k], 6, w["ret"]) for k in ("MOM", "LV", "FUND")}
        if any(v is None for v in P.values()):
            continue
        A, B, C = set(P["MOM"][0]), set(P["LV"][1]), set(P["FUND"][0])
        for lab, x, z in (("MOM long & LV short", A, B), ("MOM long & FUND long", A, C), ("LV short & FUND long", B, C)):
            pairs.setdefault(lab, []).append(len(x & z) / 6)
    L.append("  " + "  ".join(f"{k} {mean(v):.0%}" for k, v in pairs.items()) + "  (share of 6 names, weekly mean)")
    L += fama_macbeth(W)
    L.append("  (slopes = return difference from lowest to highest rank; fund = funding received by a long; perponly dummy)")
    L.append("")

    # E
    L.append("E Spot-free (perp-only, weekly flag) LV short legs, strategy sign (+ = gain)")
    dp, df, po_p, po_f, sp_p, sp_f = [], [], [], [], [], []
    byday = {w["day"]: w for w in W}
    for r in main_rows["LV"]:
        w = byday[r["day"]]
        a = [(-w["ret"][s][0], -w["ret"][s][1]) for s in r["sh"] if w["perponly"][s]]
        b = [(-w["ret"][s][0], -w["ret"][s][1]) for s in r["sh"] if not w["perponly"][s]]
        po_p += [x for x, _ in a]; po_f += [y for _, y in a]; sp_p += [x for x, _ in b]; sp_f += [y for _, y in b]
        if a and b:
            dp.append(mean(x for x, _ in a) - mean(x for x, _ in b)); df.append(mean(y for _, y in a) - mean(y for _, y in b))
    L.append(f"  perp-only leg-weeks {len(po_p)} ({len(po_p) / (len(po_p) + len(sp_p)):.0%}): price {pct(mean(po_p))} "
             f"(median {pct(statistics.median(po_p))}), funding {pct(mean(po_f))}; spot-pair legs price {pct(mean(sp_p))}, "
             f"funding {pct(mean(sp_f))}")
    bdp, tdp = nw(dp); bdf, tdf = nw(df)
    L.append(f"  weekly difference perp-only minus spot-pair ({len(dp)} weeks with both): price {pct(bdp)} (NW t {tdp:+.2f}), "
             f"funding {pct(bdf)} (NW t {tdf:+.2f}) -> rule E funding concentrated in perp-only = {tdf <= -2}")
    qs, _ = quint(W, "surge", "exf")
    L += ["  volume surge quintiles (Q5 = biggest 30d/90d spike):"] + [f"    {x}" for x in qs]
    L.append("")

    # F
    L.append("F Inference: CAPM, winsorised, log")
    allr = np.array([sum(r) for w in W for r in w["ret"].values()])
    lo1, hi1 = np.percentile(allr, 1), np.percentile(allr, 99)
    def wins(w):
        out = {}
        for s, (p, f) in w["ret"].items():
            t = min(max(p + f, lo1), hi1); out[s] = (p + (t - p - f), f)
        return out
    for k, r in main_rows.items():
        rw = ls(W, plain(k), 6, retkey=wins)
        bw, tw = nw([x["net"] for x in rw]); bl, tl = nw([math.log1p(x["net"]) for x in r])
        L.append(f"  {k:4s} CAPM {capm(r)}  winsorised 1/99 net {pct(bw)} (t {tw:+.2f})  log1p net {pct(bl)} (t {tl:+.2f})")
    L.append("")

    # G
    L.append("G Robustness")
    W50, _ = build(data, hv, spot, smap, u=50)
    W100, _ = build(data, hv, spot, smap, u=100)
    po_cost = lambda w, s: 0.003 if w["perponly"].get(s) else COST
    vw = lambda w, s: max(w["vol30"].get(s, 0.0), 1.0)
    for k in ("MOM", "LV", "FUND"):
        L.append(f" [{k}]")
        L.append(line(ls(W, plain(k, 10), 10), "top30 10/10"))
        L.append(line(ls(W50, plain(k), 6), "top50 6/6"))
        L.append(line(ls(W50, plain(k, 10), 10), "top50 10/10"))
        L.append(line(ls(W100, plain(k, 10), 10), "top100 10/10"))
        L.append(line(ls(W, plain(k), 6, weight=vw), "top30 6/6 volume-weighted"))
        L.append(line(ls(W, plain(k), 6, cost_of=po_cost), "top30 6/6, perp-only 0.30%/side"))
        r = main_rows[k]
        L.append(line([x for x in r if x["day"] < SPLIT], "first half (< 2023-09-01)"))
        L.append(line([x for x in r if x["day"] >= SPLIT], "second half"))
    yrs = {}
    for w in W:
        for s in w["ret"]:
            c = data[s]
            i, j = bisect_left(c["times"], w["day"] + 1), bisect_left(c["times"], w["day"] + 7 * D + 1)
            y_ = datetime.fromtimestamp(w["day"] / 1000, timezone.utc).year
            yrs.setdefault(y_, []).append(j - i > 21)
    L.append("  share of contract-weeks with more than 21 settlements (interval < 8h): " +
             "  ".join(f"{y_} {mean(v):.0%}" for y_, v in sorted(yrs.items())))
    txt = "\n".join(L)
    print(txt)
    (OUT / "v3.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
