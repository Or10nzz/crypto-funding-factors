#!/usr/bin/env python3
"""Registry PRE-REGISTERED paper_v5 (2026-11-20): new main specification (mark-price funding, mark-price exit at
trading stops, isolated margin with hourly liquidation) and the checks asked by the blind review of v2.1.
Reruns paper_v3 / paper_v3_extra / paper_v4 under the new main spec into results/paper/v5main/, then the new items.
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v5.py   -> results/paper/v5main/*.txt, results/paper/v5.txt"""
import json, math, statistics, sys, zipfile, io, csv
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone

import numpy as np

import paper_v3 as v
import paper_v3_extra as vx
import paper_v4 as v4
from paper_ic_ls import D, OUT, ROOT, RAW_FUNDING, fmt, mean, nw_ols, path_funding
from paper_funding import spot_opens
from paper_v2_mark import attach_mark

H = 3600_000
OUT5 = OUT / "v5main"
MARK = ROOT / "data/universe/markPriceKlines"
_orig_load, _orig_cw, _orig_build = v.load, v.coin_week, v.build
T1 = {}                                   # (sym, d) -> exit time used for the position opened at d
_mcache = {}


# ---------------------------------------------------------------- main specification
def mark_month(sym, t):
    key = (sym, datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m"))
    if key not in _mcache:
        if len(_mcache) > 300:
            _mcache.clear()
        rows = {}
        f = MARK / sym / "1h" / f"{sym}-1h-{key[1]}.zip"
        if f.exists():
            with zipfile.ZipFile(f) as z:
                for r in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))):
                    if r and r[0].isdigit():
                        tt = int(r[0]); tt = tt // 1000 if tt > 10 ** 14 else tt
                        rows[tt] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]))
        _mcache[key] = rows
    return _mcache[key].get(t)


def load5():
    data = _orig_load()
    attach_mark(data)
    for s, c in data.items():
        c["sym"] = s
        c["fpx_last"] = c["fpx"]
        c["fpx"] = [mp if mp else lp for mp, lp in zip(c["mpx"], c["fpx"])]
    return data


def coin_week5(c, hv, d):
    x = _orig_cw(c, hv, d)
    if x is None:
        return None
    main, v1, stopped, t1 = x
    T1[(c["sym"], d)] = t1 if stopped and t1 else d + 7 * D
    if stopped and t1 and t1 > d:
        k = mark_month(c["sym"], t1 - H)
        if k and c["bars"][d][0] > 0:
            main = (k[3] / c["bars"][d][0] - 1, main[1])
    return main, v1, stopped, t1


def liquidate(c, d, t1, side):
    """Side-signed (price, funding) if the isolated 1x leg is liquidated (equity <= 0 on the hourly mark range)."""
    b = c["bars"]
    p0 = b[d][0]
    days = [b[d + i * D] for i in range(7) if d + i * D < t1 and d + i * D in b]
    if not days:
        return None
    if side == -1 and max(x[3] for x in days) < 1.8 * p0:
        return None
    if side == 1 and min(x[4] for x in days) > 0.2 * p0:
        return None
    times, pre, fpx = c["times"], c["pre"], c["fpx"]
    i = bisect_right(times, d)
    cum = 0.0
    for h in range(d, t1, H):
        while i < len(times) and times[i] <= h:
            if d < times[i] <= t1:
                cum += -side * (pre[i + 1] - pre[i]) * ((fpx[i] or p0) / p0)
            i += 1
        k = mark_month(c["sym"], h)
        if k is None:
            continue
        worst = k[1] if side == -1 else k[2]
        if 1 + side * (worst / p0 - 1) + cum <= 0:
            return (-1 - cum, cum)
    return None


def build5(data, hv, spot, smap, universe=None, u=30, weekday=0):
    W, stops = _orig_build(data, hv, spot, smap, **({"universe": universe} if universe else {}), u=u, weekday=weekday)
    for w in W:
        w["liq"] = {}
        for s in w["ret"]:
            t1 = T1.get((s, w["day"]), w["day"] + 7 * D)
            for side in (1, -1):
                r = liquidate(data[s], w["day"], t1, side)
                if r is not None:
                    w["liq"][(s, side)] = r
    return W, stops


def ls5(W, pick, n, retkey="ret", parts="pf", weight=None, cost_of=None, cap=True):
    out, held, caps, liqs = [], {}, 0, 0
    for w in W:
        R = w[retkey] if isinstance(retkey, str) else retkey(w)
        pk = pick(w, R)
        if pk is None:
            continue
        lo, sh = pk
        now = {s: 1 for s in lo} | {s: -1 for s in sh}
        wt = {}
        for grp in (lo, sh):
            tot = sum(weight(w, s) for s in grp) if weight else None
            for s in grp:
                wt[s] = 0.5 * (weight(w, s) / tot if weight else 1 / len(grp))
        pr = fu = lp = sp_ = lf = sf = 0.0
        legs = []
        for s, sd in now.items():
            base = w["ret"].get(s)
            if cap and parts == "pf" and base is not None and R[s] is base and (s, sd) in w.get("liq", {}):
                p, f = w["liq"][(s, sd)]; liqs += 1
            else:
                p, f = sd * R[s][0], sd * R[s][1]
                if parts == "p":
                    f = 0.0
                if cap and p + f < -1:
                    k = -1 / (p + f); p, f = p * k, f * k; caps += 1
            pr += wt[s] * p; fu += wt[s] * f
            if sd == 1:
                lp += wt[s] * p; lf += wt[s] * f
            else:
                sp_ += wt[s] * p; sf += wt[s] * f
            legs.append((s, sd, wt[s] * p, wt[s] * f))
        cst = sum((1 / (2 * n)) * (cost_of(w, s) if cost_of else v.COST) for s, sd in now.items() if held.get(s) != sd)
        cst += sum((1 / (2 * n)) * (cost_of(w, s) if cost_of else v.COST) for s, sd in held.items() if now.get(s) != sd)
        held = now
        g = pr + fu
        out.append(dict(day=w["day"], gross=g, net=g - cst, price=pr, fund=fu, lp=lp, sp=sp_, lf=lf, sf=sf, lo=lo, sh=sh,
                        legs=legs, mkt=mean(sum(R[s]) for s in R)))
    if out:
        out[0]["caps"] = caps; out[0]["liqs"] = liqs
    return out


def spot_line5(W, key, lab, retkey="ret", boot_ci=False):
    S, P, T = v.spot_rows(W, key, retkey)
    s, ts_ = v.nw([r["net"] for r in S]); p, tp = v.nw([r["net"] for r in P]); t, tt = v.nw([r["net"] for r in T])
    F = np.array([r["fund"] for r in T])
    f, tf = v.nw(F)
    extra = f"  [long-leg funding {v.pct(mean(r['lf'] for r in T))}, short-leg {v.pct(mean(r['sf'] for r in T))}]"
    if boot_ci:
        Sa, Ta = np.array([r["net"] for r in S]), np.array([r["net"] for r in T])
        a, b_ = v.boot(lambda i: Sa[i].mean(), len(Sa)); c_, d_ = v.boot(lambda i: Ta[i].mean(), len(Ta))
        e, f_ = v.boot(lambda i: F[i].mean(), len(F))
        extra += f"  boot5-95 spot [{v.pct(a)},{v.pct(b_)}] perp total [{v.pct(c_)},{v.pct(d_)}] funding [{v.pct(e)},{v.pct(f_)}]"
    return (f"  {lab:34s} wks {len(S)} spot {v.pct(s)} (NW t {ts_:+.2f})  perp price {v.pct(p)} ({tp:+.2f})  perp total "
            f"{v.pct(t)} ({tt:+.2f})  funding {v.pct(f)} (NW t {tf:+.2f}){extra}"), tf


def install():
    v.load, v.coin_week, v.build, v.ls, v.spot_line = load5, coin_week5, build5, ls5, spot_line5
    vx.load = load5
    for mod in (v, vx, v4):
        mod.OUT = OUT5


# ---------------------------------------------------------------- new items
def nwL(x, L):
    b, t = nw_ols(list(x), [], lags=L)
    return b[0], t[0]


def block_boot(x, block=26, reps=2000, rng=np.random.default_rng(11)):
    x = np.asarray(x); n = len(x)
    m_ = []
    for _ in range(reps):
        st = rng.integers(0, n - block + 1, size=n // block + 1)
        m_.append(x[np.concatenate([np.arange(a, a + block) for a in st])[:n]].mean())
    return np.percentile(m_, 5), np.percentile(m_, 95)


def acf(x, k):
    x = np.asarray(x) - np.mean(x)
    return float(np.dot(x[:-k], x[k:]) / np.dot(x, x))


def year(t):
    return datetime.fromtimestamp(t / 1000, timezone.utc).year


def price_part_simple(W, picks_of, R_of):
    vals = []
    for w in W:
        pk = picks_of(w)
        if pk is None:
            continue
        lo, sh = pk; R = R_of(w)
        vals.append(0.5 * mean(R[s][0] for s in lo) + 0.5 * mean(max(-R[s][0], -1.0) for s in sh))
    return mean(vals)


def random_benchmark(W, Wt, key, reps=1000, rng=np.random.default_rng(5)):
    """Drop in the price part from trimming: real factor vs volatility-quintile-matched random portfolios."""
    def quint(w, R):
        ok = [s for s in R if s in w["sig"]["LV"]]
        o = sorted(ok, key=lambda s: (w["sig"]["LV"][s], s))
        q = {}
        for i, grp in enumerate(np.array_split(np.array(o, dtype=object), 5)):
            for s in grp:
                q[s] = i
        return q
    real_full = price_part_simple(W, lambda w: v.top_bottom(w["sig"][key], 6, w["ret"]), lambda w: w["ret"])
    real_trim = price_part_simple(Wt, lambda w: v.top_bottom(w["sig"][key], 6, w["ret"]), lambda w: w["ret"])
    prep = []
    for w, wt in zip(W, Wt):
        out = []
        for ww in (w, wt):
            R = ww["ret"]; pk = v.top_bottom(ww["sig"][key], 6, R)
            if pk is None:
                out.append(None); continue
            q = quint(ww, R)
            pools = {i: [s for s in R if q.get(s) == i] for i in range(5)}
            need = [(q.get(s), 1) for s in pk[0]] + [(q.get(s), -1) for s in pk[1]]
            out.append((need, pools, R))
        prep.append(out)
    drops = []
    for _ in range(reps):
        acc = [[], []]
        for out in prep:
            for j, item in enumerate(out):
                if item is None:
                    continue
                need, pools, R = item
                lo, sh = [], []
                for qq, sd in need:
                    pool = pools.get(qq) if qq is not None else list(R)
                    if not pool:
                        pool = list(R)
                    s = pool[rng.integers(len(pool))]
                    (lo if sd == 1 else sh).append(s)
                acc[j].append(0.5 * mean(R[s][0] for s in lo) + 0.5 * mean(max(-R[s][0], -1.0) for s in sh))
        drops.append(mean(acc[0]) - mean(acc[1]))
    drops = np.array(drops)
    real = real_full - real_trim
    return real, drops


def drop_funding(W, q):
    tot = np.array([abs(r[1]) for w in W for r in w["ret"].values()])
    thr = np.percentile(tot, 100 - q)
    out = []
    for w in W:
        x = dict(w)
        x["ret"] = {s: r for s, r in w["ret"].items() if abs(r[1]) <= thr}
        x["spot"] = {s: r for s, r in w["spot"].items() if s in x["ret"]}
        out.append(x)
    return out, thr


def interest(c):
    t = c["times"]
    return [1e-4 * ((t[k] - t[k - 1]) if k else 8 * H) / (8 * H) for k in range(len(t))]


def excess_funding(c, iv, t0, t1, weighted, p0=None):
    i, j = bisect_right(c["times"], t0), bisect_right(c["times"], t1)
    tot = 0.0
    for k in range(i, j):
        r = c["pre"][k + 1] - c["pre"][k] - iv[k]
        tot += r * ((c["fpx"][k] or p0) / p0 if weighted else 1.0)
    return tot


def items(L):
    data = load5()
    hv = v.hour_volume(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
    W, _ = build5(data, hv, spot, smap)
    rows = {k: ls5(W, v.plain(k), 6) for k in ("MOM", "LV", "FUND")}
    L += [f"paper_v5 items {datetime.now(timezone.utc):%Y-%m-%d}: {len(W)} weeks; liquidated legs (main): " +
          ", ".join(f"{k} {r[0]['liqs']}" for k, r in rows.items()) + "; capped (non-liquidation) legs: " +
          ", ".join(f"{k} {r[0]['caps']}" for k, r in rows.items()), ""]
    byday = {w["day"]: w for w in W}

    # F1
    L.append("F1 tails")
    allr = np.array([sum(r) for w in W for r in w["ret"].values()])
    lo1, hi1 = np.percentile(allr, 1), np.percentile(allr, 99)
    def wins(w):
        return {s: (p + (min(max(p + f, lo1), hi1) - p - f), f) for s, (p, f) in w["ret"].items()}
    for k in ("MOM", "LV", "FUND"):
        L.append(v.line(ls5(W, v.plain(k), 6, retkey=wins), f"{k} winsorised 1/99"))
    for q in (1, 5):
        Wt, thr = v.drop_extreme(W, q)
        for k in ("MOM", "LV"):
            real, drops = random_benchmark(W, Wt, k)
            L.append(f"  trim top {q}% |total| (> {thr:.0%}): {k} price-part drop real {v.pct(real)}; vol-matched random "
                     f"mean {v.pct(drops.mean())}, 95th pct {v.pct(np.percentile(drops, 95))}, share of random >= real "
                     f"{np.mean(drops >= real):.3f}")
    for q in (1, 5):
        Wf, thr = drop_funding(W, q)
        for k in ("MOM", "LV", "FUND"):
            L.append(v.line(ls5(Wf, v.plain(k), 6), f"{k} without top {q}% |funding| (> {thr:.2%})"))
    L.append("")

    # F2
    L.append("F2 time and concentration")
    for k, r in rows.items():
        yrs = {}
        for x in r:
            yrs.setdefault(year(x["day"]), []).append(x)
        L.append(f"  {k} by year: " + "  ".join(
            f"{y_} price {v.pct(mean(a['price'] for a in xs))} fund {v.pct(mean(a['fund'] for a in xs))} "
            f"(t {v.nw([a['fund'] for a in xs])[1]:+.2f})" for y_, xs in sorted(yrs.items())))
    cut = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    W25 = [w for w in W if w["day"] < cut]
    s, tf = spot_line5(W25, "LV", "LV 7d spot test, without 2026"); L.append(s)
    L.append(v.amounts(ls5(W25, v.plain("MOM"), 6), "MOM without 2026"))
    for k, r in rows.items():
        L.append(v.line([x for x in r if x["day"] < cut], f"{k} without 2026"))
    for k, r in rows.items():
        tot = sum(x["fund"] for x in r)
        per = {}
        lw = []
        for x in r:
            for s, sd, cp, cf in x["legs"]:
                per[s] = per.get(s, 0.0) + cf; lw.append(cf)
        sgn = 1 if tot >= 0 else -1
        top = sorted(per.values(), key=lambda z: -sgn * z)
        lw = np.array(lw); n1 = max(1, len(lw) // 100)
        big = np.sort(sgn * lw)[::-1][:n1]
        L.append(f"  {k} funding concentration: total {tot * 100:+.1f} (sum over weeks, % of book); top 5 contracts "
                 f"{sum(top[:5]) / tot:.0%}, top 10 {sum(top[:10]) / tot:.0%}; top 1% leg-weeks {sgn * big.sum() / tot:.0%}")
        g_short, g_long = 0.0, 0.0
        for x in r:
            w = byday[x["day"]]
            for s, sd, cp, cf in x["legs"]:
                c = data[s]; t1 = T1.get((s, w["day"]), w["day"] + 7 * D)
                nset = bisect_right(c["times"], t1) - bisect_right(c["times"], w["day"])
                if nset > 21:
                    g_short += cf
                else:
                    g_long += cf
        L.append(f"  {k} funding part per week from leg-weeks with > 21 settlements {v.pct(g_short / len(r))}, "
                 f"with <= 21 {v.pct(g_long / len(r))}")
    L.append("")

    # M5
    L.append("M5 funding net of the default interest component (0.01% per 8h, prorated)")
    IV = {s: interest(c) for s, c in data.items()}
    near = tot_s = 0
    for w in W:
        w["exx"], w["rex"] = {}, {}
        for s in w["ret"]:
            c = data[s]; d = w["day"]
            if c["times"] and c["times"][0] <= d - 14 * D:
                w["exx"][s] = excess_funding(c, IV[s], d - 7 * D, d, False)
                t1 = T1.get((s, d), d + 7 * D)
                w["rex"][s] = excess_funding(c, IV[s], d, t1, True, c["bars"][d][0])
                i, j = bisect_right(c["times"], d), bisect_right(c["times"], t1)
                for k_ in range(i, j):
                    tot_s += 1
                    near += abs(c["pre"][k_ + 1] - c["pre"][k_] - IV[s][k_]) <= 5e-6
    L.append(f"  settlements within 0.0005% of the default rate: {near / tot_s:.0%} of {tot_s}")
    for key in ("MOM", "LV"):
        for q in range(5):
            ex, re_ = [], []
            for w in W:
                ss = [s for s in w["exx"] if s in w["sig"][key]]
                if len(ss) < 25:
                    continue
                o = sorted(ss, key=lambda s: (w["sig"][key][s], s))
                grp = np.array_split(np.array(o, dtype=object), 5)[q]
                ex.append(mean(w["exx"][s] for s in grp)); re_.append(mean(w["rex"][s] for s in grp))
            be, te = v.nw(ex); br, tr = v.nw(re_)
            L.append(f"  {key} Q{q + 1} excess ex-ante {v.pct(be)} (t {te:+.2f}) excess next week {v.pct(br)} (t {tr:+.2f})")
    sl = []
    for w in W:
        ss = list(w["exx"])
        x = np.array([w["exx"][s] for s in ss]); y = np.array([w["rex"][s] for s in ss])
        if len(ss) >= 12 and x.std() > 0:
            sl.append(np.polyfit(x, y, 1)[0])
    L.append(f"  persistence slope on excess funding {mean(sl):+.3f} (NW t {v.nw(sl)[1]:+.2f})")
    L.append("")

    # M6
    L.append("M6 momentum: full universe vs spot sample (price part, % per week)")
    spot_days = {w["day"] for w in W if len(w["spot"]) >= 24}
    A = mean(x["price"] for x in rows["MOM"])
    B = mean(x["price"] for x in rows["MOM"] if x["day"] in spot_days)
    Cv = []
    for x in rows["MOM"]:
        if x["day"] not in spot_days:
            continue
        w = byday[x["day"]]
        lo = [s for s in x["lo"] if not w["perponly"][s]]; sh = [s for s in x["sh"] if not w["perponly"][s]]
        if lo and sh:
            Cv.append(0.5 * mean(w["ret"][s][0] for s in lo) + 0.5 * mean(max(-w["ret"][s][0], -1) for s in sh))
    S, P, T = v.spot_rows(W, "MOM")
    po = sum(1 for x in rows["MOM"] for s, sd, a, b in x["legs"] if byday[x["day"]]["perponly"][s])
    L.append(f"  (A) full 317 wks {v.pct(A)}  (B) full, 303 spot weeks {v.pct(B)}  (C) full picks without perp-only legs "
             f"{v.pct(mean(Cv))}  (D) spot-eligible re-sorted, perp price {v.pct(mean(r['net'] for r in P))}; perp-only MOM "
             f"leg-weeks {po} of {sum(len(x['legs']) for x in rows['MOM'])}; dropped weeks: "
             + ", ".join(fmt(d) for d in sorted({w['day'] for w in W} - spot_days)))
    Sa = np.array([r["net"] for r in S]); lo_, hi_ = v.boot(lambda i: Sa[i].mean(), len(Sa))
    L.append(f"  MOM spot {v.pct(Sa.mean())} boot 5-95 [{v.pct(lo_)},{v.pct(hi_)}]")
    L.append("")

    # M7
    L.append("M7 neutralisation in the top-100 universe (10/10)")
    W100, _ = build5(data, hv, spot, smap, u=100)
    def resid_n(n):
        def f(w, R):
            ok = [s for s in w["sig"]["MOM"] if s in w["exf"] and s in R]
            if len(ok) < 4 * n:
                return None
            rm, rf = v.ranks({s: w["sig"]["MOM"][s] for s in ok}), v.ranks({s: w["exf"][s] for s in ok})
            x = np.array([rf[s] for s in ok]); y = np.array([rm[s] for s in ok]); bb = np.polyfit(x, y, 1)
            return v.top_bottom({s: y[i] - (bb[0] * x[i] + bb[1]) for i, s in enumerate(ok)}, n, R)
        return f
    def dep(k):
        def f(w, R):
            ok = [s for s in w["sig"]["MOM"] if s in w["exf"] and s in R]
            if len(ok) < 10 * k:
                return None
            o = sorted(ok, key=lambda s: (w["exf"][s], s)); lo, sh = [], []
            for grp in np.array_split(np.array(o, dtype=object), 5):
                vv = sorted(grp, key=lambda s: (w["sig"]["MOM"][s], s)); lo += vv[-k:]; sh += vv[:k]
            return lo, sh
        return f
    for lab, pk in (("MOM top100 plain 10/10", v.plain("MOM", 10)), ("MOM top100 residual 10/10", resid_n(10)),
                    ("MOM top100 dependent 5x(2/2)", dep(2))):
        r = ls5(W100, pk, 10)
        L.append(v.line(r, lab)); L.append(v.amounts(r, lab))
    L.append("")

    # M8
    L.append("M8 serial correlation")
    S, P, T = v.spot_rows(W, "LV")
    ser = {"LV7 spot funding": [r["fund"] for r in T], "MOM funding": [x["fund"] for x in rows["MOM"]],
           "MOM net": [x["net"] for x in rows["MOM"]], "LV funding": [x["fund"] for x in rows["LV"]],
           "FUND funding": [x["fund"] for x in rows["FUND"]]}
    for k, x in ser.items():
        tl = "  ".join(f"NW{L_} {nwL(x, L_)[1]:+.2f}" for L_ in (4, 13, 26))
        a, b = block_boot(x)
        L.append(f"  {k:17s} mean {v.pct(mean(x))}  {tl}  26-wk block boot [{v.pct(a)},{v.pct(b)}]  acf "
                 + " ".join(f"{acf(x, k_):+.2f}" for k_ in (1, 4, 13, 26)))
    L.append("")

    # M11
    L.append("M11 plain sum of rates vs price-weighted funding parts")
    def raw_ret(w):
        out = {}
        for s, (p, f) in w["ret"].items():
            t1 = T1.get((s, w["day"]), w["day"] + 7 * D)
            out[s] = (p, -RAW_FUNDING(data[s], w["day"], t1))
        return out
    for k in ("MOM", "LV", "FUND"):
        r = ls5(W, v.plain(k), 6, retkey=raw_ret)
        L.append(f"  {k}: funding part plain sum {v.pct(mean(x['fund'] for x in r))} vs mark-weighted "
                 f"{v.pct(mean(x['fund'] for x in rows[k]))}")
    L.append("")

    # M12
    L.append("M12 one-hour execution delay (legs floored at -100%)")
    for w in W:
        d = w["day"]; nxt = d + 7 * D; w["ret_d1"] = {}
        for s, base in w["ret"].items():
            T_, C_ = hv.get(s, (None, None))
            if T_ is None or len(T_) == 0:
                w["ret_d1"][s] = base; continue
            i = np.searchsorted(T_, d); j = np.searchsorted(T_, nxt)
            stopped = T1.get((s, d), nxt) != nxt
            if i < len(T_) and T_[i] == d and not stopped and j < len(T_) and T_[j] == nxt:
                c = data[s]; a, b2 = bisect_right(c["times"], d + H), bisect_right(c["times"], nxt + H)
                fpaid = sum((c["pre"][k_ + 1] - c["pre"][k_]) * ((c["fpx"][k_] or C_[i]) / C_[i]) for k_ in range(a, b2))
                w["ret_d1"][s] = (C_[j] / C_[i] - 1, -fpaid)
            else:
                w["ret_d1"][s] = base
    for k in ("MOM", "LV", "FUND"):
        L.append(v.line(ls5(W, v.plain(k), 6, retkey="ret_d1"), f"{k} delayed 1h"))
    return L


def main():
    install()
    OUT5.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["v3", "extra", "v4", "items"]
    if "v3" in which:
        v.main()
    if "extra" in which:
        vx.main()
    if "v4" in which:
        v4.main()
    if "items" in which:
        L = items([])
        txt = "\n".join(L)
        print(txt)
        (OUT / "v5.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
