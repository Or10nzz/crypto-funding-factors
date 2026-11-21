#!/usr/bin/env python3
"""Registry PRE-REGISTERED paper_v7 (2026-11-20): items M1-M8 asked by the blind review of v2.3, paper_v5 main spec.
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v7.py   -> results/paper/v7.txt, results/paper/v7_series.json"""
import json, math, statistics
from bisect import bisect_right
from datetime import datetime, timezone

import numpy as np

import momentum as m
import paper_v3 as v
import paper_v5 as p5
import paper_v6 as p6
from paper_ic_ls import D, OUT, ROOT, fmt, mean, nw_ols
from paper_funding import spot_opens

H = 3600_000
year = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).year
pct = v.pct


def spot_highs(sym):
    out = {}
    for f in sorted((ROOT / "data/spot/klines" / sym).glob("*.zip")):
        for r in m.rows(f):
            if r and r[0].isdigit():
                t = int(r[0]); t = t // 1000 if t > 10 ** 14 else t
                out[t] = float(r[2])
    return out


def iid_t(x):
    x = np.asarray(x); return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def acf(x, k):
    x = np.asarray(x) - np.mean(x); return float(np.dot(x[:-k], x[k:]) / np.dot(x, x))


def ls_cost(W, pick, n, cost_of):
    """Main spec (liquidation) with a per-contract cost function."""
    out, held = [], {}
    for w in W:
        R = w["ret"]; pk = pick(w, R)
        if pk is None:
            continue
        lo, sh = pk
        now = {s: 1 for s in lo} | {s: -1 for s in sh}
        g = 0.0
        for s, sd in now.items():
            wt = 0.5 / len(lo if sd == 1 else sh)
            if (s, sd) in w["liq"]:
                p, f = w["liq"][(s, sd)]
            else:
                p, f = sd * R[s][0], sd * R[s][1]
                if p + f < -1:
                    k = -1 / (p + f); p, f = p * k, f * k
            g += wt * (p + f)
        c = sum(cost_of(w, s) / (2 * n) for s, sd in now.items() if held.get(s) != sd)
        c += sum(cost_of(w, s) / (2 * n) for s, sd in held.items() if now.get(s) != sd)
        held = now
        out.append(g - c)
    return out


def main():
    p5.install()
    data = p5.load5(); hv = v.hour_volume(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
    W, _ = p5.build5(data, hv, spot, smap)
    byday = {w["day"]: w for w in W}
    rows = {k: p6.ls_floor(W, v.plain(k), 6) for k in ("MOM", "LV", "FUND")}
    L = [f"paper_v7 {datetime.now(timezone.utc):%Y-%m-%d}: {len(W)} weeks; main check " +
         ", ".join(f"{k} {pct(mean(x['net'] for x in r))}" for k, r in rows.items()), ""]
    IV = {s: p5.interest(c) for s, c in data.items()}

    # M1
    L.append("M1 funding tails")
    for q in (1, 5):
        Wf, thr = p5.drop_funding(W, q)
        parts = []
        for key in ("LV", "LV14", "LV30"):
            S, P, T = v.spot_rows(Wf, key)
            f, tf = v.nw([r["fund"] for r in T]); parts.append(f"{key} spot-sample funding {pct(f)} ({tf:+.2f})")
        for key in ("LV", "MOM"):
            r = p6.ls_floor(Wf, v.plain(key), 6); f, tf = v.nw([x["fund"] for x in r]); parts.append(f"{key} full funding {pct(f)} ({tf:+.2f})")
        L.append(f"  without top {q}% |funding| (> {thr:.2%}): " + "; ".join(parts))
    S, P, T = v.spot_rows(W, "LV")
    series = {"LV7 spot funding": [r["fund"] for r in T], "LV funding": [x["fund"] for x in rows["LV"]],
              "MOM funding": [x["fund"] for x in rows["MOM"]]}
    for k, x in series.items():
        a = np.sort(x); n10 = len(a) // 10
        L.append(f"  {k}: mean {pct(a.mean())} median {pct(np.median(a))} 10%-trimmed mean {pct(a[n10:len(a) - n10].mean())}")
    L.append("")

    # M2
    L.append("M2 margin on spot (spot shorts liquidated at a daily high >= 2x entry)")
    sh_cache = {}
    ser_out = {}
    for key in ("LV", "LV14", "LV30"):
        S, P, T = v.spot_rows(W, key)
        Tn = p6.spot_rows_x(W, key, floor=1.0, use_liq=False)[2]
        sl, nliq = [], 0
        for r in S:
            w = byday[r["day"]]; d = w["day"]
            val = 0.0
            for s in r["sh"]:
                sym = smap[s][0]
                if sym not in sh_cache:
                    sh_cache[sym] = spot_highs(sym)
                hi = max((sh_cache[sym].get(d + i * D, 0.0) for i in range(7)), default=0.0)
                p0 = spot[s][d]
                if hi >= 2 * p0:
                    floored = max(-w["spot"][s], -1.0)               # the leg as in Table 4 (floored at -100%)
                    val += 0.5 / len(r["sh"]) * (1.0 + floored); nliq += 1   # liquidated leg returns -1 instead
            sl.append(r["net"] - val)
        s0, t0 = v.nw([r["net"] for r in S]); s1, t1 = v.nw(sl); pn, tpn = v.nw([r["net"] for r in Tn]); pt, tpt = v.nw([r["net"] for r in T])
        f, tf = v.nw([r["fund"] for r in T])
        L.append(f"  {key}: spot frictionless {pct(s0)} ({t0:+.2f}); spot with 1x margin {pct(s1)} ({t1:+.2f}) [{nliq} liquidated spot shorts]; "
                 f"perp, funding, no liquidation {pct(pn)} ({tpn:+.2f}); perp, funding, liquidation {pct(pt)} ({tpt:+.2f}); funding component {pct(f)}")
        if key == "LV":
            ser_out = dict(day=[fmt(r["day"]) for r in S], spot=[r["net"] for r in S], spot_margin=sl,
                           perp_noliq=[r["net"] for r in Tn], perp=[r["net"] for r in T])
    (OUT / "v7_series.json").write_text(json.dumps(ser_out))
    L.append("")

    # M3c
    L.append("M3c event study around the first shortened settlement interval")
    ever = {s for w in W for s in w["uni"]}
    inuni = {}
    for w in W:
        for s in w["uni"]:
            inuni.setdefault(s, set()).add(w["day"])
    ev = []
    for s in ever:
        t = data[s]["times"]
        for k in range(1, len(t)):
            if t[k] - t[k - 1] < 8 * H - 60_000:
                e = (t[k] // D) * D                         # back to the Monday 00:00 of the event week
                while ((e // D) + 3) % 7 != 0:
                    e -= D
                if any(e + j * 7 * D in inuni.get(s, ()) for j in range(-4, 5)):
                    ev.append((s, e))
                break
    rel = {j: {"exf": [], "ret": [], "nset": []} for j in range(-8, 9)}
    for s, e in ev:
        c = data[s]; b = c["bars"]
        for j in range(-8, 9):
            d0 = e + j * 7 * D; d1 = d0 + 7 * D
            i, k = bisect_right(c["times"], d0), bisect_right(c["times"], d1)
            if i == k or d0 not in b or d1 not in b or b[d0][0] <= 0:
                continue
            rel[j]["exf"].append(sum(c["pre"][q + 1] - c["pre"][q] - IV[s][q] for q in range(i, k)))
            rel[j]["ret"].append(min(b[d1][0] / b[d0][0] - 1, 10.0))
            rel[j]["nset"].append(k - i)
    L.append(f"  {len(ev)} events (contracts in the top-30 universe within 4 weeks of their first shortened interval)")
    for j in range(-8, 9):
        x = rel[j]
        if x["exf"]:
            L.append(f"  week {j:+d}: n {len(x['exf'])} excess funding {pct(mean(x['exf']))} (t {iid_t(x['exf']):+.2f}) "
                     f"median {pct(statistics.median(x['exf']))}  price {pct(mean(x['ret']))}  settlements {mean(x['nset']):.1f}")
    pre = [a for j in range(-4, 0) for a in rel[j]["exf"]]; post = [a for j in range(1, 5) for a in rel[j]["exf"]]
    L.append(f"  weeks -4..-1 mean {pct(mean(pre))}, weeks +1..+4 mean {pct(mean(post))}")
    L.append("")

    # M4
    L.append("M4 overlap: funding by whether the leg is in the other portfolio")
    a_in, a_out, b_in, b_out = [], [], [], []
    momd = {x["day"]: x for x in rows["MOM"]}; lvd = {x["day"]: x for x in rows["LV"]}
    for d in momd.keys() & lvd.keys():
        mlong = set(momd[d]["lo"]); lshort = set(lvd[d]["sh"])
        a_in.append(sum(cf for s, sd, cp, cf in lvd[d]["legs"] if sd == -1 and s in mlong))
        a_out.append(sum(cf for s, sd, cp, cf in lvd[d]["legs"] if sd == -1 and s not in mlong))
        b_in.append(sum(cf for s, sd, cp, cf in momd[d]["legs"] if sd == 1 and s in lshort))
        b_out.append(sum(cf for s, sd, cp, cf in momd[d]["legs"] if sd == 1 and s not in lshort))
    L.append(f"  LV short-leg funding (% of book/wk): from legs also in MOM long {pct(mean(a_in))} (t {v.nw(a_in)[1]:+.2f}), "
             f"other legs {pct(mean(a_out))} (t {v.nw(a_out)[1]:+.2f})")
    L.append(f"  MOM long-leg funding: from legs also in LV short {pct(mean(b_in))} (t {v.nw(b_in)[1]:+.2f}), "
             f"other legs {pct(mean(b_out))} (t {v.nw(b_out)[1]:+.2f})")
    L.append("")

    # M5
    L.append("M5 next-week price by quintile of ex-ante funding (long position, unlevered unless stated)")
    for w in W:
        w["exx"] = {}; w["exr"] = {}
        for s in w["ret"]:
            c = data[s]; d = w["day"]
            if c["times"] and c["times"][0] <= d - 14 * D:
                w["exx"][s] = p5.excess_funding(c, IV[s], d - 7 * D, d, False)
                w["exr"][s] = w["exf"].get(s)
    def quint(sortkey, val):
        qs = [[] for _ in range(5)]; days = []
        for w in W:
            ss = [s for s in w[sortkey] if w[sortkey][s] is not None]
            if len(ss) < 25:
                continue
            o = sorted(ss, key=lambda s: (w[sortkey][s], s))
            g = np.array_split(np.array(o, dtype=object), 5)
            for q in range(5):
                qs[q].append(mean(val(w, s) for s in g[q]))
            days.append(w["day"])
        return qs, days
    price_c = lambda w, s: min(w["ret"][s][0], 10.0)
    def price_liq(w, s):
        if (s, 1) in w["liq"]:
            return w["liq"][(s, 1)][0]
        return min(w["ret"][s][0], 10.0)
    total_c = lambda w, s: min(w["ret"][s][0] + w["ret"][s][1], 10.0)
    for lab, sk, fn in (("excess sort, price capped +1000%", "exx", price_c), ("excess sort, price with long liquidation", "exx", price_liq),
                        ("raw rate-sum sort, price capped", "exr", price_c), ("excess sort, total (price + funding) capped", "exx", total_c)):
        qs, days = quint(sk, fn)
        dif = np.array(qs[0]) - np.array(qs[4])
        lo, hi = v.boot(lambda i: dif[i].mean(), len(dif)); b, t = v.nw(dif)
        se = abs(b / t) if t else float("nan")
        L.append(f"  {lab}: Q1 {pct(mean(qs[0]))} Q5 {pct(mean(qs[4]))} Q1-Q5 {pct(b)} (NW t {t:+.2f}) boot [{pct(lo)}, {pct(hi)}] MDE {pct(2.8 * se)}")
        if lab.startswith("excess sort, price capped"):
            yrs = {}
            for d_, x in zip(days, dif):
                yrs.setdefault(year(d_), []).append(x)
            L.append("    by year Q1-Q5: " + "  ".join(f"{y} {pct(mean(x))}" for y, x in sorted(yrs.items())))
    fx = p6.ls_floor(W, v.plain("FUND"), 6, floor=None, use_liq=False)
    L.append(f"  FUND price part: main (liquidation) {pct(mean(x['price'] for x in rows['FUND']))} [long {pct(mean(sum(cp for s, sd, cp, cf in x['legs'] if sd == 1) for x in rows['FUND']))}, "
             f"short {pct(mean(sum(cp for s, sd, cp, cf in x['legs'] if sd == -1) for x in rows['FUND']))}]; cross-margined {pct(mean(x['price'] for x in fx))}")
    L.append("")

    # M6
    L.append("M6 inference")
    sers = {"MOM net": [x["net"] for x in rows["MOM"]], "LV net": [x["net"] for x in rows["LV"]], "FUND net": [x["net"] for x in rows["FUND"]],
            "LV7 spot funding": [r["fund"] for r in T], "MOM funding": [x["fund"] for x in rows["MOM"]]}
    for k, x in sers.items():
        L.append(f"  {k:17s} mean {pct(mean(x))} iid t {iid_t(x):+.2f} NW4 {p5.nwL(x, 4)[1]:+.2f} NW26 {p5.nwL(x, 26)[1]:+.2f}  acf1-4 "
                 + " ".join(f"{acf(x, j):+.2f}" for j in (1, 2, 3, 4)))
    L.append("")

    # M7
    L.append("M7 liquidity-tiered costs and 5% maintenance margin")
    def tier(w, s):
        if s not in w["vol30"]:
            return 0.004                                   # leaving the universe: charged the highest tier
        vs = sorted(w["vol30"].values(), reverse=True); r = sorted(w["vol30"], key=lambda z: -w["vol30"][z]).index(s) / len(vs)
        return 0.001 if r < 1 / 3 else 0.002 if r < 2 / 3 else 0.004
    for key in ("MOM", "LV", "FUND"):
        x = ls_cost(W, v.plain(key), 6, tier); b, t = v.nw(x)
        L.append(f"  {key} net with 0.10/0.20/0.40% per side by volume tercile: {pct(b)} (t {t:+.2f})")
    for w in W:
        w["liqx"] = {}
        for s in w["ret"]:
            t1 = p5.T1.get((s, w["day"]), w["day"] + 7 * D)
            for side in (1, -1):
                r = p6.liq_m(data[s], w["day"], t1, side, 1, 0.05)
                if r is not None:
                    w["liqx"][(s, side)] = r
    parts = []
    for key in ("LV", "LV14", "LV30"):
        Tm = p6.spot_rows_x(W, key, floor=1, liqkey="liqx")[2]; b, t = v.nw([r["net"] for r in Tm]); parts.append(f"{key} spot-sample perp total {pct(b)} ({t:+.2f})")
    for key in ("MOM", "LV", "FUND"):
        r = p6.ls_floor(W, v.plain(key), 6, floor=1, liqkey="liqx"); b, t = v.nw([x["net"] for x in r]); parts.append(f"{key} net {pct(b)} ({t:+.2f})")
    L.append("  isolated 1x, maintenance 5%: " + "; ".join(parts))
    L.append("")

    # M8
    L.append("M8 Fama-MacBeth with reversal, MAX, Amihud, size")
    names = ["MOM", "LV", "FUND", "surge", "age", "rev1", "max", "amihud", "size", "perponly"]
    co = {k: {n: [] for n in names} for k in ("price", "fund", "total")}
    for w in W:
        d = w["day"]; prev = d - D; X = {}
        for s in w["ret"]:
            b = data[s]["bars"]
            if not all(s in w["sig"][k] for k in ("MOM", "LV", "FUND")) or w["surge"].get(s) is None:
                continue
            closes = [b.get(prev - i * D) for i in range(31)]
            if any(x is None or x[1] <= 0 for x in closes) or prev - 7 * D not in b:
                continue
            rets = [closes[i][1] / closes[i + 1][1] - 1 for i in range(30)]
            ami = mean(abs(rets[i]) / max(closes[i][2], 1.0) for i in range(30))
            X[s] = dict(rev1=b[prev][1] / b[prev - 7 * D][1] - 1, max=max(rets), amihud=ami, size=w["vol30"][s])
        ok = list(X)
        if len(ok) < 20:
            continue
        rk = {k: v.ranks({s: w["sig"][k][s] for s in ok}) for k in ("MOM", "LV", "FUND")}
        rk["surge"] = v.ranks({s: w["surge"][s] for s in ok}); rk["age"] = v.ranks({s: w["age"][s] for s in ok})
        for k in ("rev1", "max", "amihud", "size"):
            rk[k] = v.ranks({s: X[s][k] for s in ok})
        M = np.column_stack([np.ones(len(ok))] + [[rk[k][s] for s in ok] for k in names[:-1]] + [[1.0 if w["perponly"][s] else 0.0 for s in ok]])
        ys = {"price": np.clip([w["ret"][s][0] for s in ok], -1, 10), "fund": np.array([w["ret"][s][1] for s in ok])}
        ys["total"] = np.clip(np.array([w["ret"][s][0] + w["ret"][s][1] for s in ok]), -1, 10)
        for k, y in ys.items():
            bb = np.linalg.lstsq(M, y, rcond=None)[0]
            for n_, val in zip(names, bb[1:]):
                co[k][n_].append(val)
    for k in ("price", "fund", "total"):
        L.append(f"  {k:5s}: " + "  ".join(f"{n_} {pct(v.nw(x)[0])} ({v.nw(x)[1]:+.2f})" for n_, x in co[k].items()))
    txt = "\n".join(L)
    print(txt)
    (OUT / "v7.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
