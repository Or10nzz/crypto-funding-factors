#!/usr/bin/env python3
"""Registry PRE-REGISTERED paper_v6 (2026-11-20): items A-J asked by the blind review of v2.2, on the paper_v5 main spec.
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v6.py   -> results/paper/v6.txt"""
import json, math
from bisect import bisect_right
from datetime import datetime, timezone

import numpy as np

import paper_v3 as v
import paper_v5 as p5
from kit_eval import spearman
from paper_ic_ls import D, OUT, ROOT, fmt, mean
from paper_funding import spot_opens

H = 3600_000
RNG = np.random.default_rng(17)


def ls_floor(W, pick, n, retkey="ret", parts="pf", floor=1.0, liqkey="liq", use_liq=True):
    """paper_v5.ls5 with a leg floor of -floor and a selectable liquidation dict."""
    out, held = [], {}
    for w in W:
        R = w[retkey] if isinstance(retkey, str) else retkey(w)
        pk = pick(w, R)
        if pk is None:
            continue
        lo, sh = pk
        now = {s: 1 for s in lo} | {s: -1 for s in sh}
        pr = fu = lf = sf = 0.0
        legs = []
        for s, sd in now.items():
            wt = 0.5 / (len(lo) if sd == 1 else len(sh))
            base = w["ret"].get(s)
            L = w.get(liqkey, {})
            if use_liq and parts == "pf" and base is not None and R[s] is base and (s, sd) in L:
                p, f = L[(s, sd)]
            else:
                p, f = sd * R[s][0], (0.0 if parts == "p" else sd * R[s][1])
                if floor is not None and p + f < -floor:
                    k = -floor / (p + f); p, f = p * k, f * k
            pr += wt * p; fu += wt * f
            if sd == 1:
                lf += wt * f
            else:
                sf += wt * f
            legs.append((s, sd, wt * p, wt * f))
        cst = sum(v.COST / (2 * n) for s, sd in now.items() if held.get(s) != sd)
        cst += sum(v.COST / (2 * n) for s, sd in held.items() if now.get(s) != sd)
        held = now
        out.append(dict(day=w["day"], net=pr + fu - cst, gross=pr + fu, price=pr, fund=fu, lf=lf, sf=sf, lo=lo, sh=sh, legs=legs))
    return out


def spot_rows_x(W, key, thr=24, **kw):
    def ok_w(w):
        return {s: w["ret"][s] for s in w["spot"] if s in w["ret"] and s in w["sig"][key]}
    Ws = [w for w in W if len(ok_w(w)) >= thr]
    pick = lambda w, R: v.top_bottom(w["sig"][key], 6 if len(R) >= 24 else 5, R)
    S = ls_floor(Ws, pick, 6, retkey=lambda w: {s: (w["spot"][s], 0.0) for s in ok_w(w)}, use_liq=False, floor=None)
    P = ls_floor(Ws, pick, 6, retkey=ok_w, parts="p", use_liq=False, floor=None)
    T = ls_floor(Ws, pick, 6, retkey=ok_w, **kw)
    return S, P, T


def liq_m(c, d, t1, side, m, mm):
    b = c["bars"]; p0 = b[d][0]
    days = [b[d + i * D] for i in range(7) if d + i * D < t1 and d + i * D in b]
    if not days:
        return None
    if side == -1 and max(x[3] for x in days) < (m + 0.4) / (1 + mm) * p0:
        return None
    if side == 1 and (m > 1 or min(x[4] for x in days) > 0.4 * p0):
        return None
    times, pre, fpx = c["times"], c["pre"], c["fpx"]
    i = bisect_right(times, d); cum = 0.0
    for h in range(d, t1, H):
        while i < len(times) and times[i] <= h:
            if d < times[i] <= t1:
                cum += -side * (pre[i + 1] - pre[i]) * ((fpx[i] or p0) / p0)
            i += 1
        k = p5.mark_month(c["sym"], h)
        if k is None:
            continue
        worst = k[1] if side == -1 else k[2]
        if m + side * (worst / p0 - 1) + cum <= mm * worst / p0:
            return (-m - cum, cum)
    return None


def pct(x):
    return v.pct(x)


def main():
    p5.install()
    data = p5.load5(); hv = v.hour_volume(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
    W, _ = p5.build5(data, hv, spot, smap)
    byday = {w["day"]: w for w in W}
    rows = {k: ls_floor(W, v.plain(k), 6) for k in ("MOM", "LV", "FUND")}
    L = [f"paper_v6 {datetime.now(timezone.utc):%Y-%m-%d}: {len(W)} weeks", ""]
    check = {k: mean(x["net"] for x in r) for k, r in rows.items()}
    L.append("check vs paper_v5 main net: " + ", ".join(f"{k} {pct(x)}" for k, x in check.items()))

    # A
    L += ["", "A funding / spot premium (spot sample)"]
    for key in ("LV", "LV14", "LV30"):
        S, P, T = spot_rows_x(W, key)
        s_ = np.array([r["net"] for r in S]); f_ = np.array([r["fund"] for r in T])
        lo, hi = v.boot(lambda i: -f_[i].mean() / s_[i].mean(), len(s_))
        L.append(f"  {key}: funding {pct(f_.mean())} spot {pct(s_.mean())} ratio {-f_.mean() / s_.mean():.0%} boot 5-95 [{lo:.0%}, {hi:.0%}]")

    # B
    L += ["", "B spot-sample selection"]
    spot_days = {w["day"] for w in W if len({s for s in w["spot"] if s in w["ret"] and s in w["sig"]["LV"]}) >= 24}
    lvf_all = mean(x["fund"] for x in rows["LV"])
    lvf_303 = mean(x["fund"] for x in rows["LV"] if x["day"] in spot_days)
    S, P, T = spot_rows_x(W, "LV")
    L.append(f"  LV funding part: full universe all weeks {pct(lvf_all)}; full universe on the {len(spot_days)} spot weeks "
             f"{pct(lvf_303)}; spot sample {pct(mean(r['fund'] for r in T))} -> weeks {pct(lvf_all - lvf_303)}, contracts "
             f"{pct(lvf_303 - mean(r['fund'] for r in T))}")
    for key in ("LV", "LV14", "LV30", "MOM", "FUND"):
        S, P, T = spot_rows_x(W, key, thr=20)
        s, ts_ = v.nw([r["net"] for r in S]); t, tt = v.nw([r["net"] for r in T]); f, tf = v.nw([r["fund"] for r in T])
        f26 = [r["fund"] for r in T if r["day"] >= int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)]
        L.append(f"  threshold 20, {key}: weeks {len(S)} spot {pct(s)} ({ts_:+.2f}) perp total {pct(t)} ({tt:+.2f}) funding "
                 f"{pct(f)} ({tf:+.2f}); 2026 weeks {len(f26)} funding {pct(mean(f26)) if f26 else '-'}")

    # C
    L += ["", "C margin regimes"]
    keep = {w["day"]: w["liq"] for w in W}
    for lab, m, mm in (("isolated 1x", 1, 0.0), ("isolated 1x, mm 2.5%", 1, 0.025), ("isolated 2x", 2, 0.0), ("isolated 3x", 3, 0.0)):
        for w in W:
            w["liqx"] = {}
            for s in w["ret"]:
                t1 = p5.T1.get((s, w["day"]), w["day"] + 7 * D)
                for side in (1, -1):
                    r = liq_m(data[s], w["day"], t1, side, m, mm)
                    if r is not None:
                        w["liqx"][(s, side)] = r
        parts = []
        for key in ("LV", "LV14", "LV30"):
            S, P, T = spot_rows_x(W, key, floor=m, liqkey="liqx")
            t, tt = v.nw([r["net"] for r in T]); parts.append(f"{key} spot-sample perp total {pct(t)} ({tt:+.2f})")
        for key in ("MOM", "LV", "FUND"):
            r = ls_floor(W, v.plain(key), 6, floor=m, liqkey="liqx")
            n_, tn = v.nw([x["net"] for x in r]); parts.append(f"{key} full net {pct(n_)} ({tn:+.2f})")
        nl = sum(len(w["liqx"]) for w in W)
        L.append(f"  {lab:22s} (liquidation events in universe {nl}): " + "; ".join(parts))
    parts = []
    for key in ("LV", "LV14", "LV30"):
        S, P, T = spot_rows_x(W, key, floor=None, use_liq=False)
        t, tt = v.nw([r["net"] for r in T]); parts.append(f"{key} spot-sample perp total {pct(t)} ({tt:+.2f})")
    L.append("  cross-margined           : " + "; ".join(parts))

    # D
    L += ["", "D funding component split (spot sample): default interest vs excess, long vs short leg (% of book per week)"]
    IV = {s: p5.interest(c) for s, c in data.items()}
    div = {}
    for w in W:
        d = w["day"]
        for s in w["ret"]:
            c = data[s]; t1 = p5.T1.get((s, d), d + 7 * D); p0 = c["bars"][d][0]
            i, j = bisect_right(c["times"], d), bisect_right(c["times"], t1)
            div[(s, d)] = sum(IV[s][k] * ((c["fpx"][k] or p0) / p0) for k in range(i, j))
    for key in ("LV", "LV14", "LV30", "MOM", "FUND"):
        S, P, T = spot_rows_x(W, key)
        acc = {"ld": [], "le": [], "sd": [], "se": []}
        for r in T:
            ld = le = sd_ = se = 0.0
            for s, sd, cp, cf in r["legs"]:
                wt = 0.5 / (len(r["lo"]) if sd == 1 else len(r["sh"]))
                dflt = wt * (-sd * div[(s, r["day"])])
                if sd == 1:
                    ld += dflt; le += cf - dflt
                else:
                    sd_ += dflt; se += cf - dflt
            acc["ld"].append(ld); acc["le"].append(le); acc["sd"].append(sd_); acc["se"].append(se)
        L.append(f"  {key}: long default {pct(mean(acc['ld']))} long excess {pct(mean(acc['le']))} (t {v.nw(acc['le'])[1]:+.2f}) | "
                 f"short default {pct(mean(acc['sd']))} short excess {pct(mean(acc['se']))} (t {v.nw(acc['se'])[1]:+.2f})")

    # E
    L += ["", "E concentration: share of the top 1% leg-weeks in sum |contribution|"]
    def top_share(vals):
        a = np.sort(np.abs(np.asarray(vals)))[::-1]; n1 = max(1, len(a) // 100)
        return a[:n1].sum() / a.sum()
    for key in ("MOM", "LV", "FUND"):
        r = rows[key]
        fs = top_share([cf for x in r for (_, _, _, cf) in x["legs"]]); ps = top_share([cp for x in r for (_, _, cp, _) in x["legs"]])
        prep = []
        for w in W:
            R = w["ret"]; pk = v.top_bottom(w["sig"][key], 6, R)
            if pk is None:
                continue
            ok = [s for s in R if s in w["sig"]["LV"]]
            o = sorted(ok, key=lambda s: (w["sig"]["LV"][s], s)); q = {}
            for i, grp in enumerate(np.array_split(np.array(o, dtype=object), 5)):
                for s in grp:
                    q[s] = i
            pools = {i: [s for s in R if q.get(s) == i] or list(R) for i in range(5)}
            prep.append(([(q.get(s), 1) for s in pk[0]] + [(q.get(s), -1) for s in pk[1]], pools, R))
        rf, rp = [], []
        for _ in range(1000):
            cf_, cp_ = [], []
            for need, pools, R in prep:
                for qq, sd in need:
                    pool = pools.get(qq, list(R)) if qq is not None else list(R)
                    s = pool[RNG.integers(len(pool))]
                    p, f = sd * R[s][0], sd * R[s][1]
                    if p + f < -1:
                        k = -1 / (p + f); p, f = p * k, f * k
                    cf_.append(f / 12); cp_.append(p / 12)
            rf.append(top_share(cf_)); rp.append(top_share(cp_))
        rf, rp = np.array(rf), np.array(rp)
        L.append(f"  {key}: funding top-1% share real {fs:.0%} vs random median {np.median(rf):.0%} [5-95 {np.percentile(rf, 5):.0%}-"
                 f"{np.percentile(rf, 95):.0%}], P(random >= real) {np.mean(rf >= fs):.3f}; price top-1% share real {ps:.0%} vs random "
                 f"median {np.median(rp):.0%} [{np.percentile(rp, 5):.0%}-{np.percentile(rp, 95):.0%}]")

    # F
    L += ["", "F next-week price part by quintile of ex-ante excess funding (long position, % per week)"]
    for w in W:
        w["exx"] = {}
        for s in w["ret"]:
            c = data[s]; d = w["day"]
            if c["times"] and c["times"][0] <= d - 14 * D:
                w["exx"][s] = p5.excess_funding(c, IV[s], d - 7 * D, d, False)
    qs = [[] for _ in range(5)]; diff = []
    for w in W:
        ss = list(w["exx"])
        if len(ss) < 25:
            continue
        o = sorted(ss, key=lambda s: (w["exx"][s], s))
        g = np.array_split(np.array(o, dtype=object), 5)
        m_ = [mean(min(w["ret"][s][0], 10.0) for s in grp) for grp in g]
        for q in range(5):
            qs[q].append(m_[q])
        diff.append(m_[0] - m_[4])
    L.append("  " + "  ".join(f"Q{q + 1} {pct(v.nw(qs[q])[0])} (t {v.nw(qs[q])[1]:+.2f})" for q in range(5))
             + f"  | Q1 (most negative funding) minus Q5 {pct(v.nw(diff)[0])} (t {v.nw(diff)[1]:+.2f})")

    # H
    L += ["", "H first date with a settlement interval below 8 hours (contracts ever in the top-30 universe)"]
    ever = {s for w in W for s in w["uni"]}
    first = {}
    for s in ever:
        t = data[s]["times"]
        for k in range(1, len(t)):
            if t[k] - t[k - 1] < 8 * H - 60_000:
                first[s] = t[k]; break
    by = {}
    for s, t in first.items():
        by[datetime.fromtimestamp(t / 1000, timezone.utc).year] = by.get(datetime.fromtimestamp(t / 1000, timezone.utc).year, 0) + 1
    L.append(f"  {len(first)} of {len(ever)} contracts ever have an interval below 8h; first change by year: "
             + ", ".join(f"{y} {n}" for y, n in sorted(by.items())))
    (OUT / "v6_interval_changes.json").write_text(json.dumps({s: fmt(t) for s, t in sorted(first.items())}, indent=0))

    # I
    L += ["", "I risk statistics of net returns (main) and regressor correlations"]
    for key, r in rows.items():
        x = np.array([a["net"] for a in r]); eq = np.cumprod(1 + x); dd = 1 - eq / np.maximum.accumulate(eq)
        L.append(f"  {key}: weekly SD {x.std(ddof=1):.2%}, Sharpe {x.mean() / x.std(ddof=1) * math.sqrt(52):.2f}, max drawdown {dd.max():.0%}")
    names = ["MOM", "LV", "FUND", "surge", "age"]
    acc = {(a, b): [] for i, a in enumerate(names) for b in names[i + 1:]}
    for w in W:
        ok = [s for s in w["ret"] if all(s in w["sig"][k] for k in ("MOM", "LV", "FUND")) and w["surge"].get(s) is not None]
        if len(ok) < 20:
            continue
        val = {k: [w["sig"][k][s] for s in ok] for k in ("MOM", "LV", "FUND")}
        val["surge"] = [w["surge"][s] for s in ok]; val["age"] = [w["age"][s] for s in ok]
        for a, b in acc:
            acc[(a, b)].append(spearman(val[a], val[b]))
    L.append("  mean weekly Spearman: " + "  ".join(f"{a}-{b} {mean(x):+.2f}" for (a, b), x in acc.items()))

    # J
    L += ["", "J block-bootstrap intervals of net returns (main)"]
    for key, r in rows.items():
        x = np.array([a["net"] for a in r])
        a8 = v.boot(lambda i: x[i].mean(), len(x)); a26 = p5.block_boot(x, 26)
        L.append(f"  {key}: mean {pct(x.mean())}  8-week [{pct(a8[0])}, {pct(a8[1])}]  26-week [{pct(a26[0])}, {pct(a26[1])}]")
    txt = "\n".join(L)
    print(txt)
    (OUT / "v6.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
