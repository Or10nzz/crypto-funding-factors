#!/usr/bin/env python3
"""paper_v8 A4b: OKX funding on the same picks (registry PRE-REGISTERED paper_v8 A4b, 2026-11-21). Same method as A4
(Bybit) in paper_v8.py. Run once:  ~/bin/heavy ~/.venvs/quant/bin/python paper_v8_okx.py -> results/paper/v8_okx.txt"""
import json
from bisect import bisect_right
from datetime import datetime, timezone

import numpy as np

import paper_v3 as v
import paper_v5 as p5
import paper_v6 as p6
from paper_ic_ls import D, ROOT, mean

H = 3600_000
OUTF = ROOT / "results/paper/v8_okx.txt"
pct = v.pct


def okx_base(sym):
    b = sym[:-4] if sym.endswith("USDT") else None
    if b is None:
        return None
    for pre in ("1000000", "1000", "1M"):
        if b.startswith(pre) and len(b) > len(pre):
            return b[len(pre):]
    return b


def load_okx(syms):
    out = {}
    for s in syms:
        b = okx_base(s)
        f = ROOT / "data/okx_funding" / f"{b}.json" if b else None
        if not f or not f.exists():
            continue
        x = json.loads(f.read_text())
        if len(x) < 3:
            continue
        t = np.array([a for a, _ in x], dtype=np.int64); r = np.array([c for _, c in x])
        vals, cnt = np.unique(np.diff(t), return_counts=True)
        out[s] = (t, r, int(vals[np.argmax(cnt)]))
    return out


def mark_at(s, t):
    h = t - t % H
    for k in range(4):
        x = p5.mark_month(s, h - k * H)
        if x:
            return x[0]
    return None


def okx_leg(OK, s, d, t1, p0, excess):
    """Funding RECEIVED by a long on OKX over (d, t1], Binance-mark weighted; excess or raw; None if not covered."""
    if s not in OK or not p0:
        return None
    t, r, modal = OK[s]
    i, j = bisect_right(t, d), bisect_right(t, t1)
    if j <= i:
        return None
    tw = t[i:j]
    if not (tw[0] <= d + 24 * H and tw[-1] > t1 - 24 * H):
        return None
    if len(tw) > 1 and np.max(np.diff(tw)) > 2 * modal:
        return None
    tot = 0.0
    for k in range(i, j):
        iv = 1e-4 * ((t[k] - t[k - 1]) if k > 0 else modal) / (8 * H) if excess else 0.0
        mk = mark_at(s, int(t[k]))
        if mk is None:
            return None
        tot += (r[k] - iv) * mk / p0
    return -tot


def year(t):
    return datetime.fromtimestamp(t / 1000, timezone.utc).year


def main():
    p5.install()
    data = p5.load5(); hv = v.hour_volume(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    from paper_funding import spot_opens
    spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
    W, _ = p5.build5(data, hv, spot, smap)
    L = [f"paper_v8 A4b OKX {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: {len(W)} weeks"]
    main_rows = {k: p6.ls_floor(W, v.plain(k), 6) for k in ("MOM", "LV")}
    L.append("check vs paper_v5 main net: " + ", ".join(f"{k} {pct(mean(x['net'] for x in r))}" for k, r in main_rows.items()))
    IV = {s: p5.interest(c) for s, c in data.items()}
    bin_ex, bin_raw = {}, {}
    for w in W:
        d = w["day"]
        for s in w["ret"]:
            c = data[s]; t1 = p5.T1.get((s, d), d + 7 * D); p0 = c["bars"][d][0]
            i, j = bisect_right(c["times"], d), bisect_right(c["times"], t1)
            dv = sum(IV[s][k] * ((c["fpx"][k] or p0) / p0) for k in range(i, j))
            bin_raw[(s, d)] = w["ret"][s][1]; bin_ex[(s, d)] = w["ret"][s][1] + dv
    syms = sorted({s for w in W for s in w["ret"]})
    OK = load_okx(syms)
    L.append(f"  OKX files mapped to {len(OK)} of {len(syms)} universe symbols")
    cache = {}
    def get(s, d, ex):
        if (s, d, ex) not in cache:
            c = data[s]; t1 = p5.T1.get((s, d), d + 7 * D)
            cache[(s, d, ex)] = okx_leg(OK, s, d, t1, c["bars"][d][0], ex)
        return cache[(s, d, ex)]

    def compare(rows, side, ex):
        ok_, bn, cov, tot, days = [], [], {}, {}, []
        B = bin_ex if ex else bin_raw
        for r in rows:
            d = r["day"]; grp = r["sh"] if side == -1 else r["lo"]
            cv = [s for s in grp if get(s, d, ex) is not None]
            y = year(d); cov[y] = cov.get(y, 0) + len(cv); tot[y] = tot.get(y, 0) + len(grp)
            if not cv:
                continue
            wt = 0.5 / len(cv)
            ok_.append(sum(wt * side * get(s, d, ex) for s in cv)); bn.append(sum(wt * side * B[(s, d)] for s in cv)); days.append(d)
        return ok_, bn, cov, tot, days

    S, P, T = p6.spot_rows_x(W, "LV")
    for ex, lab in ((True, "excess"), (False, "raw")):
        a, b, cov, tot, days = compare(T, -1, ex)
        A, Bn = np.array(a), np.array(b)
        lo, hi = v.boot(lambda i: A[i].mean() / Bn[i].mean(), len(A))
        ra, rb = v.nw(a), v.nw(b)
        if ex:
            L.append(f"  LV 7-day spot-sample short legs covered {sum(cov.values())}/{sum(tot.values())} = "
                     f"{sum(cov.values()) / sum(tot.values()):.1%}; by year " + ", ".join(f"{y} {cov[y] / tot[y]:.0%}" for y in sorted(tot)))
        L.append(f"  [{lab}] short-leg funding, covered legs: OKX {pct(ra[0])} (t {ra[1]:+.2f}) vs Binance same legs "
                 f"{pct(rb[0])} (t {rb[1]:+.2f}); weeks {len(a)}; NW26 t OKX {p5.nwL(a, 26)[1]:+.2f}; ratio {A.mean() / Bn.mean():.2f} "
                 f"[{lo:.2f}, {hi:.2f}]")
        if ex:
            L.append(f"      C1 {'HOLDS' if ra[0] < 0 and ra[1] <= -2 else 'DOES NOT HOLD'} (mean < 0, t <= -2)")
            L.append(f"      C2 {'HOLDS' if 0.5 <= A.mean() / Bn.mean() <= 1.5 else 'DOES NOT HOLD'} (ratio in [0.5, 1.5])")
            by = {}
            for d, x, y_ in zip(days, a, b):
                by.setdefault(year(d), ([], []))[0].append(x); by[year(d)][1].append(y_)
            L.append("      by year (OKX / Binance, % of book per week): " +
                     "; ".join(f"{y} {pct(mean(x))} / {pct(mean(z))} ({len(x)} wks)" for y, (x, z) in sorted(by.items())))
    for lab, rows, side in (("MOM full universe, long leg", main_rows["MOM"], 1), ("LV full universe, short leg", main_rows["LV"], -1)):
        a, b, cov, tot, _ = compare(rows, side, True)
        L.append(f"  {lab}: covered {sum(cov.values())}/{sum(tot.values())} = {sum(cov.values()) / sum(tot.values()):.1%}; OKX "
                 f"{pct(v.nw(a)[0])} (t {v.nw(a)[1]:+.2f}) vs Binance {pct(v.nw(b)[0])} (t {v.nw(b)[1]:+.2f}); ratio {np.mean(a) / np.mean(b):.2f}")
    OUTF.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
