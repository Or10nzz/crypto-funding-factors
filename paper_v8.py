#!/usr/bin/env python3
"""paper_v8: revision-pack tests A3-A6 for funding_paper v2.5 (registry PRE-REGISTERED paper_v8 2026-11-21 and the
CLARIFICATION written before computation). Run once:
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v8.py   -> results/paper/v8.txt
Cross-sectional tests use the RATE convention (+ = paid by longs); portfolio quantities are funding received by
the book, as in Table 4."""
import csv, io, json, math, os, zipfile
from bisect import bisect_right
from datetime import datetime, timezone

import numpy as np

import paper_v3 as v
import paper_v5 as p5
import paper_v6 as p6
from paper_ic_ls import D, ROOT, mean
from paper_funding import spot_opens

H = 3600_000
OUTF = ROOT / "results/paper/v8.txt"
A3_START = int(datetime(2021, 12, 1, tzinfo=timezone.utc).timestamp() * 1000)
A3_END = int(datetime(2026, 10, 8, tzinfo=timezone.utc).timestamp() * 1000)
Y2019 = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
pct = v.pct


def year(t):
    return datetime.fromtimestamp(t / 1000, timezone.utc).year


def tstr(bt):
    b, t = bt
    return f"{pct(b)} (t {t:+.2f})"


def fm(weeks, names):
    """weeks: list of (y, {name: regressor list}); returns {name: (mean slope, NW4 t)}, number of weeks."""
    co = {n: [] for n in names}
    for y, X in weeks:
        A = np.column_stack([np.ones(len(y))] + [np.array(X[n], float) for n in names])
        b = np.linalg.lstsq(A, np.array(y, float), rcond=None)[0]
        for n, bb in zip(names, b[1:]):
            co[n].append(bb)
    return {n: v.nw(c) for n, c in co.items()}, {n: p5.nwL(c, 26) for n, c in co.items()}, len(weeks)


def fm_line(res, names):
    r4, r26, n = res
    return f"weeks {n}: " + "; ".join(f"{k} {r4[k][0]:+.4f} (NW4 t {r4[k][1]:+.2f}, NW26 t {r26[k][1]:+.2f})" for k in names)


# ---------------------------------------------------------------- data helpers
def load_metrics(syms, need):
    out = {}
    for s in syms:
        f = ROOT / "data/hist/metrics" / f"{s}.csv"
        if not f.exists():
            continue
        d = {}
        with open(f) as fh:
            for line in fh:
                p = line.rstrip("\n").split(",")
                if len(p) < 3 or not p[0].isdigit():
                    continue
                t = int(p[0])
                if t in need:
                    try:
                        d[t] = (float(p[1]), float(p[2]))
                    except ValueError:
                        pass
        out[s] = d
    return out


def position(M, s, d):
    m = M.get(s)
    if not m:
        return None
    a, b = m.get(d - H), m.get(d - 169 * H)
    if not a or not b or a[0] <= 0 or b[0] <= 0:
        return None
    g1 = [m[d - k * H][1] for k in range(1, 25) if d - k * H in m]
    g0 = [m[d - 7 * D - k * H][1] for k in range(1, 25) if d - 7 * D - k * H in m]
    if len(g1) < 20 or len(g0) < 20:
        return None
    G1, G0 = mean(g1), mean(g0)
    if G1 <= 0 or G0 <= 0:
        return None
    return dict(dOI=math.log(a[0] / b[0]), GLS=G1, dGLS=math.log(G1 / G0))


def spot_qv(sym):
    out = {}
    for f in sorted((ROOT / "data/spot/klines" / sym).glob("*.zip")):
        with zipfile.ZipFile(f) as z:
            for r in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))):
                if r and r[0].isdigit():
                    t = int(r[0]); t = t // 1000 if t > 10 ** 14 else t
                    out[t] = float(r[7])
    return out


def load_bybit(syms):
    out = {}
    for s in syms:
        f = ROOT / "data/bybit_funding" / f"{s}.json"
        if not f.exists():
            continue
        x = sorted((int(t), float(r)) for t, r in json.load(open(f)) if int(t) >= Y2019)
        if len(x) < 3:
            continue
        t = np.array([a for a, _ in x], dtype=np.int64); r = np.array([b for _, b in x])
        gaps = np.diff(t)
        vals, cnt = np.unique(gaps, return_counts=True)
        out[s] = (t, r, int(vals[np.argmax(cnt)]))
    return out


def mark_at(s, t):
    h = t - t % H
    for k in range(4):
        x = p5.mark_month(s, h - k * H)
        if x:
            return x[0]
    return None


def bybit_leg(BY, s, d, t1, p0):
    """Excess funding RECEIVED by a long on Bybit over (d, t1], Binance-mark weighted; None if not covered."""
    if s not in BY or not p0:
        return None
    t, r, modal = BY[s]
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
        iv = 1e-4 * ((t[k] - t[k - 1]) if k > 0 else modal) / (8 * H)
        mk = mark_at(s, int(t[k]))
        if mk is None:
            return None
        tot += (r[k] - iv) * mk / p0
    return -tot


def ar_cost(c, d):
    """(half-spread, sigma, mean daily quote volume) from the 30 daily bars ending d-1; None if not enough data."""
    b = c["bars"]
    days = [d - i * D for i in range(30, 0, -1) if d - i * D in b]
    if len(days) < 20:
        return None
    V = mean(b[x][2] for x in days)
    if V <= 0:
        return None
    lc = [math.log(b[x][1]) for x in days]
    eta = [(math.log(b[x][3]) + math.log(b[x][4])) / 2 for x in days]
    s2 = 4 * mean((lc[k] - eta[k]) * (lc[k] - eta[k + 1]) for k in range(len(days) - 1))
    hs = math.sqrt(max(s2, 0.0)) / 2
    rets = [b[days[k]][1] / b[days[k - 1]][1] - 1 for k in range(1, len(days))]
    return hs, float(np.std(rets, ddof=1)), V


def main():
    p5.install()
    data = p5.load5(); hv = v.hour_volume(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
    W, _ = p5.build5(data, hv, spot, smap)
    L = [f"paper_v8 {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: {len(W)} weeks", ""]
    main_rows = {k: p6.ls_floor(W, v.plain(k), 6) for k in ("MOM", "LV", "FUND")}
    L.append("check vs paper_v5 main net: " + ", ".join(f"{k} {pct(mean(x['net'] for x in r))}" for k, r in main_rows.items()))

    IV = {s: p5.interest(c) for s, c in data.items()}
    div, exx, nxx = {}, {}, {}
    for w in W:
        d = w["day"]
        for s in w["ret"]:
            c = data[s]; t1 = p5.T1.get((s, d), d + 7 * D); p0 = c["bars"][d][0]
            i, j = bisect_right(c["times"], d), bisect_right(c["times"], t1)
            div[(s, d)] = sum(IV[s][k] * ((c["fpx"][k] or p0) / p0) for k in range(i, j))
            nxx[(s, d)] = -(w["ret"][s][1] + div[(s, d)])
            if c["times"] and c["times"][0] <= d - 14 * D:
                exx[(s, d)] = p5.excess_funding(c, IV[s], d - 7 * D, d, False)

    # ================================================================ A3
    L += ["", "A3 POSITIONING (rate convention: + = paid by longs)"]
    A3W = [w for w in W if w["day"] - 8 * D >= A3_START and w["day"] - H <= A3_END]
    need = set()
    for w in A3W:
        d = w["day"]
        for k in range(1, 25):
            need.add(d - k * H); need.add(d - 7 * D - k * H)
        need.add(d - 169 * H)
    syms = sorted({s for w in A3W for s in w["ret"]})
    M = load_metrics(syms, need)
    POS = {}
    cov = {}
    for w in A3W:
        d = w["day"]
        for s in w["ret"]:
            x = position(M, s, d)
            y_ = year(d); a, b = cov.get(y_, (0, 0)); cov[y_] = (a + (x is not None), b + 1)
            if x is not None:
                POS[(s, d)] = x
    tot_a, tot_b = sum(a for a, b in cov.values()), sum(b for a, b in cov.values())
    L.append(f"  A3 weeks {len(A3W)} ({v.fmt(A3W[0]['day'])}..{v.fmt(A3W[-1]['day'])}); metrics files {len(M)} of {len(syms)} symbols; "
             f"coverage {tot_a}/{tot_b} = {tot_a / tot_b:.1%}; by year " +
             ", ".join(f"{y_} {a / b:.0%}" for y_, (a, b) in sorted(cov.items())))

    def a3_rows(w, dep, extra_exx):
        d = w["day"]
        ok = [s for s in w["ret"] if (s, d) in POS and (s, d) in exx and s in w["sig"]["LV"] and s in w["sig"]["MOM"]]
        if len(ok) < 20:
            return None
        rk = lambda f: v.ranks({s: f(s) for s in ok})
        X = {"GLS": rk(lambda s: POS[(s, d)]["GLS"]), "dGLS": rk(lambda s: POS[(s, d)]["dGLS"]),
             "dOI": rk(lambda s: POS[(s, d)]["dOI"]), "vol7": rk(lambda s: -w["sig"]["LV"][s]),
             "mom30": rk(lambda s: w["sig"]["MOM"][s]), "volume30": rk(lambda s: w["vol30"][s])}
        if extra_exx:
            X["exante"] = rk(lambda s: exx[(s, d)])
        y = [dep(s, d) for s in ok]
        return y, {k: [X[k][s] for s in ok] for k in X}

    n1 = ["GLS", "dGLS", "dOI", "vol7", "mom30", "volume30"]
    r1 = fm([x for x in (a3_rows(w, lambda s, d: exx[(s, d)], False) for w in A3W) if x], n1)
    L.append("  P1  dep = ex-ante excess funding (7d before D): " + fm_line(r1, n1))
    L.append(f"      P1 {'HOLDS' if r1[0]['GLS'][0] > 0 and r1[0]['GLS'][1] >= 2 else 'DOES NOT HOLD'} (GLS slope > 0, t >= 2)")
    n1b = n1 + ["exante"]
    r1b = fm([x for x in (a3_rows(w, lambda s, d: nxx[(s, d)], True) for w in A3W) if x], n1b)
    L.append("  P1b dep = next-week excess funding (mark-weighted): " + fm_line(r1b, n1b))
    L.append(f"      P1b {'HOLDS' if r1b[0]['GLS'][0] > 0 and r1b[0]['GLS'][1] >= 2 else 'DOES NOT HOLD'} (GLS slope > 0, t >= 2)")

    dif3, nc3 = [], []
    for w in A3W:
        d = w["day"]
        ok = [s for s in w["ret"] if (s, d) in POS and s in w["sig"]["LV"]]
        top = sorted(ok, key=lambda s: (w["sig"]["LV"][s], s))[:10]
        if len(top) < 10:
            continue
        mo = float(np.median([POS[(s, d)]["dOI"] for s in top])); mg = float(np.median([POS[(s, d)]["GLS"] for s in top]))
        cr = [s for s in top if POS[(s, d)]["dOI"] > mo and POS[(s, d)]["GLS"] < mg]
        ot = [s for s in top if s not in cr]
        if len(cr) >= 2 and len(ot) >= 2:
            dif3.append(mean(nxx[(s, d)] for s in cr) - mean(nxx[(s, d)] for s in ot)); nc3.append(len(cr))
    b3 = v.nw(dif3)
    L.append(f"  P3  10 most volatile, CROWDED (dOI > median and GLS < median) minus others, next-week excess funding: "
             f"weeks {len(dif3)}, mean crowded count {mean(nc3):.2f}, {tstr(b3)}, NW26 t {p5.nwL(dif3, 26)[1]:+.2f}")
    L.append(f"      P3 {'HOLDS' if b3[0] < 0 and b3[1] <= -2 else 'DOES NOT HOLD'} (mean < 0, t <= -2)")

    # ================================================================ P2
    L += ["", "P2 ARBITRAGE CONSTRAINT (rate convention)"]
    SQ = {p: spot_qv(s) for p, (s, mult) in smap.items()}
    hard = {}
    for w in W:
        d = w["day"]
        rat = {}
        for s in w["ret"]:
            if not w["perponly"][s] and w["vol30"].get(s, 0) > 0 and s in SQ:
                q = SQ[s]
                sv = sum(q.get(d - i * D, 0.0) for i in range(1, 31))
                rat[s] = sv / w["vol30"][s]
        cut = float(np.quantile(list(rat.values()), 1 / 3)) if len(rat) >= 3 else None
        for s in w["ret"]:
            if w["perponly"][s]:
                hard[(s, d)] = True
            elif s in rat and cut is not None:
                hard[(s, d)] = rat[s] <= cut
    nh = sum(hard.values()); L.append(f"  HARD contract-weeks {nh} of {len(hard)} ({nh / len(hard):.1%}); perp-only "
                                      f"{sum(1 for w in W for s in w['ret'] if w['perponly'][s])}")
    dif2a, sn, sp_ = [], [], []
    for w in W:
        d = w["day"]
        neg = [(exx[(s, d)], nxx[(s, d)]) for s in w["ret"] if (s, d) in exx and exx[(s, d)] < 0]
        pos = [(exx[(s, d)], nxx[(s, d)]) for s in w["ret"] if (s, d) in exx and exx[(s, d)] > 0]
        if len(neg) >= 5 and len(pos) >= 5:
            sl = lambda pts: np.polyfit([a for a, _ in pts], [b for _, b in pts], 1)[0]
            a, b = sl(neg), sl(pos); sn.append(a); sp_.append(b); dif2a.append(a - b)
    b2a = v.nw(dif2a)
    L.append(f"  P2a persistence slope: negative side {v.nw(sn)[0]:+.3f} (t {v.nw(sn)[1]:+.2f}), positive side {v.nw(sp_)[0]:+.3f} "
             f"(t {v.nw(sp_)[1]:+.2f}); difference weeks {len(dif2a)} mean {b2a[0]:+.3f} (NW4 t {b2a[1]:+.2f}, NW26 t {p5.nwL(dif2a, 26)[1]:+.2f})")
    L.append(f"      P2a {'HOLDS' if b2a[0] > 0 and b2a[1] >= 2 else 'DOES NOT HOLD'} (difference > 0, t >= 2)")
    dif2b, nhb = [], []
    for w in W:
        d = w["day"]
        ss = [s for s in w["ret"] if (s, d) in exx and (s, d) in hard]
        if len(ss) < 25:
            continue
        o = sorted(ss, key=lambda s: (exx[(s, d)], s))
        q1 = list(np.array_split(np.array(o, dtype=object), 5)[0])
        hh = [s for s in q1 if hard[(s, d)]]; ee = [s for s in q1 if not hard[(s, d)]]
        if hh and ee:
            dif2b.append(mean(nxx[(s, d)] for s in hh) - mean(nxx[(s, d)] for s in ee)); nhb.append(len(hh))
    b2b = v.nw(dif2b)
    L.append(f"  P2b bottom ex-ante quintile, HARD minus EASY next-week excess funding: weeks {len(dif2b)}, mean HARD count "
             f"{mean(nhb):.2f}, {tstr(b2b)}, NW26 t {p5.nwL(dif2b, 26)[1]:+.2f}")
    L.append(f"      P2b {'HOLDS' if b2b[0] < 0 and b2b[1] <= -2 else 'DOES NOT HOLD'} (mean < 0, t <= -2)")

    # ================================================================ A4
    L += ["", "A4 BYBIT (funding received by the book; full-week legs, no liquidation truncation)"]
    BY = load_bybit(sorted({s for w in W for s in w["ret"]}))
    byleg = {}
    def get_by(s, d):
        if (s, d) not in byleg:
            c = data[s]; t1 = p5.T1.get((s, d), d + 7 * D)
            byleg[(s, d)] = bybit_leg(BY, s, d, t1, c["bars"][d][0])
        return byleg[(s, d)]
    binleg = lambda s, d: -nxx[(s, d)]        # excess received by a long on Binance, full week

    def compare(rows, side):
        bb, bn, cov_, tot_ = [], [], {}, {}
        days = []
        for r in rows:
            d = r["day"]; grp = r["sh"] if side == -1 else r["lo"]
            cv = [s for s in grp if get_by(s, d) is not None]
            y_ = year(d); cov_[y_] = cov_.get(y_, 0) + len(cv); tot_[y_] = tot_.get(y_, 0) + len(grp)
            if not cv:
                continue
            wt = 0.5 / len(cv)
            bb.append(sum(wt * side * get_by(s, d) for s in cv)); bn.append(sum(wt * side * binleg(s, d) for s in cv)); days.append(d)
        return bb, bn, cov_, tot_, days

    S, P, T = p6.spot_rows_x(W, "LV")
    bb, bn, cv_, tt_, days = compare(T, -1)
    cvs = sum(cv_.values()) / sum(tt_.values())
    L.append(f"  BYBIT files {len(BY)}; LV 7-day spot-sample short legs covered {sum(cv_.values())}/{sum(tt_.values())} = {cvs:.1%}; by year "
             + ", ".join(f"{y_} {cv_[y_] / tt_[y_]:.0%}" for y_ in sorted(tt_)))
    B1 = v.nw(bb); BN = v.nw(bn)
    BBa, BNa = np.array(bb), np.array(bn)
    lo_, hi_ = v.boot(lambda i: BBa[i].mean() / BNa[i].mean(), len(BBa))
    ratio = BBa.mean() / BNa.mean()
    L.append(f"  B1  LV 7-day spot sample, short-leg excess funding, covered legs: Bybit {tstr(B1)} vs Binance same legs {tstr(BN)}; "
             f"weeks {len(bb)}; NW26 t Bybit {p5.nwL(bb, 26)[1]:+.2f}")
    L.append(f"      B1 {'HOLDS' if B1[0] < 0 and B1[1] <= -2 else 'DOES NOT HOLD'} (mean < 0, t <= -2)")
    L.append(f"  B2  Bybit / Binance ratio {ratio:.2f} (8-week block bootstrap 5-95 [{lo_:.2f}, {hi_:.2f}])")
    L.append(f"      B2 {'HOLDS' if 0.5 <= ratio <= 1.5 else 'DOES NOT HOLD'} (ratio in [0.5, 1.5])")
    by_y = {}
    for d, a, b in zip(days, bb, bn):
        by_y.setdefault(year(d), ([], []))[0].append(a); by_y[year(d)][1].append(b)
    L.append("      by year (Bybit / Binance, % of book per week): " +
             "; ".join(f"{y_} {pct(mean(a))} / {pct(mean(b))} ({len(a)} wks)" for y_, (a, b) in sorted(by_y.items())))
    for lab, rows, side in (("MOM full universe, long leg", main_rows["MOM"], 1), ("LV full universe, short leg", main_rows["LV"], -1)):
        bb2, bn2, cv2, tt2, _ = compare(rows, side)
        L.append(f"  {lab}: covered {sum(cv2.values())}/{sum(tt2.values())} = {sum(cv2.values()) / sum(tt2.values()):.1%}; Bybit "
                 f"{tstr(v.nw(bb2))} vs Binance {tstr(v.nw(bn2))}; ratio {np.mean(bb2) / np.mean(bn2):.2f}")

    # ================================================================ A5
    L += ["", "A5 TAILS AND 2026 (funding received by the book)"]
    F = [r["fund"] for r in T]; Fd = [r["day"] for r in T]
    L.append(f"  main LV 7-day spot-sample funding component {tstr(v.nw(F))} over {len(F)} weeks")
    for y_ in range(2020, 2027):
        x = [f for f, d in zip(F, Fd) if year(d) != y_]
        L.append(f"    without {y_}: {tstr(v.nw(x))} ({len(F) - len(x)} weeks removed)")
    srt = sorted(F)
    L.append(f"  10 most negative weeks: sum {pct(sum(srt[:10]))} of total {pct(sum(F))} = {sum(srt[:10]) / sum(F):.0%}")

    excl = {}
    for w in W:
        d = w["day"]; vals = [exx[(s, d)] for s in w["ret"] if (s, d) in exx]
        cut = float(np.quantile(vals, 0.1)) if len(vals) >= 10 else None
        excl[d] = {s for s in w["ret"] if cut is not None and (s, d) in exx and exx[(s, d)] <= cut}

    def pick_f(w, R):
        sig = w["sig"]["LV"]
        cand = sorted((sig[s], s) for s in R if s in sig)
        if len(cand) < 12:
            return None
        lo = [s for _, s in cand[-6:]]
        sh = [s for _, s in cand if s not in excl[w["day"]] and s not in lo][:6]
        return (lo, sh) if len(sh) == 6 else None

    def ok_w(w):
        return {s: w["ret"][s] for s in w["spot"] if s in w["ret"] and s in w["sig"]["LV"]}
    Ws = [w for w in W if len(ok_w(w)) >= 24]
    Tf = p6.ls_floor(Ws, pick_f, 6, retkey=ok_w)
    ff = v.nw([r["fund"] for r in Tf]); f0 = v.nw(F)
    L.append(f"  ex-ante filter (drop bottom-decile ex-ante excess funding from LV short candidates): mean excluded per week "
             f"{mean(len(x) for x in excl.values()):.1f}")
    L.append(f"  F1  spot-sample funding component: filtered {tstr(ff)} ({len(Tf)} weeks) vs main {tstr(f0)}; "
             f"magnitude ratio {abs(ff[0]) / abs(f0[0]):.2f}")
    L.append(f"      F1 {'HOLDS' if abs(ff[0]) <= 0.5 * abs(f0[0]) else 'DOES NOT HOLD'} (magnitude falls by at least half)")
    L.append(f"      filtered spot-sample net {tstr(v.nw([r['net'] for r in Tf]))} vs main {tstr(v.nw([r['net'] for r in T]))}")
    Rf = p6.ls_floor(W, pick_f, 6); Rm = main_rows["LV"]
    mf = {r["day"]: r["net"] for r in Rf}; mm = {r["day"]: r["net"] for r in Rm}
    common = sorted(set(mf) & set(mm)); dd = [mf[d] - mm[d] for d in common]
    bd = v.nw(dd)
    L.append(f"  F2  full universe LV net: filtered {tstr(v.nw([r['net'] for r in Rf]))} vs main {tstr(v.nw([r['net'] for r in Rm]))}; "
             f"difference {tstr(bd)} over {len(dd)} weeks; filtered funding part {tstr(v.nw([r['fund'] for r in Rf]))}")
    L.append(f"      F2 {'HOLDS' if v.nw([r['net'] for r in Rf])[0] > v.nw([r['net'] for r in Rm])[0] and bd[1] >= 2 else 'DOES NOT HOLD'}"
             f" (filtered net higher, difference t >= 2)")
    Rfx = p6.ls_floor(W, pick_f, 6, floor=None, use_liq=False); Rmx = p6.ls_floor(W, v.plain("LV"), 6, floor=None, use_liq=False)
    L.append(f"      cross-margined full universe net: filtered {tstr(v.nw([r['net'] for r in Rfx]))} vs main {tstr(v.nw([r['net'] for r in Rmx]))}")
    def turnover(rows):
        ch, n_ = 0, 0
        for a, b in zip(rows, rows[1:]):
            pa = set(a["lo"]) | {"-" + s for s in a["sh"]}; pb = set(b["lo"]) | {"-" + s for s in b["sh"]}
            ch += len(pb - pa); n_ += len(pb)
        return ch / n_ if n_ else float("nan")
    L.append(f"      turnover (share of legs changed): filtered {turnover(Rf):.0%} vs main {turnover(Rm):.0%}")

    # ================================================================ A6
    L += ["", "A6 COSTS AND CAPACITY (net % per week of the book; fee 0.05% + Abdi-Ranaldo half-spread + Y*sigma*sqrt(Q/V))"]
    CC = {}
    def cc(s, d):
        if (s, d) not in CC:
            CC[(s, d)] = ar_cost(data[s], d)
        return CC[(s, d)]
    fall = {"n": 0}
    def costf(aum, Y):
        q = aum / 12
        def f(w, s):
            x = cc(s, w["day"])
            if x is None:
                fall["n"] += 1
                return v.COST
            hs, sg, V = x
            return 0.0005 + hs + Y * sg * math.sqrt(q / V)
        return f
    zero = [0.0005 + cc(s, w["day"])[0] for w in W for s in w["ret"] if cc(s, w["day"]) is not None]
    L.append(f"  per-side cost at AUM -> 0 (fee + half-spread), all universe contract-weeks: mean {pct(mean(zero))}, median "
             f"{pct(float(np.median(zero)))}, 90th pct {pct(float(np.percentile(zero, 90)))}; main assumption {pct(v.COST)}")
    grid = [1e4, 1e5, 1e6, 1e7, 1e8]
    for Y in (0.5, 1.0):
        for k in ("MOM", "LV", "FUND"):
            nets = []
            for aum in grid:
                fall["n"] = 0
                r = p5.ls5(W, v.plain(k), 6, cost_of=costf(aum, Y))
                nets.append((v.nw([x["net"] for x in r]), fall["n"]))
            be = "<1e4" if nets[0][0][0] <= 0 else ">1e8"
            for (a, na), (b, nb), g0, g1 in zip(nets, nets[1:], grid, grid[1:]):
                if a[0] > 0 >= b[0]:
                    x = math.log10(g0) + (a[0] / (a[0] - b[0])) * (math.log10(g1) - math.log10(g0))
                    be = f"{10 ** x:,.0f}"
                    break
            L.append(f"  Y={Y} {k:4s}: " + "  ".join(f"{g:.0e} {pct(n[0])} (t {n[1]:+.2f})" for g, (n, _) in zip(grid, nets)) +
                     f"  | break-even AUM {be} USD | fallback legs {nets[0][1]}")

    OUTF.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
