#!/usr/bin/env python3
"""Item F of registry PRE-REGISTERED paper_v2: funding weighted by the 1-hour MARK price at each settlement instead of
the last-price kline open; MOM / LV decomposition under both, and the ALPACA 2025-04-28 appendix. Description.
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v2_mark.py -> results/paper/v2_mark.txt"""
from bisect import bisect_right
from datetime import datetime, timezone

import momentum as m
import paper_ic_ls as P
from paper_ic_ls import D, OUT, ROOT, analyse, fmt, load, mean, weeks

H = 3600_000


def attach_mark(data):
    for sym, c in data.items():
        hours = {}
        for f in sorted((ROOT / "data/universe/markPriceKlines" / sym / "1h").glob("*.zip")):
            for r in m.rows(f):
                if r and r[0].isdigit():
                    t = int(r[0]); hours[t // 1000 if t > 10 ** 14 else t] = float(r[1])
        c["mpx"] = [hours.get(t - t % H) for t in (c.get("times") or [])]


def use(field):
    """Point path_funding at the last-price ('fpx') or mark-price ('mpx') settlement prices."""
    for c in DATA.values():
        c["fpx"] = c["_last"] if field == "fpx" else [mp if mp else lp for mp, lp in zip(c["mpx"], c["_last"])]


def main():
    global DATA
    DATA = data = load()
    attach_mark(data)
    for c in data.values():
        c["_last"] = list(c.get("fpx") or [])
    W_by = {}
    L = [f"paper_v2_mark {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    for field, lab in (("fpx", "last-price kline open"), ("mpx", "mark-price kline open")):
        use(field)
        W = weeks(data)
        for key in ("MOM", "LV"):
            R = analyse(data, W, key)
            L.append(f"  {lab:22s} {key}: LS net {mean(r['net'] for r in R) * 100:+.3f}%  long funding {mean(r['lf'] for r in R) * 100:+.3f}%  "
                     f"short funding {mean(r['sf'] for r in R) * 100:+.3f}%")
    cov = sum(1 for c in data.values() for x in c["mpx"] if x) / max(1, sum(len(c["mpx"]) for c in data.values()))
    L.append(f"  settlements with a mark price: {cov:.1%}")
    c = data["ALPACAUSDT"]; b = c["bars"]
    d = int(datetime(2025, 4, 28, tzinfo=timezone.utc).timestamp() * 1000); p0 = b[d][0]
    i, j = bisect_right(c["times"], d), bisect_right(c["times"], d + 7 * D)
    L += ["", f"Appendix: ALPACA week of 2025-04-28 (entry open {p0}); daily totals of settlements (rate sum; weighted by last / mark price)"]
    acc = {}
    for k in range(i, j):
        t = c["times"][k]; r = c["pre"][k + 1] - c["pre"][k]
        day = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%m-%d")
        a = acc.setdefault(day, [0, 0.0, 0.0, 0.0, 0.0])
        lp, mp = c["_last"][k], c["mpx"][k]
        a[0] += 1; a[1] += r; a[2] += r * (lp / p0 if lp else 1); a[3] += r * (mp / p0 if mp else 1); a[4] = max(a[4], abs(r))
    tot = [0.0, 0.0, 0.0]
    for day, (n, r, wl, wm, mx) in acc.items():
        L.append(f"  {day}  settlements {n:2d}  rate sum {r * 100:+7.2f}%  last-weighted {wl * 100:+7.2f}%  mark-weighted {wm * 100:+7.2f}%  max |rate| {mx * 100:.2f}%")
        tot[0] += r; tot[1] += wl; tot[2] += wm
    L.append(f"  week: rate sum {tot[0] * 100:+.2f}%  last-weighted {tot[1] * 100:+.2f}%  mark-weighted {tot[2] * 100:+.2f}%  (+ = paid by longs, received by shorts)")
    bars = [(datetime.fromtimestamp(t / 1000, timezone.utc).strftime('%m-%d'), b[t]) for t in range(d, d + 8 * D, D) if t in b]
    L.append("  daily bars (open, close, volume): " + "; ".join(f"{dd} {x[0]:.4f}/{x[1]:.4f}/{x[2]:.0f}" for dd, x in bars))
    txt = "\n".join(L)
    print(txt)
    (OUT / "v2_mark.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
