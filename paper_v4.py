#!/usr/bin/env python3
"""Registry PRE-REGISTERED paper_v4 H (2026-11-20): funding around token unlocks (hedging demand), plus weekly series
for the figures of funding_paper v2.x. Description on seen data.
  ~/bin/heavy ~/.venvs/quant/bin/python paper_v4.py   -> results/paper/v4.txt, results/paper/v4_series.json"""
import json
from datetime import datetime, timezone

import numpy as np

import paper_v3 as v
from paper_ic_ls import D, OUT, ROOT, fmt, mean
from paper_funding import spot_opens


def events(min_size):
    e = json.loads((ROOT / "data/unlock/events.json").read_text())
    mapped = {val[2] for key, val in e["mapping"].items() if val[2] and key != "flowswap"}
    ev = {}
    for x in e["events"]:
        if x["proj"] == "flowswap" or x["size"] < min_size:
            continue
        t = int(datetime.strptime(x["day"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        ev.setdefault(x["perp"], []).append(t)
    return mapped, ev


def flags(W, mapped, ev):
    for w in W:
        d = w["day"]
        w["UP"], w["RECENT"] = {}, {}
        for s in w["ret"]:
            if s not in mapped:
                continue
            ts = ev.get(s, [])
            w["UP"][s] = any(d < t <= d + 30 * D for t in ts)
            w["RECENT"][s] = any(d - 30 * D < t <= d for t in ts)


def test(W, lab):
    L = [f" [{lab}]"]
    n_up = sum(sum(w["UP"].values()) for w in W)
    n_all = sum(len(w["UP"]) for w in W)
    L.append(f"  mapped contract-weeks {n_all}, with UP {n_up}, with RECENT {sum(sum(w['RECENT'].values()) for w in W)}")
    if n_up < 100:
        L.append("  -> insufficient sample (< 100 UP contract-weeks)")
        return L, None
    out = {}
    for yk, lab_y in (("ex", "ex-ante funding"), ("re", "realised funding")):
        slopes, diffs = [], []
        for w in W:
            ok = [s for s in w["UP"] if s in w["exf"] and all(s in w["sig"][k] for k in ("MOM", "LV"))
                  and w["surge"].get(s) is not None]
            up = [s for s in ok if w["UP"][s]]
            if len(ok) < 12 or not up or len(up) == len(ok):
                continue
            y = np.array([w["exf"][s] if yk == "ex" else -w["ret"][s][1] for s in ok])
            rk = {k: v.ranks({s: w["sig"][k][s] for s in ok}) for k in ("MOM", "LV")}
            rk["surge"] = v.ranks({s: w["surge"][s] for s in ok})
            X = np.column_stack([np.ones(len(ok)), [1.0 if w["UP"][s] else 0.0 for s in ok],
                                 [1.0 if w["RECENT"][s] else 0.0 for s in ok]] +
                                [[rk[k][s] for s in ok] for k in ("MOM", "LV", "surge")] +
                                [[1.0 if w["perponly"][s] else 0.0 for s in ok]])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            slopes.append(b[1])
            diffs.append(mean(y[i] for i, s in enumerate(ok) if w["UP"][s]) - mean(y[i] for i, s in enumerate(ok) if not w["UP"][s]))
        bs, tsl = v.nw(slopes); bd, td = v.nw(diffs)
        out[yk] = (bs, tsl)
        L.append(f"  {lab_y:17s}: FM slope on UP {v.pct(bs)} (NW t {tsl:+.2f}, {len(slopes)} weeks); raw UP minus non-UP "
                 f"{v.pct(bd)} (NW t {td:+.2f})")
    # share of most-negative ex-ante quintile with UP
    q1, q1up = 0, 0
    for w in W:
        ss = [s for s in w["UP"] if s in w["exf"]]
        if len(ss) < 10:
            continue
        o = sorted(ss, key=lambda s: (w["exf"][s], s))[: len(ss) // 5]
        q1 += len(o); q1up += sum(w["UP"][s] for s in o)
    L.append(f"  most-negative ex-ante funding quintile (within mapped contracts): {q1up}/{q1} with UP ({q1up / max(q1, 1):.0%}); "
             f"unconditional UP share {n_up / n_all:.0%}")
    bs, tsl = out["re"]
    word = ("funding is more negative ahead of large insider unlocks, consistent with hedging demand" if tsl <= -2 and out["ex"][0] < 0
            else "evidence against the hedging explanation" if tsl >= 2
            else "no evidence that unlock-related hedging explains negative funding")
    L.append(f"  -> rule H: {word}")
    return L, word


def main():
    data = v.load()
    hv = v.hour_volume(data)
    smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
    spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
    W, _ = v.build(data, hv, spot, smap)
    L = [f"paper_v4 {datetime.now(timezone.utc):%Y-%m-%d}: {len(W)} weeks", "", "H funding around insider/investor cliff unlocks"]
    for size, lab in ((0.005, "size >= 0.5% of max supply (main)"), (0.02, "size >= 2%")):
        mapped, ev = events(size)
        flags(W, mapped, ev)
        lines, _ = test(W, lab)
        L += lines
    # series for figures
    ser = {}
    for k in ("MOM", "LV", "FUND"):
        r = v.ls(W, v.plain(k), 6)
        ser[k] = dict(day=[fmt(x["day"]) for x in r], price=[x["price"] for x in r], fund=[x["fund"] for x in r],
                      net=[x["net"] for x in r])
    S, P, T = v.spot_rows(W, "LV")
    ser["LV_spot"] = dict(day=[fmt(x["day"]) for x in S], spot=[x["net"] for x in S], perp_price=[x["net"] for x in P],
                          perp_total=[x["net"] for x in T])
    (OUT / "v4_series.json").write_text(json.dumps(ser))
    txt = "\n".join(L)
    print(txt)
    (OUT / "v4.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
