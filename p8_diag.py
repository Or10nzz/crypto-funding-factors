#!/usr/bin/env python3
"""paper_v8 P2b post-hoc diagnostic (registry NOTE 2026-11-21). Run once -> results/paper/v8_diag.txt"""
import json
from bisect import bisect_right
import numpy as np
import paper_v3 as v
import paper_v5 as p5
import paper_v8 as p8
from paper_ic_ls import D, ROOT, mean
from paper_funding import spot_opens

p5.install()
data = p5.load5(); hv = v.hour_volume(data)
smap = json.loads((ROOT / "results/paper/spot_map.json").read_text())
spot = {p: spot_opens(s) for p, (s, mult) in smap.items()}
W, _ = p5.build5(data, hv, spot, smap)
IV = {s: p5.interest(c) for s, c in data.items()}
exx, nxx = {}, {}
for w in W:
    d = w["day"]
    for s in w["ret"]:
        c = data[s]; t1 = p5.T1.get((s, d), d + 7 * D); p0 = c["bars"][d][0]
        i, j = bisect_right(c["times"], d), bisect_right(c["times"], t1)
        dv = sum(IV[s][k] * ((c["fpx"][k] or p0) / p0) for k in range(i, j))
        nxx[(s, d)] = -(w["ret"][s][1] + dv)
        if c["times"] and c["times"][0] <= d - 14 * D:
            exx[(s, d)] = p5.excess_funding(c, IV[s], d - 7 * D, d, False)
SQ = {p: p8.spot_qv(s) for p, (s, mult) in smap.items()}
hard = {}
for w in W:
    d = w["day"]; rat = {}
    for s in w["ret"]:
        if not w["perponly"][s] and w["vol30"].get(s, 0) > 0 and s in SQ:
            rat[s] = sum(SQ[s].get(d - i * D, 0.0) for i in range(1, 31)) / w["vol30"][s]
    cut = float(np.quantile(list(rat.values()), 1 / 3)) if len(rat) >= 3 else None
    for s in w["ret"]:
        if w["perponly"][s]:
            hard[(s, d)] = True
        elif s in rat and cut is not None:
            hard[(s, d)] = rat[s] <= cut
dex, dvol, bh, diffraw = [], [], [], []
for w in W:
    d = w["day"]
    ss = [s for s in w["ret"] if (s, d) in exx and (s, d) in hard]
    if len(ss) < 25:
        continue
    o = sorted(ss, key=lambda s: (exx[(s, d)], s))
    q1 = list(np.array_split(np.array(o, dtype=object), 5)[0])
    hh = [s for s in q1 if hard[(s, d)]]; ee = [s for s in q1 if not hard[(s, d)]]
    if hh and ee:
        dex.append(mean(exx[(s, d)] for s in hh) - mean(exx[(s, d)] for s in ee))
        vv = lambda s: -w["sig"]["LV"][s] if s in w["sig"]["LV"] else None
        a = [vv(s) for s in hh if vv(s) is not None]; b = [vv(s) for s in ee if vv(s) is not None]
        if a and b:
            dvol.append(mean(a) - mean(b))
        diffraw.append(mean(nxx[(s, d)] for s in hh) - mean(nxx[(s, d)] for s in ee))
    if len(hh) >= 2 and len(ee) >= 2:
        X = np.column_stack([np.ones(len(q1)), [exx[(s, d)] for s in q1], [1.0 if hard[(s, d)] else 0.0 for s in q1]])
        y = np.array([nxx[(s, d)] for s in q1])
        bh.append(np.linalg.lstsq(X, y, rcond=None)[0][2])
L = ["paper_v8 P2b diagnostic (post-hoc, registry NOTE 2026-11-21)",
     f"  P2b raw difference (check vs v8): {v.pct(v.nw(diffraw)[0])} (t {v.nw(diffraw)[1]:+.2f}), weeks {len(diffraw)}",
     f"  (i) HARD minus EASY ex-ante excess funding inside bottom quintile: {v.pct(v.nw(dex)[0])} (t {v.nw(dex)[1]:+.2f})",
     f"      HARD minus EASY 7-day volatility: {v.nw(dvol)[0]:+.4f} (t {v.nw(dvol)[1]:+.2f}), weeks {len(dvol)}",
     f"  (ii) FM HARD coefficient controlling for ex-ante excess funding: {v.pct(v.nw(bh)[0])} (NW4 t {v.nw(bh)[1]:+.2f}, "
     f"NW26 t {p5.nwL(bh, 26)[1]:+.2f}), weeks {len(bh)}"]
(ROOT / "results/paper/v8_diag.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
