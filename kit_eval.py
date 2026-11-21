#!/usr/bin/env python3
"""kitcrypto evaluation (pre-registered 2026-11-17, results/registry.txt): KiT predictions from the user's laptop
(data/kit_output/predictions.csv) vs realized Binance returns; momentum.run long/short; N6V2 on the same weeks.
Run with ~/bin/heavy."""
import csv, math, statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import holdout_momentum as hm   # END = 2026-11-01
import momentum as m
from spikefilter import data, picks, sharpe

ROOT = Path(__file__).parent
DAY = m.DAY
ts = lambda s: int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    return statistics.correlation(rank(x), rank(y))


def tstat(x):
    return statistics.fmean(x) / statistics.stdev(x) * math.sqrt(len(x))


def main():
    pred = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "data/kit_output/predictions.csv")):
        pred[r["monday"]][r["symbol"]] = (float(r["pred7"]), float(r["pred20"]), float(r["vol7"]))
    days = sorted(ts(d) for d in pred)
    ic7, ic20, icv, unis, P, PB = [], [], [], [], [], []
    for d in days:
        p = pred[datetime.fromtimestamp(d / 1000, timezone.utc).strftime("%Y-%m-%d")]
        uni = [s for s in m.universe(data, d, 30) if s in p]
        real, vol, ok = [], [], []
        for s in uni:
            b = data[s]["bars"]
            if d in b and d + 7 * DAY in b:
                real.append(b[d + 7 * DAY][0] / b[d][0] - 1)
                rr = [b[d + i * DAY][1] / b[d + (i - 1) * DAY][1] - 1 for i in range(1, 7) if d + i * DAY in b and d + (i - 1) * DAY in b]
                vol.append(statistics.stdev(rr) if len(rr) > 2 else 0.0)
                ok.append(s)
        if len(ok) < 12:
            continue
        ic7.append(spearman([p[s][0] for s in ok], real))
        ic20.append(spearman([p[s][1] for s in ok], real))
        icv.append(spearman([p[s][2] for s in ok], vol))
        srt = sorted(ok, key=lambda s: p[s][0])
        unis.append((d, m.universe(data, d, 30)))
        P.append((srt[-6:], srt[:6]))
        PB.append(picks(d, m.universe(data, d, 30), "V2"))
    dd = [d for d, _ in unis]
    uu = [u for _, u in unis]
    rk = [x["ret"] for x in m.run(data, dd, uu, lambda w: P[w], 6)]
    rb = [x["ret"] for x in m.run(data, dd, uu, lambda w: PB[w], 6)]
    L = [f"KiT 币圈零样本考试：{len(dd)} 周（{datetime.fromtimestamp(dd[0]/1000, timezone.utc):%Y-%m-%d} 起）",
         f"  收益 RankIC（pred7 vs 实际 7 天）均值 {statistics.fmean(ic7):+.4f}  t {tstat(ic7):+.2f}  正的周 {sum(x > 0 for x in ic7)/len(ic7):.0%}",
         f"  描述：pred20 的 RankIC {statistics.fmean(ic20):+.4f}（t {tstat(ic20):+.2f}）；波动率 RankIC {statistics.fmean(icv):+.4f}（t {tstat(icv):+.2f}）",
         f"  KiT 多 6 空 6：每周 {statistics.fmean(rk)*100:+.2f}%  夏普(年) {sharpe(rk):.2f}  累计 {(math.prod(1 + x for x in rk) - 1)*100:+.1f}%",
         f"  同期 N6V2：每周 {statistics.fmean(rb)*100:+.2f}%  夏普(年) {sharpe(rb):.2f}  累计 {(math.prod(1 + x for x in rb) - 1)*100:+.1f}%  相关 {statistics.correlation(rk, rb):+.2f}"]
    ok = statistics.fmean(ic7) > 0 and tstat(ic7) >= 2.0 and statistics.fmean(rk) > 0
    L.append(f"判定: {'PASS（只能上模拟盘）' if ok else '不过'}")
    txt = "\n".join(L); print(txt)
    (ROOT / "results" / "kitcrypto.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
