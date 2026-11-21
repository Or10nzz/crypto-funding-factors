"""Pre-registered 2026-11-12 (results/registry.txt): does dropping one-day pump coins help momentum N6?
Description only (all history already used). 7 start weekdays averaged."""
import math
import holdout_momentum as hm   # sets momentum.END to 2026-11-01
import momentum as m

DAY, SPLIT, N, L = m.DAY, hm.START, 6, 30
data = m.load()
for s in hm.non_coins():
    data.pop(s, None)
mondays, _ = m.setup(data, 30, N, 60)


def close(b, d):
    return b[d][1]


def daily(b, prev, k):
    """close/close returns of the k days ending at prev (None where missing)."""
    out = []
    for i in range(k):
        d = prev - i * DAY
        if d in b and d - DAY in b:
            out.append(b[d][1] / b[d - DAY][1] - 1)
    return out


def picks(day, uni, v):
    prev = day - DAY
    end, start = (prev - DAY, prev - (L + 1) * DAY) if v == "V1" else (prev, prev - L * DAY)
    sig = []
    for s in uni:
        b = data[s]["bars"]
        if start in b and end in b:
            sig.append((b[end][1] / b[start][1] - 1, s))
    sig.sort()

    def ok(s, side):
        b = data[s]["bars"]
        if v == "V2":
            r7 = daily(b, prev, 7)
            return not (side == 1 and any(x > 0.5 for x in r7) or side == -1 and any(x < -0.33 for x in r7))
        if v == "V3":
            if not (prev - L * DAY in b):
                return True
            tot = math.log(b[prev][1] / b[prev - L * DAY][1])
            lr = [math.log(1 + x) for x in daily(b, prev, L)]
            if side == 1 and tot > 0:
                return max(lr) <= 0.5 * tot
            if side == -1 and tot < 0:
                return min(lr) >= 0.5 * tot
            return True
        if v == "V4":
            for i in range(7):
                d = prev - i * DAY
                if d in b and b[d][4] > 0 and b[d][3] / b[d][4] - 1 > 1.0:
                    return False
        return True

    longs = [s for _, s in reversed(sig) if ok(s, 1)][:N]
    shorts = [s for _, s in sig if ok(s, -1) and s not in longs][:N]
    return longs, shorts


def sharpe(rs):
    mu = sum(rs) / len(rs)
    sd = (sum((x - mu) ** 2 for x in rs) / (len(rs) - 1)) ** 0.5
    return mu / sd * 52 ** 0.5 if sd else 0


if __name__ == "__main__":
    print("做法   | 2020-24 年化  回撤 夏普 | 2025-26 年化  回撤 夏普 | 平均每周换掉几个币   (7 个起始日平均)", flush=True)
    for v in ("base", "V1", "V2", "V3", "V4"):
        acc = {"a": [], "b": []}
        swapped = []
        for k in range(7):
            days = [d + k * DAY for d in mondays if d + (k + 7) * DAY <= m.END]
            unis = [m.universe(data, d, 30) for d in days]
            p = [picks(d, u, v) for d, u in zip(days, unis)]
            if v != "base":
                bp = [picks(d, u, "base") for d, u in zip(days, unis)]
                swapped += [len(set(a[0] + a[1]) - set(c[0] + c[1])) for a, c in zip(p, bp)]
            res = m.run(data, days, unis, lambda w: p[w], N)
            for key, sel in (("a", lambda d: d < SPLIT), ("b", lambda d: d >= SPLIT)):
                rs = [r["ret"] for d, r in zip(days, res) if sel(d)]
                mt = m.metrics(rs)
                acc[key].append((mt["annual"], mt["dd"], sharpe(rs)))
        row = []
        for key in ("a", "b"):
            x = acc[key]
            row.append(f"{sum(t[0] for t in x) / 7:+6.0%}  {sum(t[1] for t in x) / 7:4.0%} {sum(t[2] for t in x) / 7:4.2f}")
        sw = f"{sum(swapped) / len(swapped):.2f}" if swapped else "-"
        print(f"{v:6s} |  {row[0]}  |  {row[1]}  | {sw}", flush=True)
