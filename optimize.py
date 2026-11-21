#!/usr/bin/env python3
"""Four predefined changes to the momentum rule, aimed at its weaknesses, not at higher returns.

A multi-period rank (7, 14, 30, 90 days)  B inverse-volatility weights within each side
C 30% stop per coin per week              D funding filter: no long if 7-day funding sum > +1%,
                                            no short if < -1%; the next-ranked coin takes the slot
Adopted only if BOTH 2020-2024 and 2025-2026 show a higher return/drawdown ratio AND a higher annual
return without the best 3 weeks than the base rule. 2025-2026 has been seen once, so this cannot
prove anything by itself; only new live data can.
"""
import random, statistics, sys
from datetime import datetime, timezone
import holdout_momentum as h, momentum as m

D = m.DAY
L, N, U = 30, 5, 30
PERIODS = {"2020-24": (0, h.START), "2025-26": (h.START, m.END)}
STOP = 0.30
FUND_LIMIT = 0.01


def rank_single(data, uni, day, lookback=L):
    return [s for _, s in m.signal(data, uni, day, lookback)]


def rank_multi(data, uni, day):
    """Average rank across lookbacks. Ties break by symbol so the result is repeatable."""
    score = {s: [] for s in uni}
    for l in (7, 14, 30, 90):
        for i, s in enumerate(rank_single(data, uni, day, l)):
            score[s].append(i / max(len(uni) - 1, 1))
    usable = [(statistics.mean(v), s) for s, v in score.items() if v]
    usable.sort()
    return [s for _, s in usable]


def funding_sum(c, day, days=7):
    return m.funding(c, day - days * D, day)


def choose(data, uni, day, ranker, fund_filter):
    r = ranker(data, uni, day)
    longs = []
    for s in reversed(r):
        if len(longs) == N:
            break
        if not fund_filter or funding_sum(data[s], day) <= FUND_LIMIT:
            longs.append(s)
    shorts = []
    for s in r:
        if len(shorts) == N:
            break
        if s in longs:
            continue
        if not fund_filter or funding_sum(data[s], day) >= -FUND_LIMIT:
            shorts.append(s)
    return longs, shorts


def volatility(c, day, n=30):
    v = []
    for i in range(n):
        a, b_ = c["bars"].get(day - (i + 2) * D), c["bars"].get(day - (i + 1) * D)
        if a and b_:
            v.append(b_[1] / a[1] - 1)
    return statistics.pstdev(v) if len(v) > 5 else None


def weights_for(data, names, day, inverse_vol):
    """Each side totals 0.5 of the account."""
    if not inverse_vol:
        return {s: 0.5 / len(names) for s in names}
    vol = {s: volatility(data[s], day) for s in names}
    fallback = statistics.mean([v for v in vol.values() if v] or [1.0])
    raw = {s: 1 / (vol[s] or fallback) for s in names}
    total = sum(raw.values())
    return {s: 0.5 * raw[s] / total for s in names}


def leg_with_stop(c, day, side):
    """Weekly leg with a stop STOP away from the entry, checked on daily bars.

    Returns (return, stopped). If a day's range contains the stop, it is assumed hit: there is no way
    to tell from a daily bar whether the good or bad extreme came first.
    """
    b = c["bars"]
    entry = b[day][0]
    stop = entry * (1 - side * STOP)
    for i in range(7):
        d = day + i * D
        if d not in b:
            break
        bar = b[d]
        if i > 0 and (bar[0] <= stop if side == 1 else bar[0] >= stop):
            return side * (bar[0] / entry - 1) - side * m.funding(c, day, d), True
        if (bar[4] <= stop if side == 1 else bar[3] >= stop):
            return side * (stop / entry - 1) - side * m.funding(c, day, d + D), True
    return m.leg_return(c, day, side)[0], False


def run(data, mondays, unis, ranker=rank_single, inverse_vol=False, use_stop=False, fund_filter=False):
    out, held = [], {}
    for day, uni in zip(mondays, unis):
        longs, shorts = choose(data, uni, day, ranker, fund_filter)
        weight = {**weights_for(data, longs, day, inverse_vol), **weights_for(data, shorts, day, inverse_vol)}
        now = {s: 1 for s in longs} | {s: -1 for s in shorts}
        cost = sum(weight[s] for s, sd in now.items() if held.get(s, (None,))[0] != sd) + sum(prev for s, (sd, prev) in held.items() if now.get(s) != sd)
        ret = lg = sh = 0.0
        survivors = {}
        for s, sd in now.items():
            if use_stop:
                r, stopped = leg_with_stop(data[s], day, sd)
                gone = stopped or m.leg_return(data[s], day, sd)[1]
            else:
                r, gone = m.leg_return(data[s], day, sd)
            ret += weight[s] * r
            if sd == 1:
                lg += weight[s] * r
            else:
                sh += weight[s] * r
            if gone:
                cost += weight[s]
            else:
                survivors[s] = (sd, weight[s])
        held = survivors
        ret -= cost * m.SIDE_COST
        out.append(dict(day=day, ret=ret, long=lg, short=sh, cost=cost * m.SIDE_COST))
    return out


def summary(res):
    rets = [r["ret"] for r in res]
    x = m.metrics(rets)
    x["drop3"] = m.metrics(sorted(rets)[:-3])["annual"]
    x["worst"] = min(rets)
    x["median"] = statistics.median(rets)
    x["turn"] = statistics.mean(r["cost"] for r in res)
    return x


def block_p(a, b_, runs=4000, block=8, seed=1):
    """Share of 8-week block resamples in which the mean weekly gain from the change is <= 0."""
    d = [x["ret"] - y["ret"] for x, y in zip(a, b_)]
    rng = random.Random(seed)
    n, bad = len(d), 0
    for _ in range(runs):
        tot = 0.0
        for _ in range(n // block):
            i = rng.randrange(n - block + 1)
            tot += sum(d[i:i + block])
        bad += tot <= 0
    return bad / runs


VARIANTS = {
    "base": {},
    "A multi-period": dict(ranker=rank_multi),
    "B inverse vol": dict(inverse_vol=True),
    "C 30% stop": dict(use_stop=True),
    "D funding filter": dict(fund_filter=True),
}


def main():
    m.END = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
    data = m.load()
    nc = h.non_coins()
    for s in [s for s in data if s in nc or h.stable_like(data[s])]:
        del data[s]
    mondays, unis = m.setup(data, U, N, L)
    picks = m.rule_picks(data, mondays, unis, L, N)
    ref = m.run(data, mondays, unis, lambda w: picks[w], N)
    mine = run(data, mondays, unis)
    assert max(abs(a["ret"] - b_["ret"]) for a, b_ in zip(ref, mine)) < 1e-9, "base disagrees with momentum.run"
    print("base matches momentum.run on every week\n", flush=True)
    results = {}
    for name, kw in VARIANTS.items():
        results[name] = run(data, mondays, unis, **kw)
    verdict = {name: [] for name in VARIANTS}
    for period, (a, b_) in PERIODS.items():
        idx = [i for i, d in enumerate(mondays) if a <= d < b_]
        print(f"=== {period}: {len(idx)} weeks")
        print("variant            annual  maxDD ratio  drop3   worst  median  meanW   cost   gain-p")
        base = None
        for name in VARIANTS:
            res = [results[name][i] for i in idx]
            s = summary(res)
            if name == "base":
                base, base_res = s, res
                p = ""
            else:
                p = f"{block_p(res, base_res):.3f}"
                verdict[name].append(s["ratio"] > base["ratio"] and s["drop3"] > base["drop3"])
            print(f"{name:17s} {s['annual']:+7.0%} {s['dd']:6.0%} {s['ratio']:5.2f} {s['drop3']:+6.0%} {s['worst']:+7.1%} {s['median']:+7.2%} {s['mean']:+6.2%} {s['turn']:6.3%}  {p}")
        print()
    print("=== Decision: better ratio AND better drop-3 in both periods")
    for name in list(VARIANTS)[1:]:
        print(f"  {name:17s} 2020-24 {'yes' if verdict[name][0] else 'no ':3s}  2025-26 {'yes' if verdict[name][1] else 'no ':3s}  -> {'ADOPT' if all(verdict[name]) else 'reject'}")


if __name__ == "__main__":
    main()
