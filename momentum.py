#!/usr/bin/env python3
"""Cross-sectional momentum on the point-in-time liquid universe, up to 2024.

Each Monday, using data through Sunday's close: take the top U coins by 30-day quote volume with at
least 90 days of history, buy the N best and short the N worst over the last L days, equal notional,
held to next Monday's open. A coin that stops trading mid-week exits at its last close.

Pass criteria, fixed before the first run (L=30, N=5, U=30):
  1. annual return > 0 after costs
  2. mean weekly return above 99% of 500 random portfolios from the same weekly universes
  3. mean weekly return without the best year >= 0
  4. L=14 and L=60 also have mean weekly return > 0
  5. annual return / max drawdown above holding BTC over the same weeks
"""
import csv, io, random, statistics, sys, zipfile
from bisect import bisect_right
from datetime import datetime, timezone
from itertools import accumulate
from pathlib import Path

DAY = 86400_000
SIDE_COST = 0.0005 + 0.0005       # taker fee plus assumed slippage, per side
MIN_HISTORY = 90
END = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
EXCLUDE = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "FDUSDUSDT", "DAIUSDT", "USTUSDT", "BTCDOMUSDT", "DEFIUSDT", "BLUEBIRDUSDT", "FOOTBALLUSDT", "ALLUSDT"}
ROOT = Path(__file__).parent / "data/universe"
RUNS = 500


def rows(path):
    with zipfile.ZipFile(path) as z:
        yield from csv.reader(io.StringIO(z.read(z.namelist()[0]).decode()))


def load():
    data = {}
    for folder in sorted((ROOT / "klines").iterdir()):
        sym = folder.name
        if sym in EXCLUDE:
            continue
        bars = {}
        for f_ in (folder / "1d").glob("*.zip"):
            for r in rows(f_):
                if r[0] != "open_time" and int(r[0]) < END:
                    bars[int(r[0])] = (float(r[1]), float(r[4]), float(r[7]), float(r[2]), float(r[3]))
        patch = folder / "1d" / "patch.csv"     # days the archive lacks, from fill_universe.py
        if patch.exists():
            for r in csv.reader(patch.open()):
                if int(r[0]) < END:
                    bars[int(r[0])] = (float(r[1]), float(r[4]), float(r[7]), float(r[2]), float(r[3]))
        if bars:
            data[sym] = dict(bars=bars, times=[], pre=[0.0])
        ff = ROOT / "fundingRate" / sym
        if bars and ff.exists():
            pts = []
            for f_ in ff.glob("*.zip"):
                for r in rows(f_):
                    if r[0] != "calc_time":
                        pts.append((int(r[0]), float(r[2])))
            pts.sort()
            data[sym]["times"] = [t for t, _ in pts]
            data[sym]["pre"] = [0.0] + list(accumulate(v for _, v in pts))
    return data


def funding(c, t0, t1):
    return c["pre"][bisect_right(c["times"], t1)] - c["pre"][bisect_right(c["times"], t0)]


def universe(data, day, u):
    """Coins tradable at this open, known from data strictly before it."""
    prev = day - DAY
    ranked = []
    for sym, c in data.items():
        b = c["bars"]
        if day not in b or prev not in b or prev - (MIN_HISTORY - 1) * DAY not in b:
            continue
        vol = sum(b[prev - i * DAY][2] for i in range(30) if prev - i * DAY in b)
        ranked.append((vol, sym))
    ranked.sort(reverse=True)
    return [s for _, s in ranked[:u]]


def signal(data, syms, day, lookback):
    prev = day - DAY
    out = []
    for s in syms:
        b = data[s]["bars"]
        past = prev - lookback * DAY
        if past in b:
            out.append((b[prev][1] / b[past][1] - 1, s))
    out.sort()
    return out


def leg_return(c, day, side):
    """Open of day to next Monday's open, or to the last close if the coin stops trading."""
    b = c["bars"]
    entry = b[day][0]
    nxt = day + 7 * DAY
    if nxt in b:
        px, t1, delisted = b[nxt][0], nxt, False
    else:
        d = day
        while d + DAY in b and d + DAY < nxt:
            d += DAY
        px, t1, delisted = b[d][1], d + DAY, True
    return side * (px / entry - 1) - side * funding(c, day, t1), delisted


def run(data, mondays, unis, pick, n):
    """Weekly results. pick(week_index) gives (longs, shorts). Costs only on positions that change."""
    out, held = [], {}
    for w, day in enumerate(mondays):
        longs, shorts = pick(w)
        now = {s: 1 for s in longs} | {s: -1 for s in shorts}
        weight = 1 / (2 * n)
        cost = sum(1 for s, sd in now.items() if held.get(s) != sd) + sum(1 for s, sd in held.items() if now.get(s) != sd)
        ret = long_part = short_part = 0.0
        survivors = {}
        for s, sd in now.items():
            r, gone = leg_return(data[s], day, sd)
            ret += weight * r
            if sd == 1:
                long_part += weight * r
            else:
                short_part += weight * r
            if gone:
                cost += 1            # the forced exit is paid now
            else:
                survivors[s] = sd
        held = survivors
        ret -= cost * weight * SIDE_COST
        out.append(dict(day=day, ret=ret, long=long_part, short=short_part, cost=cost * weight * SIDE_COST))
    return out


def metrics(week):
    eq, peak, dd = 1.0, 1.0, 0.0
    for r in week:
        eq *= 1 + r
        peak = max(peak, eq)
        dd = max(dd, 1 - eq / peak)
    annual = eq ** (52.18 / len(week)) - 1 if eq > 0 else -1.0
    return dict(annual=annual, dd=dd, ratio=annual / dd if dd else float("inf"), mean=statistics.mean(week), final=eq)


def year_of(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).year


def ex_best(res):
    years = {}
    for r in res:
        years.setdefault(year_of(r["day"]), []).append(r["ret"])
    best = max(years, key=lambda y: sum(years[y]))
    rest = [x for y, v in years.items() if y != best for x in v]
    return statistics.mean(rest) if rest else float("nan"), best


def setup(data, u, n, lookback):
    first = min(min(c["bars"]) for c in data.values())
    last = max(max(c["bars"]) for c in data.values())
    d = first + (3 - (first // DAY) % 7) % 7 * DAY    # epoch day 0 was a Thursday, so +4 mod 7 is Monday
    while ((d // DAY) + 3) % 7 != 0:
        d += DAY
    mondays, unis = [], []
    while d + 7 * DAY <= last:
        uni = universe(data, d, u)
        if len(uni) >= u and len(signal(data, uni, d, lookback)) >= 2 * n:
            mondays.append(d)
            unis.append(uni)
        d += 7 * DAY
    return mondays, unis


def rule_picks(data, mondays, unis, lookback, n):
    picks = []
    for day, uni in zip(mondays, unis):
        s = signal(data, uni, day, lookback)
        picks.append(([x for _, x in s[-n:]], [x for _, x in s[:n]]))
    return picks


def main():
    data = load()
    print(f"{len(data)} coins loaded", flush=True)
    L, N, U = 30, 5, 30
    mondays, unis = setup(data, U, N, 60)    # same weeks for every lookback
    print(f"{len(mondays)} weeks, {datetime.fromtimestamp(mondays[0] / 1000, timezone.utc):%Y-%m-%d} to {datetime.fromtimestamp(mondays[-1] / 1000, timezone.utc):%Y-%m-%d}", flush=True)
    held = {s for uni in unis for s in uni}
    gone = sum(1 for s in held if max(data[s]["bars"]) < END - 30 * DAY)
    print(f"{len(held)} different coins entered the universe, {gone} of them stopped trading before the end", flush=True)
    btc = data["BTCUSDT"]["bars"]
    hold = [btc[d + 7 * DAY][0] / btc[d][0] - 1 if d + 7 * DAY in btc else 0.0 for d in mondays]
    hm = metrics(hold)
    print(f"hold BTC: annual {hm['annual']:+.0%}  maxDD {hm['dd']:.0%}  ratio {hm['ratio']:.2f}  mean weekly {hm['mean']:+.2%}\n")

    def evaluate(l, n, u):
        ms, us = (mondays, unis) if u == U else (mondays, [universe(data, d, u) for d in mondays])
        p = rule_picks(data, ms, us, l, n)
        return run(data, ms, us, lambda w: p[w], n)

    print("L   N  U    annual  maxDD ratio  meanW   longW  shortW  costW   exBest")
    results = {}
    for l, n, u in ((30, 5, 30), (14, 5, 30), (60, 5, 30), (30, 3, 30), (30, 10, 30), (30, 5, 50)):
        res = evaluate(l, n, u)
        results[l, n, u] = res
        m = metrics([r["ret"] for r in res])
        eb, by = ex_best(res)
        print(f"{l:<3d} {n:<2d} {u:<3d} {m['annual']:+7.0%} {m['dd']:6.0%} {m['ratio']:5.2f} {m['mean']:+7.3%} {statistics.mean(r['long'] for r in res):+7.3%} {statistics.mean(r['short'] for r in res):+7.3%} {statistics.mean(r['cost'] for r in res):7.3%} {eb:+7.3%} (best {by})", flush=True)
    main_res = results[L, N, U]
    print("\nby year (L=30, N=5, U=30):")
    ys = {}
    for r in main_res:
        ys.setdefault(year_of(r["day"]), []).append(r)
    hy = {}
    for d, x_ in zip(mondays, hold):
        hy.setdefault(year_of(d), []).append(x_)
    for y, v in sorted(ys.items()):
        eq = 1.0
        for r in v:
            eq *= 1 + r["ret"]
        h = 1.0
        for x_ in hy[y]:
            h *= 1 + x_
        print(f"  {y}  weeks={len(v):2d}  strategy {eq - 1:+6.0%}  long leg {statistics.mean(r['long'] for r in v):+.2%}/wk  short leg {statistics.mean(r['short'] for r in v):+.2%}/wk  BTC {h - 1:+6.0%}")
    worst = sorted(main_res, key=lambda r: r["ret"])[:3]
    print("worst weeks:", [(datetime.fromtimestamp(r["day"] / 1000, timezone.utc).strftime("%Y-%m-%d"), f"{r['ret']:+.1%}") for r in worst])

    rng = random.Random(2026)
    rand = []
    for run_i in range(RUNS):
        picks = []
        for uni in unis:
            c = rng.sample(uni, 2 * N)
            picks.append((c[:N], c[N:]))
        rand.append(run(data, mondays, unis, lambda w: picks[w], N))
        if (run_i + 1) % 100 == 0:
            print(f"random {run_i + 1}/{RUNS}", flush=True)
    means = sorted(statistics.mean(r["ret"] for r in x_) for x_ in rand)
    mm = metrics([r["ret"] for r in main_res])
    above = sum(m >= mm["mean"] for m in means)
    p = (above + 1) / (RUNS + 1)
    print(f"\nrandom portfolios: mean weekly {statistics.mean(means):+.3%}; rule {mm['mean']:+.3%}; p={p:.3f}")

    def passes(res, pval, neighbours):
        m = metrics([r["ret"] for r in res])
        return [m["annual"] > 0, pval <= 0.01, ex_best(res)[0] >= 0, neighbours, m["ratio"] > hm["ratio"]]
    nb = all(metrics([r["ret"] for r in results[l, N, U]])["mean"] > 0 for l in (14, 60))
    verdict = passes(main_res, p, nb)
    print("\n=== Verdict")
    for label, ok in zip(("1 annual return > 0", "2 above 99% of random", "3 without best year >= 0", "4 L=14 and L=60 mean > 0", "5 return/drawdown above BTC"), verdict):
        print(f"  {label:32s} {'PASS' if ok else 'FAIL'}")
    print("  OVERALL:", "PASS" if all(verdict) else "FAIL")
    # False-pass check: judge each random portfolio like a rule (criterion 4 has no analogue).
    false_pass = 0
    for i, x_ in enumerate(rand):
        mean = statistics.mean(r["ret"] for r in x_)
        pv = (sum(m >= mean for m in means) + 1) / (RUNS + 1)
        if all(passes(x_, pv, True)):
            false_pass += 1
    print(f"\nFalse-pass check: {false_pass}/{RUNS} random portfolios would pass criteria 1, 2, 3 and 5")


if __name__ == "__main__":
    sys.exit(main())
