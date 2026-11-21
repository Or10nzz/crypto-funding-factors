#!/usr/bin/env python3
"""One-time exam of the momentum rule on 2025-01 to 2026-10, data it has never been run on.

Only the original setting: L=30, N=5, U=30. It failed criterion 4 in development, and the user chose
to examine it anyway. Pass, fixed before the first run:
  1. annual return > 0 after costs
  2. mean weekly return above 99% of 500 random portfolios
  3. annual return / max drawdown above holding BTC
"""
import json, random, statistics, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import momentum as m

LOCK = Path(__file__).parent / "results/holdout_momentum.lock"
START = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
m.END = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
L, N, U = 30, 5, 30
fmt = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")


def non_coins():
    """Contracts Binance itself does not class as crypto coins.
    Uses the frozen snapshot results/paper/non_coins.json when present (replication); otherwise queries Binance."""
    snap = Path(__file__).parent / "results/paper/non_coins.json"
    if snap.exists():
        return set(json.loads(snap.read_text())["symbols"])
    with urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30) as r:
        info = json.load(r)
    kinds = {}
    for s in info["symbols"]:
        kinds.setdefault(s.get("underlyingType"), []).append(s["symbol"])
    print("Binance underlying types:", {k: len(v) for k, v in kinds.items()})
    return {s for k, v in kinds.items() if k != "COIN" for s in v}


def stable_like(c):
    """A coin that sat at about 1 USD for its whole life."""
    px = [v[1] for v in c["bars"].values()]
    return 0.97 <= statistics.median(px) <= 1.03 and max(px) / min(px) < 1.15


def main():
    if LOCK.exists() and "--after-data-fix" not in sys.argv:
        raise SystemExit("This exam already ran. Rerun only with --after-data-fix for a real data bug, and say so.")
    data = m.load()
    nc = non_coins()
    drop = sorted(s for s in data if s in nc or stable_like(data[s]))
    print(f"excluded before any returns were computed ({len(drop)}):", drop)
    for s in drop:
        del data[s]
    mondays, unis = m.setup(data, U, N, L)
    keep = [i for i, d in enumerate(mondays) if d >= START]
    mondays, unis = [mondays[i] for i in keep], [unis[i] for i in keep]
    print(f"{len(data)} coins, {len(mondays)} weeks, {fmt(mondays[0])} to {fmt(mondays[-1])}")
    picks = m.rule_picks(data, mondays, unis, L, N)
    forced = [(fmt(d), s, sd) for w, d in enumerate(mondays) for sd, names in ((1, picks[w][0]), (-1, picks[w][1])) for s in names if m.leg_return(data[s], d, sd)[1]]
    print("forced exits:", forced or "none")
    print("coins most often held:", sorted({s: sum(s in p[0] + p[1] for p in picks) for w in picks for s in w[0] + w[1]}.items(), key=lambda kv: -kv[1])[:10])
    LOCK.parent.mkdir(exist_ok=True)
    LOCK.write_text(fmt(int(datetime.now(timezone.utc).timestamp() * 1000)))
    res = m.run(data, mondays, unis, lambda w: picks[w], N)
    rets = [r["ret"] for r in res]
    x = m.metrics(rets)
    btc = data["BTCUSDT"]["bars"]
    hold = [btc[d + 7 * m.DAY][0] / btc[d][0] - 1 if d + 7 * m.DAY in btc else 0.0 for d in mondays]
    h = m.metrics(hold)
    print(f"\nrule: annual {x['annual']:+.0%}  maxDD {x['dd']:.0%}  ratio {x['ratio']:.2f}  mean weekly {x['mean']:+.3%}  long {statistics.mean(r['long'] for r in res):+.3%}  short {statistics.mean(r['short'] for r in res):+.3%}  cost {statistics.mean(r['cost'] for r in res):.3%}")
    print(f"hold BTC: annual {h['annual']:+.0%}  maxDD {h['dd']:.0%}  ratio {h['ratio']:.2f}  mean weekly {h['mean']:+.3%}")
    print("weeks up:", f"{sum(r > 0 for r in rets) / len(rets):.0%}", " best", f"{max(rets):+.1%}", " worst", f"{min(rets):+.1%}")
    ys = {}
    for r in res:
        ys.setdefault(m.year_of(r["day"]), []).append(r)
    for y, v in sorted(ys.items()):
        eq = 1.0
        for r in v:
            eq *= 1 + r["ret"]
        print(f"  {y}: {len(v)} weeks, strategy {eq - 1:+.0%}, long {statistics.mean(r['long'] for r in v):+.2%}/wk, short {statistics.mean(r['short'] for r in v):+.2%}/wk")
    rng = random.Random(2027)
    means = []
    for _ in range(m.RUNS):
        pk = []
        for uni in unis:
            c = rng.sample(uni, 2 * N)
            pk.append((c[:N], c[N:]))
        means.append(statistics.mean(r["ret"] for r in m.run(data, mondays, unis, lambda w: pk[w], N)))
    p = (sum(v >= x["mean"] for v in means) + 1) / (m.RUNS + 1)
    print(f"random portfolios: mean weekly {statistics.mean(means):+.3%}; p={p:.3f}")
    print("\n=== Exam verdict")
    checks = (("1 annual return > 0", x["annual"] > 0), ("2 above 99% of random", p <= 0.01), ("3 return/drawdown above BTC", x["ratio"] > h["ratio"]))
    for label, ok in checks:
        print(f"  {label:32s} {'PASS' if ok else 'FAIL'}")
    print("  OVERALL:", "PASS" if all(ok for _, ok in checks) else "FAIL")


if __name__ == "__main__":
    main()
