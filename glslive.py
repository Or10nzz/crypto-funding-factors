#!/usr/bin/env python3
"""gls (all-accounts long/short ratio) for the weekly gls rules, from the live API, and its check against the
official history files (the source the exam used: data.binance.vision daily metrics, rows stamped on the hour).

Definition (registry.txt, glsweekly.py): for a week opening at `day` (00:00 UTC), g = the most recent on-the-hour
value in [day-24h, day-1h].

  python3 glslive.py --check 2026-11-09    compare API and official values for that Monday's recorded universe
  python3 glslive.py --recheck [log]       recompute every logged Monday's picks (paper/gls_picks.jsonl) from the official files
Public data only, no key, no orders.
"""
import json, sys, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
H, DAY = 3_600_000, 86_400_000
URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
# Measured 2026-11-11 on the 2026-11-09 universe: the official file's row with create_time T has the value the API
# returns under timestamp T + 5 minutes (API rounded to 4 decimals). So API stamps are moved back by 5 minutes.
API_LAG = 300_000


def http(params, timeout=20):
    with urllib.request.urlopen(f"{URL}?{urllib.parse.urlencode(params)}", timeout=timeout) as r:
        return json.load(r)


def pick_stamp(rows, day):
    """rows: (t_ms, value) -> (t, value) of the most recent on-the-hour stamp in [day-24h, day-1h], or None"""
    ok = [(t, v) for t, v in rows if t % H == 0 and day - DAY <= t <= day - H and v > 0]
    return max(ok) if ok else None


def api_gls(sym, day, fetch=http):
    """(stamp_ms, g) from the live API (5-minute rows; the API keeps 30 days), or None"""
    rows = fetch(dict(symbol=sym, period="5m", startTime=day - DAY, endTime=day - H + API_LAG, limit=500))
    return pick_stamp([(int(r["timestamp"]) - API_LAG, float(r["longShortRatio"])) for r in rows], day)


def official_gls(sym, day):
    """(stamp_ms, g) from the official daily file of the day before `day`, or None if not published / no rows"""
    import histdata as hd
    rows = hd.job_metrics(sym, datetime.fromtimestamp((day - DAY) / 1000, timezone.utc).strftime("%Y-%m-%d"))
    return pick_stamp([(t, float(g)) for t, _, g in rows], day) if rows else None


def recheck(log):
    """Gate 1 of the forward test: every logged Monday's picks must come out the same from the official files."""
    import glsrank
    bad = 0
    for line in Path(log).read_text().splitlines():
        rec = json.loads(line)
        off = {s: o for s in rec["mom"] if (o := official_gls(s, rec["day"]))}
        date = datetime.fromtimestamp(rec["day"] / 1000, timezone.utc).strftime("%Y-%m-%d")
        if len(off) < len(rec["g"]):
            print(f"{date}: official files not complete yet ({len(off)} of {len(rec['g'])} coins)")
            continue
        r = glsrank.picks(rec["mom"], {s: v for s, (_, v) in off.items()}, 5, 20)
        norm = lambda x: {k: [list(v[0]), list(v[1])] for k, v in x.items()} if x else None
        same = norm(r) == norm(rec["picks"])
        stamps = all(rec["stamps"].get(s) == off[s][0] for s in rec["g"])
        rel = max((abs(rec["g"][s] - off[s][1]) / off[s][1] for s in rec["g"]), default=0)
        extra = sorted(set(off) - set(rec["g"]))
        print(f"{date}: picks {'IDENTICAL' if same else 'DIFFERENT'}; stamps {'same' if stamps else 'DIFFERENT'}; largest value difference {rel:.4%}"
              + (f"; coins the API missed: {extra}" if extra else ""))
        bad += not same
    return bad


def main():
    if "--recheck" in sys.argv:
        i = sys.argv.index("--recheck")
        return recheck(sys.argv[i + 1] if len(sys.argv) > i + 1 else ROOT / "paper" / "gls_picks.jsonl")
    if "--check" not in sys.argv:
        print(__doc__)
        return 0
    day = int(datetime.strptime(sys.argv[sys.argv.index("--check") + 1], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    weeks = json.loads((ROOT / "paper" / "momentum.json").read_text())
    uni = next(w["universe"] for w in weeks if w["day"] == day)
    same = diff = missing = 0
    worst, rel, pairs = [], [], []
    for s in uni:
        a, o = api_gls(s, day), official_gls(s, day)
        if a is None or o is None:
            missing += 1
            worst.append((s, "missing", a, o))
        elif a[0] == o[0] and abs(a[1] - o[1]) <= 1e-3 * o[1]:
            same += 1
            rel.append(abs(a[1] - o[1]) / o[1])
            pairs.append((s, a[1], o[1]))
        else:
            diff += 1
            worst.append((s, "differs", a, o))
    print(f"{len(uni)} coins: same {same}, different {diff}, missing {missing}; largest relative difference {max(rel, default=0):.4%}")
    by_api, by_off = sorted(pairs, key=lambda x: (x[1], x[0])), sorted(pairs, key=lambda x: (x[2], x[0]))
    print("order by g identical:", [x[0] for x in by_api] == [x[0] for x in by_off],
          "| lowest 5 identical:", {x[0] for x in by_api[:5]} == {x[0] for x in by_off[:5]},
          "| highest 5 identical:", {x[0] for x in by_api[-5:]} == {x[0] for x in by_off[-5:]})
    for w in worst[:10]:
        print("  ", w)
    return 0 if same == len(uni) else 1


if __name__ == "__main__":
    sys.exit(main())
