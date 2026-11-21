#!/usr/bin/env python3
"""Download the history for the historical exam (results/registry.txt, 'HISTORICAL EXAM PRE-REGISTRATION').

From data.binance.vision (public archive), only what the registered tests need, kept small:
  data/hist/universe.json      {day: top-100 symbols by mean daily quote volume over the 30 days before}
  data/hist/k1h/SYM.csv        hour_ms,open                         (monthly 1h klines)
  data/hist/metrics/SYM.csv    hour_ms,oi,gls                       (rows stamped exactly on the hour)
  data/hist/depth/SYM.csv      T_ms,imb                             (last +-1% snapshot strictly before T)
Resumable: data/hist/done.txt records every file fetched or missing. Run in tmux via ~/bin/heavy.
This script never computes a return or a signal-return relation; histexam.py does that once.
"""
import csv, io, json, sys, time, urllib.error, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import momentum

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "hist"
BASE = "https://data.binance.vision/data/futures/um"
H, D = 3_600_000, 86_400_000
POS_FROM, DEPTH_FROM, UNTIL = "2020-09-02", "2023-01-02", "2026-10-09"   # hours T in [FROM, UNTIL)
TOP_POS, TOP_DEPTH, WORKERS = 100, 40, 8
ms = lambda day: int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
day_of = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")


def days(a, b):
    d, end = datetime.strptime(a, "%Y-%m-%d"), datetime.strptime(b, "%Y-%m-%d")
    while d < end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def universe():
    f = OUT / "universe.json"
    if f.exists():
        return json.loads(f.read_text())
    vol = {}                                              # sym -> {day_ms: quote volume}
    for folder in sorted((momentum.ROOT / "klines").iterdir()):
        if folder.name in momentum.EXCLUDE or not folder.name.endswith("USDT"):
            continue
        v = {}
        for z in (folder / "1d").glob("*.zip"):
            for r in momentum.rows(z):
                if r[0] != "open_time":
                    v[int(r[0])] = float(r[7])
        p = folder / "1d" / "patch.csv"
        if p.exists():
            for r in csv.reader(p.open()):
                v[int(r[0])] = float(r[7])
        if v:
            vol[folder.name] = v
    out = {}
    for day in days(POS_FROM, UNTIL):
        t = ms(day)
        score = {s: sum(v.get(t - k * D, 0.0) for k in range(1, 31)) / 30 for s, v in vol.items()}
        out[day] = [s for s, x in sorted(score.items(), key=lambda kv: -kv[1]) if x > 0][:TOP_POS]
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def fetch(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                return z.read(z.namelist()[0]).decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 5 * attempt)
        except Exception:
            time.sleep(2 + 5 * attempt)
    raise RuntimeError(f"failed 5 times: {url}")


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


def job_k1h(sym, month):
    text = fetch(f"{BASE}/monthly/klines/{sym}/1h/{sym}-1h-{month}.zip")
    if text is None:
        return None
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if r and r[0].isdigit():
            t = int(r[0])
            rows.append((t // 1000 if t > 10 ** 14 else t, r[1]))
    return rows


def job_metrics(sym, day):
    text = fetch(f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{day}.zip")
    if text is None:
        return None
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            t = ts(r["create_time"])
        except (ValueError, KeyError):
            continue
        if t % H == 0 and r.get("sum_open_interest") and r.get("count_long_short_ratio"):
            rows.append((t, r["sum_open_interest"], r["count_long_short_ratio"]))
    return rows


def job_depth(sym, day):
    text = fetch(f"{BASE}/daily/bookDepth/{sym}/{sym}-bookDepth-{day}.zip")
    if text is None:
        return None
    snap = {}                                             # timestamp -> {pct: notional}
    for r in csv.reader(io.StringIO(text)):
        if len(r) < 4 or r[0] == "timestamp" or r[1] not in ("-1.00", "1.00", "-1", "1"):
            continue
        snap.setdefault(r[0], {})[float(r[1])] = float(r[3])
    last = {}
    for k in sorted(snap):
        d = snap[k]
        if -1.0 in d and 1.0 in d and d[-1.0] + d[1.0] > 0:
            t = ts(k)
            last[t - t % H + H] = (d[-1.0] - d[1.0]) / (d[-1.0] + d[1.0])   # later snapshots overwrite
    return sorted(last.items())


def run(kind, jobs, fn, done, log):
    todo = [j for j in jobs if f"{kind} {j[0]} {j[1]}" not in done]
    print(f"{kind}: {len(jobs)} files, {len(todo)} to fetch", flush=True)
    folder = OUT / kind
    folder.mkdir(parents=True, exist_ok=True)
    t0, got, miss = time.time(), 0, 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, ((sym, key), rows) in enumerate(zip(todo, ex.map(lambda j: fn(*j), todo))):
            if rows is None:
                miss += 1
            else:
                got += 1
                with open(folder / f"{sym}.csv", "a") as f:
                    f.writelines(",".join(map(str, r)) + "\n" for r in rows)
            log.write(f"{kind} {sym} {key} {'missing' if rows is None else 'ok'}\n")
            if i % 500 == 0:
                log.flush()
                print(f"  {kind} {i}/{len(todo)} ok {got} missing {miss} ({time.time() - t0:.0f}s)", flush=True)
    log.flush()
    print(f"{kind} done: ok {got}, missing {miss}", flush=True)


def main():
    uni = universe()
    print("universe days:", len(uni), "coins ever in top-100:", len({s for v in uni.values() for s in v}), flush=True)
    pos_pairs, depth_pairs, months = set(), set(), set()
    for day, top in uni.items():
        prev = day_of(ms(day) - D)
        for s in top:
            pos_pairs |= {(s, day), (s, prev)}
            for d in (prev, day, day_of(ms(day) + D), day_of(ms(day) + 2 * D)):
                months.add((s, d[:7]))
        if day >= DEPTH_FROM:
            for s in top[:TOP_DEPTH]:
                depth_pairs |= {(s, day), (s, prev)}
    done_f = OUT / "done.txt"
    done = set(l.rsplit(" ", 1)[0] for l in done_f.read_text().splitlines()) if done_f.exists() else set()
    with open(done_f, "a") as log:
        for kind, pairs, fn in (("k1h", months, job_k1h), ("metrics", pos_pairs, job_metrics), ("depth", depth_pairs, job_depth)):
            if kind in sys.argv[1:] or len(sys.argv) == 1:
                run(kind, sorted(pairs), fn, done, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
