#!/usr/bin/env python3
"""Fill days missing from the daily archive using the public REST API, and report what can't be filled."""
import csv, json, sys, time, urllib.error, urllib.request, zipfile, io
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent / "data/universe"
API = "https://fapi.binance.com/fapi/v1/klines"
D = 86400_000
END = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() * 1000) if "--all" in sys.argv else int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
filled = unfilled = 0
still = {}
for folder in sorted((ROOT / "klines").iterdir()):
    d = folder / "1d"
    opens = set()
    for f in d.glob("*.zip"):
        with zipfile.ZipFile(f) as z:
            for r in csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())):
                if r[0] != "open_time" and int(r[0]) < END:
                    opens.add(int(r[0]))
    patch = d / "patch.csv"
    if patch.exists():
        opens.update(int(r[0]) for r in csv.reader(patch.open()))
    ts = sorted(opens)
    rows = []
    for a, b in zip(ts, ts[1:]):
        if b - a == D:
            continue
        want = (b - a) // D - 1
        url = f"{API}?symbol={folder.name}&interval=1d&startTime={a + D}&endTime={b - D}&limit=100"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                got = [x for x in json.load(r) if a < int(x[0]) < b]
        except urllib.error.HTTPError as e:
            got = []
            print(folder.name, "API", e.code)
        rows += got
        filled += len(got)
        if len(got) < want:
            unfilled += want - len(got)
            still.setdefault(folder.name, []).append((a, b, want - len(got)))
        time.sleep(0.15)
    if rows:
        with patch.open("a", newline="") as out:
            csv.writer(out).writerows(rows)
print("filled coin-days:", filled, "still missing:", unfilled)
fmt = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")
print({s: [(fmt(a), fmt(b), n) for a, b, n in v] for s, v in still.items()})
