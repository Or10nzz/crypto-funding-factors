#!/usr/bin/env python3
"""Binance USDT-M perp 1-hour klines (monthly archive, 2020-06..2026-10) for the 299 contracts ever in the paper's top-30
universe, into data/universe/klines/<SYM>/1h/ (same layout as the daily files). Public data, no key. Re-runnable.
Used only for the hourly price at each funding settlement (registry CLARIFICATION 2026-11-20)."""
import json, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
K = ROOT / "data/universe/klines"
MONTHS = [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13) if (2020, 6) <= (y, m) <= (2026, 10)]


def one(sym):
    d = K / sym / "1h"
    d.mkdir(parents=True, exist_ok=True)
    days = sorted(p.name for p in (K / sym / "1d").glob("*.zip"))            # only months the daily archive has
    want = [m for m in MONTHS if any(m in x for x in days)] or MONTHS
    n = 0
    for mth in want:
        f = d / f"{sym}-1h-{mth}.zip"
        if f.exists():
            n += 1
            continue
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{mth}.zip"
        r = subprocess.run(["curl", "-s", "-f", "-m", "60", "-o", str(f), url])
        if r.returncode == 0:
            n += 1
        elif f.exists():
            f.unlink()
    return sym, n


def main():
    syms = json.loads((ROOT / "results/paper/universe_syms.json").read_text())
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(one, syms))
    print(f"{sum(1 for _, n in got if n)} of {len(syms)} contracts, {sum(n for _, n in got)} monthly files")
    print("none:", " ".join(s for s, n in got if not n))


if __name__ == "__main__":
    main()
