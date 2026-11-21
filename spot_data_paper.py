#!/usr/bin/env python3
"""Binance SPOT monthly 1d klines 2020-06..2026-10 for every perp ever in the paper's top-30 universe
(results/paper/universe_syms.json). Perp 1000X / 1000000X -> spot X. Into data/spot/klines/{SPOT}/ (same layout as
spot_data.py). Public data, no key. Safe to re-run (skips files already there)."""
import json, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
D = ROOT / "data/spot/klines"
MONTHS = [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13) if (2020, 6) <= (y, m) <= (2026, 10)]


def spot_of(perp):
    m = re.fullmatch(r"(1000000|1000)(.+)", perp)
    return (m.group(2), int(m.group(1))) if m else (perp, 1)


def one(perp):
    s, mult = spot_of(perp)
    d = D / s
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for mth in MONTHS:
        f = d / f"{s}-1d-{mth}.zip"
        if f.exists():
            n += 1
            continue
        url = f"https://data.binance.vision/data/spot/monthly/klines/{s}/1d/{s}-1d-{mth}.zip"
        r = subprocess.run(["curl", "-s", "-f", "-m", "30", "-o", str(f), url])
        if r.returncode == 0:
            n += 1
        elif f.exists():
            f.unlink()
    if n == 0 and not any(d.iterdir()):
        d.rmdir()
    return perp, s, mult, n


def main():
    perps = json.loads((ROOT / "results/paper/universe_syms.json").read_text())
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(one, perps))
    have = [g for g in got if g[3]]
    json.dump({p: [s, mult] for p, s, mult, n in have}, open(ROOT / "results/paper/spot_map.json", "w"))
    print(f"{len(have)} of {len(perps)} perps have spot data; months fetched/kept: {sum(g[3] for g in got)}")
    print("no spot:", " ".join(p for p, s, m, n in got if not n))


if __name__ == "__main__":
    main()
