#!/usr/bin/env python3
"""Download OKX daily funding-rate files (public, no key) into data/okx_funding/daily/, then build
data/okx_funding/<OKX_BASE>.json (e.g. PEPE.json) as [[time_ms rounded to the minute, rate], ...] for USDT swaps.
Registry: PRE-REGISTERED paper_v8 A4b (2026-11-21). Re-runnable; skips files already present."""
import csv, io, json, subprocess, sys, time, zipfile
from datetime import date, timedelta
from pathlib import Path
ROOT = Path(__file__).parent
DD = ROOT / "data/okx_funding/daily"; OUT = ROOT / "data/okx_funding"
URL = "https://static.okx.com/cdn/okex/traderecords/swaprates/daily/{c}/allswap-fundingrates-{d}.zip"

def download(first=date(2022, 1, 17), last=date(2026, 11, 1)):
    d, miss = first, []
    while d <= last:
        f = DD / f"{d}.zip"
        if not f.exists():
            r = subprocess.run(["curl", "-s", "-f", "-m", "60", "-A", "research-script", "-o", str(f),
                                URL.format(c=d.strftime("%Y%m%d"), d=d.isoformat())])
            if r.returncode != 0:
                f.unlink(missing_ok=True); miss.append(d.isoformat())
            time.sleep(0.2)
        d += timedelta(days=1)
    return miss

def build():
    rows = {}
    for f in sorted(DD.glob("*.zip")):
        with zipfile.ZipFile(f) as z:
            for r in csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")):
                inst = r.get("instrument_name", "")
                if not inst.endswith("-USDT-SWAP"):
                    continue
                try:
                    t = (int(r["funding_time"]) + 30_000) // 60_000 * 60_000; x = float(r["funding_rate"])
                except (ValueError, KeyError):
                    continue
                rows.setdefault(inst[:-len("-USDT-SWAP")], {})[t] = x
    for base, d in rows.items():
        (OUT / f"{base}.json").write_text(json.dumps(sorted(d.items())))
    return len(rows)

if __name__ == "__main__":
    miss = download()
    print(f"missing days: {len(miss)}", miss[:10])
    print("bases written:", build())
