#!/usr/bin/env python3
"""Daily candles and funding for every USDT perp, including delisted ones, up to the month given (default 2024-12)."""
import hashlib, re, sys, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision/"
DATA = Path(__file__).parent / "data/universe"
LAST = sys.argv[1] if len(sys.argv) > 1 else "2024-12"


def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            err = e
        except OSError as e:
            err = e
    raise err


def listing(prefix):
    out, marker = [], ""
    while True:
        x = fetch(f"{S3}?{urllib.parse.urlencode({'delimiter': '/', 'prefix': prefix, 'marker': marker})}").decode()
        out += re.findall(r"<Key>([^<]+)</Key>", x) + re.findall(r"<Prefix>([^<]+)</Prefix>", x)[1:]
        if "<IsTruncated>true" not in x:
            return out
        marker = re.findall(r"<NextMarker>([^<]+)</NextMarker>", x)[0] if "<NextMarker>" in x else out[-1]


def months(prefix):
    keys = [k for k in listing(prefix) if k.endswith(".zip")]
    return [k for k in keys if re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) <= LAST]


def get(key):
    out = DATA / key.split("/monthly/")[1]
    if out.exists():
        return "skip"
    body = fetch(BASE + key)
    want = fetch(BASE + key + ".CHECKSUM").split()[0].decode()
    if hashlib.sha256(body).hexdigest() != want:
        raise ValueError(f"checksum mismatch {key}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    return "ok"


def main():
    root = "data/futures/um/monthly/"
    syms = [p.rstrip("/").split("/")[-1] for p in listing(root + "klines/")]
    syms = sorted(s for s in syms if re.fullmatch(r"[A-Z0-9]+USDT", s))
    print(len(syms), "symbols", flush=True)
    with ThreadPoolExecutor(6) as ex:
        keys = []
        lists = ex.map(lambda s: months(f"{root}klines/{s}/1d/") + months(f"{root}fundingRate/{s}/"), syms)
        for i, k in enumerate(lists, 1):
            keys += k
            if i % 100 == 0:
                print(f"listed {i}, files {len(keys)}", flush=True)
        print("files to get", len(keys), flush=True)
        res = list(ex.map(get, keys))
    print("ok", res.count("ok"), "skip", res.count("skip"), flush=True)


if __name__ == "__main__":
    main()
