#!/usr/bin/env python3
"""Measure what a market order would really cost on the live order book. Public data, no orders."""
import json, statistics, time, urllib.request
from pathlib import Path

SIZES = (20, 100, 500, 1000, 5000)
LOG = Path(__file__).parent / "paper/depth.jsonl"


def walk(levels, notional):
    """Average fill price for spending notional USDT through the levels, or None if the book runs out."""
    left, cost, qty = notional, 0.0, 0.0
    for price, size in levels:
        value = price * size
        take = min(value, left)
        qty += take / price
        cost += take
        left -= take
        if left <= 1e-9:
            return cost / qty
    return None


def measure(book, side):
    """Spread and slippage for one coin. Slippage is cost against the mid price, in percent; never negative."""
    bids = [(float(p), float(q)) for p, q in book["bids"]]
    asks = [(float(p), float(q)) for p, q in book["asks"]]
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2
    levels = asks if side == 1 else bids        # opening long buys asks, opening short sells bids
    out = dict(mid=mid, spread_pct=(best_ask - best_bid) / mid * 100, depth_1pct=sum(p * q for p, q in levels if abs(p / mid - 1) <= 0.01))
    for n in SIZES:
        fill = walk(levels, n)
        out[f"slip_{n}"] = None if fill is None else abs(fill / mid - 1) * 100
    return out


def snapshot(positions, get, when, path=LOG):
    """positions: {symbol: side}. Appends one line per coin; one bad coin doesn't stop the others."""
    rows = []
    for sym, side in sorted(positions.items()):
        try:
            rows.append(dict(t=when, symbol=sym, side=side, **measure(get("depth", symbol=sym, limit=500), side)))
        except Exception as ex:
            rows.append(dict(t=when, symbol=sym, side=side, error=f"{type(ex).__name__}: {ex}"))
        time.sleep(0.1)
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    return rows


def summarise(rows):
    """Median and worst slippage per order size across the coins in rows."""
    good = [r for r in rows if "error" not in r]
    out = {}
    for n in SIZES:
        v = [r[f"slip_{n}"] for r in good if r[f"slip_{n}"] is not None]
        out[n] = dict(median=statistics.median(v) if v else None, worst=max(v) if v else None, ran_out=sum(r[f"slip_{n}"] is None for r in good))
    return out
