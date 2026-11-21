"""Spike filter V2 (pre-registered and tested in spikefilter.py, 2026-11-12), shared by the paper trader and
the live book so they pick exactly like the backtest:
  a coin is not bought if any daily close/close in the 7 days before the open rose more than 50%,
  and not shorted if any fell more than 33%. Excluded coins are replaced by the next in the ranking;
  the short side skips coins already on the long side."""
DAY, UP, DOWN, DAYS = 86400_000, 0.5, -0.33, 7


def daily(b, prev, k=DAYS):
    """close/close returns of the k days ending at prev (days without both closes are skipped)."""
    out = []
    for i in range(k):
        d = prev - i * DAY
        if d in b and d - DAY in b and b[d][1] and b[d - DAY][1]:
            out.append(b[d][1] / b[d - DAY][1] - 1)
    return out


def ok(b, prev, side):
    r = daily(b, prev)
    return not (side == 1 and any(x > UP for x in r) or side == -1 and any(x < DOWN for x in r))


def pick(sig, bars, day, n):
    """sig: [(30-day return, symbol)] sorted ascending (momentum.signal); bars: symbol -> daily bars."""
    prev = day - DAY
    longs = [s for _, s in reversed(sig) if ok(bars[s], prev, 1)][:n]
    shorts = [s for _, s in sig if ok(bars[s], prev, -1) and s not in longs][:n]
    return longs, shorts
