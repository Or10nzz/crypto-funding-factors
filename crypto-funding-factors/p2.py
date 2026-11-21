"""Activity filter P2 (pre-registered 2026-11-12, activefilter.py), shared by the paper trader so it picks like the backtest:
before taking the top 30 by 30-day turnover, drop coins whose mean daily turnover over d-30..d-1 exceeds 5x the median
of their daily turnover over d-120..d-31 (the median needs >= 60 days, otherwise the coin is dropped).
bars: symbol -> {day: (open, close, quote volume, ...)}; only finished days before d are read."""
import statistics
DAY = 86400_000
MIN_HISTORY = 90                      # momentum.MIN_HISTORY


def old_vol(b, d):
    v = [b[d - i * DAY][2] for i in range(31, 121) if d - i * DAY in b]
    return statistics.median(v) if len(v) >= 60 else None


def recent(b, d):
    v = [b[d - i * DAY][2] for i in range(1, 31) if d - i * DAY in b]
    return sum(v) / len(v) if v else 0.0


def universe(data, d, u=30):
    prev = d - DAY
    ranked = []
    for s, c in data.items():
        b = c["bars"]
        if d not in b or prev not in b or prev - (MIN_HISTORY - 1) * DAY not in b:
            continue
        o = old_vol(b, d)
        if o is None or recent(b, d) > 5 * o:
            continue
        ranked.append((sum(b[prev - i * DAY][2] for i in range(30) if prev - i * DAY in b), s))
    ranked.sort(key=lambda z: (-z[0], z[1]))
    return [s for _, s in ranked[:u]]
