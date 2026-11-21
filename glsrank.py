"""The weekly gls rules' ranking, shared by the exam scripts (glsweekly_dev2.py) and the paper trader, so both run
the same code. Pure function: no data loading, no network. Rules as registered in results/registry.txt (2026-11-11):
  Gc : long the n lowest g (crowd short), short the n highest g (crowd long)
  MGc: sort by (momentum rank - gls rank), ties by symbol; long the top n, short the bottom n
"""


def picks(mom, g, n=5, min_both=20):
    """mom: symbols in ascending 30-day momentum order; g: {symbol: gls}. -> {'Gc': (longs, shorts), 'MGc': (...)} or None"""
    both = [s for s in mom if s in g]
    if len(both) < min_both:
        return None
    by_g = sorted(both, key=lambda s: (g[s], s))
    rg = {s: i for i, s in enumerate(by_g)}
    rm = {s: i for i, s in enumerate(both)}
    by_c = sorted(both, key=lambda s: (rm[s] - rg[s], s))
    return dict(Gc=(by_g[:n], by_g[-n:]), MGc=(by_c[-n:], by_c[:n]))
