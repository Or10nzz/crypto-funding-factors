# What Funding Does to Crypto Factor Returns — replication code

Code for *What Funding Does to Crypto Factor Returns: Evidence from Perpetual Futures* (Chaksan Liu, University of Macau).
It downloads public Binance data, builds the weekly top-30 perpetual universe (Sep 2020 – Oct 2026), and produces every
table and figure in the paper.

No data are redistributed here except three small derived lists (see *Included files*). All price, mark-price and
funding data come from Binance's public archive, [data.binance.vision](https://data.binance.vision), and need no API key.

## Requirements

- Python 3.10 or later, with `numpy` and `matplotlib` (`pip install -r requirements.txt`)
- `curl` on the PATH (the hourly and spot downloaders call it)
- About 0.9 GB of disk for the downloaded archives, plus about 0.2 GB for caches the scripts build on first use
  (`data/universe/*.pkl`)

## Steps

Run everything from this folder. Outputs go to `results/paper/`, and figures go to `figures/`.

```bash
# 1. Data (public archive; re-runnable, skips files already downloaded)
python download_universe.py 2026-10   # daily klines and funding for every USDT-M perpetual, including delisted ones
python fill_universe.py --all         # fills days missing from the daily archive via the public REST API
python hourly_data_paper.py           # 1h last-price klines for the 299 contracts ever in the top-30 universe
python markprice_data_paper.py        # 1h mark-price klines for the same contracts
python spot_data_paper.py             # Binance spot daily klines for the matching spot pairs

# 2. Main specification and robustness (each script writes a text report)
python paper_v5.py    # main specification -> results/paper/v5main/{v3,v3_extra,v4}.txt, v4_series.json, results/paper/v5.txt
python paper_v6.py    # margin regimes, spot-sample selection, interval changes -> results/paper/v6.txt
python paper_v7.py    # spot margin, event study, overlap, quintiles, extended Fama-MacBeth, tiered costs -> results/paper/v7.txt

# 3. Figures
python paper_v4_figs.py   # Figure 2 (fig_parts)
python paper_v7_figs.py   # Figure 1 (fig_lv_spot) and Figure 3 (fig_spec); run after paper_v4_figs.py
```

`paper_v5.py` installs the main conventions (mark-price-weighted funding, hourly isolated-margin liquidation, exit at
the mark close of the last traded hour) on top of `paper_v3.py`, `paper_v3_extra.py` and `paper_v4.py`. Do not run
those three directly if you want the paper's numbers: on their own they use the earlier conventions.

## Where each number comes from

| Paper | Output file |
|---|---|
| Table 1, persistence, turnover, weekday and tail statistics | `results/paper/v5main/v3_extra.txt` |
| Tables 2, 3, 4, 6, 8, 9 and most of the text | `results/paper/v5main/v3.txt`, `results/paper/v5.txt` |
| Unlock (hedging-demand) test | `results/paper/v5main/v4.txt` |
| Table 5A, event study, momentum/low-volatility overlap, Table 7 extensions, tiered costs | `results/paper/v7.txt` |
| Table 5B, the spot-sample selection paragraph, settlement-interval counts | `results/paper/v6.txt`, last block of `v7.txt` |
| Figures 1–3 | `figures/` |

`results/expected/` holds the reports exactly as they were produced for the submitted version, so a rerun can be
compared line by line.

Two checks in the paper were run interactively and have no script. The first compares the delivery-price proxy with
Binance's 30-minute index-price average for the nine contract-weeks with a delisting (maximum difference 5.3%). Its
archive files are listed in `data/index_files.txt` (`data/futures/um/monthly/indexPriceKlines/<SYMBOL>/1m/`). The second
is the day-by-day ALPACA table in Appendix A, which is read from the settlement records in the same data.

## Pre-registration

`results/registry_paper.txt` is an extract of the research log. Each test was written into it before it was computed.
Changes made after seeing results are marked `CLARIFICATION` (POST-HOC where applicable) and are described in the
paper. These include the dead-contract filter, rounding funding timestamps to the minute, and the exit rule for
contracts that leave the universe.

## Included files

- `results/paper/universe_syms.json`: the 299 contracts that are ever in the top-30 universe. These are the download
  targets for the hourly and spot data.
- `results/paper/spot_map.json`: the matching Binance spot pair and price multiplier for each perpetual (for example,
  1000-unit contracts).
- `results/paper/non_coins.json`: contracts that Binance does not classify as crypto coins (equity, index and
  commodity perpetuals), excluded from the universe. This is a frozen snapshot of `exchangeInfo`, so later
  reclassifications do not change the sample.
- `data/unlock/events.json`: insider and investor cliff-unlock events derived from DefiLlama's public emissions
  datasets (downloaded 2026-11-16), used only for the unlock test.

## Notes

Several helper modules (`momentum.py`, `paper_momentum.py`, `holdout_momentum.py` and others) come from the wider
research project and contain code paths the paper does not use. They are included because the paper scripts import
them.

## License

MIT (see `LICENSE`). If you use the code, please cite the paper.
