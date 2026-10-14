# Targeted daily-input corrections

The public event ledger describes **60 events across 21 securities**. Apply the same events to the ordinary and extended daily files: remove 55 invalid security-date observations, correct four closing-price fields, and preserve one continuous canonical security-code history. The ledger contains security codes, dates, fields and validation rules; it contains no original or corrected prices.

[`prepare_daily_corrections.py`](../code/analysis/prepare_daily_corrections.py) builds the 21 + 21 overlay files from independently acquired base inputs. It requires a separate local UTF-8 CSV with `stock_code,date,close` for the four events on 2025-06-09: sh603207, sh603341, sz301539 and sz301662. Obtain those closing prices independently through the market-data access described in [data-access instructions](../../DATA_ACCESS.md). No replacement values are embedded in the public code or event ledger. The next day's ex-dividend `前收盘价` is retained unchanged; it is not a substitute for the corrected prior close.

```bash
python reproduction/code/analysis/prepare_daily_corrections.py \
  --raw-root ../data \
  --events reproduction/data/daily-correction-events.csv \
  --replacement-closes local-inputs/verified-closes.csv \
  --output work/daily-corrections
```

Pass `work/daily-corrections/stock-trading-data-pro` as `--overlays` to `input_build.py`; the price constructor uses the paired ordinary directory. No saved overlay file is needed by this preparation step. It validates the listed duplicate observations, preserves ordinary field names and unchanged source-cell strings, and replaces the descriptive CSV line with a neutral local-generation line. It records hashes and counts without writing price values into its receipt. Its default output encoding and one-line metadata convention match the daily-input reader; explicit format arguments are available.

The deleted sh603336 observation on 2025-06-09 duplicates its 2016-11-24 IPO observation. The issuer confirms suspension from 2025-06-09 in its [continuation notice](https://static.cninfo.com.cn/finalpage/2025-06-11/1223834297.PDF). The other 54 deleted observations are exact copies of the terminal OHLC, previous-close, volume and amount fields. Their keys were reconstructed from retained audit counts and that duplication invariant; the earlier per-issuer terminal-date evidence was not recovered. These are targeted study corrections, not a certification of every market observation.

The [issuer's code-transition notice](https://static.cninfo.com.cn/finalpage/2025-02-07/1222485220.PDF) identifies the change from 300114 to 302132 on 2025-02-17 with unchanged holdings. The supplied canonical sz302132 file already includes its former-code history. The constructor preserves that file and does not append a second copy of the same history.

The old extended overlays also contain 861 last-digit float differences in three financial fields after CSV rewriting. All are within relative/absolute tolerance 1e-12 and none changes missingness. They are not separate correction events, and those financial fields do not enter the 51-column PIT constructor. The new preparation step preserves unchanged source strings rather than deliberately reproducing serialization noise.

## Validation boundary

The local validation used the four already retained corrected closes as private input to test the rule implementation. That test does not independently reacquire or certify those four market prices. They remain an explicit external input for another researcher. No input file is modified, and no original header or private path is placed in the public event ledger.

Files: [event ledger](../data/daily-correction-events.csv), [input contract](../data/daily-correction-contract.json), [covariate constructor](../code/analysis/input_build_README.md).
