# Daily attention inputs

`attention_inputs.py` builds the 51-column daily panel from local raw CSVs, a market calendar and the released date/security index.

```shell
python code/analysis/attention_inputs.py --config config/local.json --workers 4
```

Run from `reproduction`. Default inputs are `snapshot_root/raw_snapshot/stock-popular-concept-detail` and `snapshot_root/raw_snapshot/stock-main-index-data/sh000300.csv`. Override them with `--raw-attention`, `--calendar`, `--observation-index` and `--output`. The default output is `output_root/attention-inputs/attention_daily.parquet`.

The raw format uses GB18030, an introductory line and a header. After the first two columns, fields are observation date, total popularity T, watchlist popularity S, stickiness K and concepts. `--encoding` changes text encoding; field order and units must match.

Filter raw observations through 2026-08-31 before transformation. Map the latest observation no later than two calendar days before each signal date. S coverage starts on 2022-03-23. Changes require uninterrupted observation-date differences; rolling standardizations use prior observations. The 24 future-attention fields are diagnostic outcomes at t+H+1, never predictors.

The constructor records source, calendar and observation-index checksums, mapped counts and clock checks. The index defines the retained sample; reconstruction does not authenticate historical publication timestamps. Existing outputs are not overwritten.
