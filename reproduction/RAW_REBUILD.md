# Research calculations

Run from the repository root with the scientific environment. The default `raw-rebuilt` mode calculates feature values from raw records, then selects the published date/security keys.

## Sources

Copy `reproduction/config/local.example.json` to `local.json` in the repository root. Set `snapshot_root` to the source directory and choose a fresh `output_root`. Relative `data_root` and `output_root` paths resolve beside the configuration file; a relative `snapshot_root` resolves under `data_root`. See [configuration](config/README.md).

```text
<snapshot_root>/
  raw_snapshot/
    stock-trading-data/
    stock-trading-data-pro/
    stock-main-index-data/
      sh000300.csv
      sh000001.csv
    stock-popular-concept-detail/
  attention/inputs/
    沪深_1h.csv
    沪深_24h.csv
  input_recovery/patched_raw/
    stock-trading-data/
    stock-trading-data-pro/
```

The daily stock files provide research prices and covariates. CSI 300 supplies the main market calendar and returns; M2 reads the SSE Composite calendar separately. Attention analyses use daily T/S/K observations and both timestamped popularity-list files. Concept memberships are supplied in `concept-data/concept-data.zip`. [Data access](../DATA_ACCESS.md) defines fields, encoding and history.

`sh000300.csv` requires `candle_end_time` and `close`. M2 reads only `candle_end_time` from `sh000001.csv`, as a UTF-8 CSV with a first-row header. Its dates must be unique and ascending. M2 filters dates to 2016-01-01 through 2026-08-31 and requires the last retained date to equal the cutoff. Preserve the observed market-session dates.

Use this directory layout for the complete research chain. The optional `raw_inputs` configuration overrides paths for input preparation, not every analysis module. Its keys are `prices`, `pit_prices`, `attention`, `calendar`, `membership_archive`, `price_overlay` and `pit_overlay`; paths can be absolute or relative to `snapshot_root`.

An overlay replaces a same-named raw CSV. Existing conventional overlay directories are used automatically; an explicit absent path stops preparation, while `null` disables that overlay. Apply the [daily corrections](docs/daily-input-corrections.md) when the acquired files contain the listed events. Their four replacement closing prices are provided locally.

## Preparation

```text
python reproduction/run.py --mode prepare --config local.json --plan
python reproduction/run.py --mode prepare --config local.json
```

| Step | Output |
|---|---|
| Prices, market calendar and normalized memberships | `core/cache/` |
| Daily covariates from raw observations | `pit/raw-build/` |
| Covariates selected on published sample keys | `pit/` |
| Daily attention transformations on published sample keys | `attention-inputs/` |

Preparation requires fresh output locations and stops on missing sources or failed calculations. The [covariate contract](code/analysis/input_build_README.md) and [attention constructor](code/analysis/attention_inputs_README.md) specify the transformations. The date/security indices contain sample keys, not saved feature values; their historical scope is stated in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Analysis

```text
python reproduction/run.py --artifact all --mode analyse --config local.json --plan
python reproduction/run.py --artifact all --mode analyse --config local.json
```

Add `--prepare-inputs` to prepare and analyse in one fresh output directory. Omit it when inputs already exist. The shared runner orders scores, model features, matched controls, attention inputs and their dependent calculations. Branch-specific thread and resampling settings are retained. Individual objects can be selected after their upstream inputs have been constructed.

For account results, `analyse` recomputes summary metrics from the 96 released daily paths. It does not generate trades. The same calculation can be run directly:

```text
python reproduction/code/tables/account_metrics.py --config local.json
```

## Output checks

Each module writes its results and execution record under `output_root`. Use a new directory when sources, input modes or scientific settings change. Optional historical comparison files are not inputs to fresh calculations; a missing reference is recorded as an unavailable comparison.

`render` uses released results, `verify` checks object files, and `python reproduction/verify_archive.py` checks `checksums.sha256`. These commands do not repeat the research calculations.
