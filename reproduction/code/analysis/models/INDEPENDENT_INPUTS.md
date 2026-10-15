# Input dependencies

The default `input_mode: raw-rebuilt` constructs research inputs from local raw data. Paths are resolved from `snapshot_root` and `output_root` in the configuration.

| Calculation | Required input |
|---|---|
| Core scores and model features | Calendar, corrected prices and concept membership |
| PIT covariates | Daily stock and flow fields, market index, observation index |
| Daily attention | Raw T/S/K fields, calendar and attention observation index |
| C2 panel | Core scores, indexed PIT panel and indexed daily attention |
| N1 | C2 panel/folds and PIT covariates |
| Attention dynamics | Indexed panels; C2 sample keys/cards for D2; 1h/24h ranking snapshots for cross-source measurement |
| M2 measurement | Raw T/S/K observations and the `sh000001.csv` market calendar |

Run `prepare_inputs.py` for the four input-construction steps, then the core and model entrypoints. Generated panels are stored at:

- `core/cache`
- `pit/stock_pit_covariates_daily.universe.parquet`
- `attention-inputs/attention_daily.parquet`

The published observation indices reproduce the selected sample keys. The reconstructed PIT convention starts on 2016-07-13 and treats missing amount as zero for the specified rolling calculations. This convention was checked against the retained panel; the original historical admission program is not part of this reconstruction.

Archived panels, legacy coefficients and prior diagnostic tables are optional comparison inputs. Missing optional comparisons are recorded as skipped. They do not replace an estimation step or count as a successful result comparison. A changed input source, configuration or mode requires a new output directory.
