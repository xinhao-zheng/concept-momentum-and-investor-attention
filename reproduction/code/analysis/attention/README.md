# Attention measurement and dynamics

Run from the repository root with the [local configuration](../../../config/README.md):

```shell
python reproduction/code/analysis/attention/run.py --config local.json
```

The runner prepares source bindings, checks inputs and executes three pipelines with at most two concurrent jobs and two numerical threads per process. `--prepare-only` checks preparation without estimation.

| Pipeline | Contents |
|---|---|
| `cf1_cf3_d2` | Attention persistence, bidirectional attention/trading associations and the concept-comovement bridge |
| `cf2` | Cross-source measurement, past-state controls and temporal/code-shift diagnostics |
| `m2` | Rank persistence, common-sample reranking and reference-set composition |

Inputs are the indexed attention and PIT panels, market calendar/index, C2 sample keys/cards and the declared 1h/24h ranking snapshots. `input_mode: raw-rebuilt` selects newly constructed inputs. Optional archived comparisons are reported separately from estimation.

M2 also reads `snapshot_root/raw_snapshot/stock-main-index-data/sh000001.csv` for its market dates. This UTF-8 CSV has a first-row header and a `candle_end_time` column with unique, ascending dates. M2 retains dates from 2016-01-01 through 2026-08-31 and requires the last retained date to equal the cutoff. It does not use this file's price fields. [Input preparation](../../../RAW_REBUILD.md) specifies the complete source layout.

Results are conditional associations and measurement diagnostics. Temporal ordering, cross-fitting and negative controls determine what each experiment tests; they do not themselves identify an exogenous attention effect.
