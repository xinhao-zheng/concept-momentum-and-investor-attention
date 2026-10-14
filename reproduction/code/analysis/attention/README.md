# Attention measurement and dynamics

Run from `reproduction`:

```shell
python code/analysis/attention/run.py --config config/local.json
```

The runner prepares source bindings, checks inputs and executes three pipelines with at most two concurrent jobs and two numerical threads per process. `--prepare-only` checks preparation without estimation.

| Pipeline | Contents |
|---|---|
| `cf1_cf3_d2` | Attention persistence, bidirectional attention/trading associations and the concept-comovement bridge |
| `cf2` | Cross-source measurement, past-state controls and temporal/code-shift diagnostics |
| `m2` | Rank persistence, common-sample reranking and reference-set composition |

Inputs are the indexed attention and PIT panels, market calendar/index, C2 sample keys/cards and the declared 1h/24h ranking snapshots. `input_mode: raw-rebuilt` selects newly constructed inputs. Optional archived comparisons are reported separately from estimation.

Results are conditional associations and measurement diagnostics. Temporal ordering, cross-fitting and negative controls determine what each experiment tests; they do not themselves identify an exogenous attention effect.
