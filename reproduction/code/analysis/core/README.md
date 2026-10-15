# Core research calculations

Run from the repository root with the [local configuration](../../../config/README.md):

```shell
python reproduction/code/analysis/core/run_core.py --config local.json
```

The engine constructs F/W scores and evaluates the declared stock-selection designs. It uses configured research inputs and writes to `output_root/core`; it does not simulate trading accounts.

| Stage | Calculation |
|---|---|
| scores | Leave-one-out peer returns, synchronicity and rankings |
| S2 | Score comparisons and common-sample results |
| AGG | Aggregation formulas |
| D1 | Parameter grid |
| B1 | Benchmark comparisons |
| B3 | Monthly mappings |
| D4 | Industry-size matched controls |
| V | Annual periods and source checks |
| T2 | Information-clock comparison |
| S9 | Membership lag sensitivity |
| T3 | Alpha/beta/gamma timing designs |

Select a subset with `--stages`. Run `--check-inputs` to validate prerequisites without estimation. The cutoff remains 2026-08-31. All forward returns use the declared market-calendar endpoints; unmatched or missing outcomes retain their stated handling.

Each stage records inputs, settings and result checksums. Separate runs use separate output directories. References supplied for optional comparisons do not enter the estimators.
