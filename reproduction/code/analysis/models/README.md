# Predictive models and attention diagnostics

Run from the repository root with the [local configuration](../../../config/README.md):

```shell
python reproduction/code/analysis/models/run.py --config local.json --stages B2,C2,C1,C1S,E1,Risk,N1,R2,R4,S6bounds
```

Build the research inputs and core scores first. Outputs are written to `output_root/models`. `--prepare-only` binds inputs and compiles source without fitting models. Each separate run requires a new output directory.

| Stage | Calculation | Inputs |
|---|---|---|
| B2 / B2D | Ordinary features versus concept-augmented predictions; 224 annual fits and 28 dense-date H3 fits | Core scores, corrected prices, daily stock data |
| B2_threads4 / B2D_threads4 | Separate four-thread branch used by R2 and R4 | Same inputs; the newly constructed feature matrix may be reused |
| C2 | Attention sample, membership cards, scores, outcomes and time folds | Core outputs, daily attention, PIT covariates |
| C1 / C1S | Temporal cross-fitting across three lenses; 4,392 nuisance fits | C2 panel and folds |
| E1 / Risk | Conditional intensity and common-risk models; 14 and 28 fits | B2 features, prices and market index |
| N1 | Common-support and strictly prior controls; 64 fits and 16 reference refits | C2 panel and indexed PIT covariates |
| R2 / R4 | Portfolio-width contrast and simulation precision; 100,000 and 1,000,000 bootstrap draws | Four-thread B2 paths |
| S6bounds | Missing-outcome bounds and independent weight checks | Core D4 selections and B2 features |

Stage names retain the experiment identifiers used in the source design. B2/B2D and C1/C1S share preparation when requested together. Completed dependencies are reused only after their input, code and output hashes agree.

`input_mode: raw-rebuilt` reads newly constructed PIT and attention panels. Optional `frozen-reference` mode accepts supplied archived panels for comparison. These modes do not authenticate historical publication timestamps. See [input dependencies](INDEPENDENT_INPUTS.md).

Use the pinned environment and branch-specific thread counts. A matching MKL environment reproduced a full H3/2020 fold exactly. OpenBLAS produced small Ridge differences, including three Top100 stock-date membership changes in that fold; all Top20 returns and histogram-gradient-boosting predictions matched. Numerical equivalence is environment dependent.
