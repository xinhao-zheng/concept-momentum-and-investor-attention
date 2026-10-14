# Experimental results

[index.csv](index.csv) links the complete parameter grids, aggregate time series, model estimates, sensitivity analyses and account paths supporting the article. The `sections` column follows the final supplementary numbering. Files larger than 1 MB use lossless CSV compression (`.csv.gz`); pandas reads both formats directly. Raw source observations, stock-level model inputs and fitted-model caches are not included.

## Manuscript resource labels

| Label in the article | Released files |
|---|---|
| Supplementary Data S9 | [Horizon associations](horizon-associations/): the nine result tables, including 486 signed values. |
| S3 Dataset | [Account results](../sections/section-s17-retrospective-account-simulations/data/): 48 paired comparisons, 96 daily paths and their path index. |

S9 is the retained dataset label; it does not denote Supplementary Section S9. The horizon results correspond to Figure S1 and Section S5. [Account metric calculations](../code/tables/account_metrics.py) use the released paths with a comparison tolerance of `1e-12`.
