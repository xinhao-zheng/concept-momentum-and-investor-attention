# Measures

`results/decomposition_daily.csv` contains one date × measure × support row. Means and standard deviations use percentile units; variances and covariances use squared percentile units, with ddof=0.

| Field group | Definition |
|---|---|
| current_past_common_no_future_filter | Main current/past intersection; no future-data condition |
| H3_future_complete_common | The same current/past ranks, evaluated on the H3-observed subset; at least 30 stocks and six defined baseline/fixed Spearman correlations |
| n_full / n_previous / n_common / n_evaluated | Current, past, common and evaluated pool sizes |
| n_entered / n_exited | Changes in finite-data pool membership; not listings, delistings or investor counts |
| full / common / reference | Full-pool change, common-pool change and their difference |
| identity_error | Maximum error of the stock-level identity |
| variance_identity_error | Var(full)-Var(common)-Var(reference)-2*Cov(common,reference) |
| strict_sign_flip_fraction | Opposite signs, both beyond 1e-12, divided by evaluated stocks |
| sign_flip_given_both_nonzero | Same numerator, divided by stocks with both changes nonzero |
| sign_class_mismatch_fraction | Different negative/zero/positive classes under 1e-12 tolerance |
| reference_abs_larger_fraction | abs(reference)>abs(common)+1e-12; not a causal or explained share |
| topN_overlap_fraction / topN_jaccard | Intersection/N and intersection/union |
| topN_*_boundary_ties | Count tied at the Nth boundary; stock code resolves selection |

For the complete common set, mean_common is approximately zero by rank algebra; that property need not hold after H3 subsetting. It is not an economic zero-effect result. Insufficient-N and zero-denominator statistics remain missing.

`decomposition_summary.csv` distinguishes equal-date means from stock-day weighted ratios. Mean standard deviations must not be squared to obtain mean variances.

The raw cache preserves original values, including dates outside the calculation calendar. Integer half-rank caches use zero for missing ranks; the smallest valid half-rank is two. Stock-day components store A/B/E_numerator and pool sizes, with:

- full=A/(2*n_full*n_previous)
- common=B/(2*n_common)
- reference=E_numerator/(2*n_full*n_previous*n_common)

The eight-horizon validation files compare measurement ranks with prior diagnostics. They do not turn the acquisition batch into an authenticated historical snapshot.
