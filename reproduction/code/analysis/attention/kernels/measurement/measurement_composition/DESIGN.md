# Rank-reference-set composition

This descriptive analysis decomposes five-session rank changes after inspection of earlier measurement diagnostics. Raw T/S/K fields and the trading calendar are cut off at 2026-08-31. Record the actual source hashes; matching cutoff dates do not establish identical acquisition vintages.

Read stock code, date and the three original fields in Float64. Do not replace missing values with zero. Each file contains one stock and unique stock/date keys. Use the sh000001 calendar from 2016-01-01. U_t comprises stocks with all three finite fields on date t; require 30 stocks. Percentiles use average tied rank divided by the relevant pool size.

For C*=U_t intersect U_(t-5), compute:

- D_full = R_t,full - R_(t-5),full.
- D_common = R_t,common - R_(t-5),common.
- E_reference = (R_t,full - R_t,common) - (R_(t-5),full - R_(t-5),common).

Thus D_full = D_common + E_reference for each common stock. The main sample does not condition on future observations. A separate H3-complete sample aligns with the earlier measurement diagnostic.

Let h equal twice the average tied rank, an integer. With current, previous and common sizes n_t/n_p/n_c, define A=h_t,full*n_p-h_p,full*n_t and B=h_t,common-h_p,common. Then D_full=A/(2*n_t*n_p), D_common=B/(2*n_c), and E_reference has numerator A*n_c-B*n_t*n_p. Use Int64 with overflow checks. Rank A/B directly rather than rounded differences. Break Top20/Top100 boundary ties by stock code and report tie counts.

Check the integer identity and a 1e-12 percentile-scale tolerance. Strict sign reversal requires both values outside that tolerance. Report sign-class disagreement, conditional sign reversal, absolute-reference-component prevalence, overlap/Jaccard and boundary ties. Insufficient TopN support stays missing; N is not reduced.

For each date and measure, report sample sizes, entries/exits, means, population variances/covariances (ddof=0) and verify Var(full)=Var(common)+Var(reference)+2*Cov(common,reference). Aggregate by equal date weights; distinguish count-based stock-day weighted ratios. A variance ratio without covariance is not an explained share.

Reconstruct the previously inspected eight horizons 1/3/5/10/20/60/120/250 on the corresponding future-observed common stock set. Compare all 24 measure/horizon summaries and daily Spearman coefficients with tolerance 1e-10. Preserve discrepancies and acquisition labels; add no significance tests. Outcomes here are future attention ranks, not stock returns.

Run construction tests for fixed pools, inserted low-value stocks and tied values before processing data. Save source hashes, raw/rank caches, decomposition tables and comparison receipts in the configured output directory.
