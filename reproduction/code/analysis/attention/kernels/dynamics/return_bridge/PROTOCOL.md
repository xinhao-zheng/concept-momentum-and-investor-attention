# Attention and subsequent concept comovement

This observational diagnostic tests whether stock-level attention innovations covary with later peer comovement, relative to matched permuted graphs and a strictly past window. It uses the declared beta sample from 2022-06-01 and the inclusive observation cutoff 2026-08-31. Results are reported for the full fixed family.

Exposures are date-percentile fresh_rank_T/S/K from T_delta, S_delta_5 and K_same_weekday_z56; coefficients represent a ten-percentage-point increment. Exposures are not transformed through the concept graph. For each signal date, use the latest member snapshot strictly earlier than every included attention observation date.

The horizon is H20. Future returns span market sessions 1–20 after the signal date; past returns span sessions 1–20 before the stock's observation date. Subtract the same-day industry return. Exclude the focal stock from every concept's equal-weight peer return, then average across its eligible concepts. Require at least five return-observed peers and 16 of 20 common days. The outcome is Pearson comovement, not a propagation direction.

Permute membership rows within industry × size-tercile × concept-degree-tercile strata, requiring at least three nodes per stratum. Use the 50 fixed seeds, preserving concept sizes and total edges. Average the 50 graph correlations as the matched reference; this is not a randomization p-value. Require at least 60% exchangeable nodes and absolute mean standardized discrepancies no greater than 0.10 for size, degree and liquidity. Failed matches remain descriptive.

On common stock-days, estimate future real-minus-permuted, past real-minus-permuted and future-minus-past outcomes for all three exposures: Holm9. Date regressions include all exposures and the declared prior-or-contemporaneous momentum, risk, size, liquidity, slow-state and industry controls. Rank controls within date, median-fill missing controls and include missing indicators. Require at least 200 rows/date, 40 dates and eight complete length-eight date blocks.

Average coefficients equally across dates. Use 2,000 circular moving-block draws of eight sampled dates, shared across all nine cells. Report centered two-sided p-values, percentile intervals and Holm9. Failed support, timing and matching gates remain in the output. A future association that is not distinguishable from its past counterpart does not establish temporal specificity.
