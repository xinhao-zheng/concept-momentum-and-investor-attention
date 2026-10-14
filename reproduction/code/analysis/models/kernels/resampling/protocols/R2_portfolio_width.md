# Portfolio-width contrast

This diagnostic follows inspection of the historical H5 Ridge results. It tests the direct difference between Top100 and Top20 increments on the fixed four-thread prediction paths; it does not refit or select models.

For each formation date, D20 is the enhanced-minus-ordinary Top20 return and D100 is the corresponding Top100 difference. The estimand is Q = D100 - D20. Positive Q denotes a larger increment in the broader list. Pair dates across all four means; retain the complete-outcome and fixed-denominator zero-imputation conventions.

Use the 2020–2026 annual H5 daily and selection files. H5 enters at t+1 open and exits at t+6 open. The declared common support is 1,609 dates, 2020-01-02 through 2026-08-21, with endpoints no later than 2026-08-31. Verify unique keys, Top20 prefixes, maturity and reproduced displayed means before inference.

The family contains two missing-outcome conventions and block lengths 20, 40 and 60: six tests under Holm correction. Use 100,000 circular moving-block draws, seed 2026100823, shared paired indices and equal date weights. Report pointwise percentile 95% intervals and centered two-sided plus-one p-values. Require floor(n/L) >= 20. Block lengths are dependence sensitivities, not independent replications.

Report both positive and negative estimates, yearly and leave-one-year-out means, correlation and dispersion. The new contrast does not revise the original Holm96 decisions; failure to reject Q=0 is not equivalence. Stop after the fixed family and draw budget. Any implementation repair receives a separate receipt.
