# Common support and strictly prior controls

This historical diagnostic evaluates the gamma/H3 own-attention coefficients under a fixed sample and control sequence. It was specified after inspection of prior historical results. It is neither a new holdout nor a treatment-effect design.

Use own_T, own_S and own_K in the original common S-coverage window and four time folds. Keep the six-exposure finite intersection, historical 1st/99th-percentile boundaries, date demeaning and fixed historical standard deviations. Outcomes are the original complete H3 relative returns, with endpoints through 2026-08-31. Peer effects are not estimated because the selected concept is formed at the signal date.

Before reading outcomes, match each stock/observation date to the nearest strictly earlier covariate record, at most seven calendar days earlier. Both feature_asof and source_last_trade must precede that record's bound. Missing matches are explicit exclusions. Common evaluation support requires three calibrated finite doses, a finite outcome and a valid prior-record match.

| Method | Controls and training support |
|---|---|
| M0 | Original 26 numerical controls plus industry; exposure-specific dose support, finite-outcome response support; reaggregate on common evaluation rows |
| M1 | Same controls; dose and outcome use the same H3-matured, purged common training rows |
| M2 | Remove score_F5_W10, protocol_concept_degree, network_member_n and peer_count_all; retain 22 signal-date controls plus industry |
| M3 | Same 22 controls and industry as M2, measured strictly before the exposure |

M1-M0 tests support/purge alignment. M2-M1 tests removal of the concept-control group. M3-M2 tests timing with the same control set. M3-M0 cannot be attributed solely to timing.

Retain the fixed 80-round boosting specification, learning rate, leaves, regularization, seed and deterministic 30,000-row cap. Fit industry dictionaries on training rows. Each method/exposure/fold requires at least 1,000 training and 100 test rows; aggregate inference requires 10,000 rows and 100 dates. Unsupported cases remain unestimated.

Use stock-day-weighted partial-linear projection coefficients and date-aggregated scores. Form adjacent-method differences from paired influence contributions. Three exposures, three adjacent contrasts and HAC lags 1/8/12 form Holm27. The twelve method/exposure coefficients across those lags form a separate descriptive Holm36 family. Lags refer to sampled five-session dates. Report pointwise 95% intervals, fold estimates, residual fit, residual-energy concentration and date-level information.

An adjacent contrast is confirmed only if all three dependence settings support it. Non-rejection does not establish equivalence. Unmeasured common information, feedback, measurement error, historical revision and network interference remain outside this timing diagnostic. Even a nonzero M3 coefficient is a conditional historical projection.

Write inputs, matching decisions, residuals and execution receipts to this module's output directory. The execution contains 64 diagnostic nuisance fits and 16 independently repeated reference fits. Original C1/C1S results are unchanged.
