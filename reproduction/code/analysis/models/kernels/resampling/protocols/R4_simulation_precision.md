# Simulation-precision sensitivity

This analysis follows inspection of a near-threshold result. It repeats inference over the full fixed 96-test prediction family; models, features, training dates and selected stocks remain unchanged. The original 10,000-draw estimates remain the reported reference results.

The family comprises eight horizons, two learners and six endpoints: raw IC, industry-conditional IC, and Top20/Top100 returns under complete and zero-imputed outcomes. For base block lengths 20, 40 and 60, use L=max(base,H), with separate Holm96 correction. Require floor(n/L) >= 20; unsupported cells retain p=1 as a correction placeholder and no confidence interval. Check paired dates and exact maturity through 2026-08-31.

For each H/L, use 1,000,000 circular moving-block draws with SeedSequence([2026100824,H,L]). Share indices across paired paths; preserve sample length and equal-date weighting. Compute Float64 means, centered two-sided plus-one p-values and pointwise percentile 95% intervals.

Report Clopper-Pearson 95% intervals for Monte Carlo tail probabilities, then map all lower and upper limits through Holm separately. These ranges describe simulation uncertainty, not simultaneous effect intervals. A zero exceedance count has lower bound zero. Report all 288 cells and their original comparisons. Investigate mean differences above 1e-5 percentage points for returns or 1e-7 for IC.

The fixed draw budget is run once. No additional sampling, model search or block selection is triggered by significance. Greater Monte Carlo precision adds no independent market observations.
