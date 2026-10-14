# Concept momentum and investor attention: Evidence from China’s A-share market

## Abstract

Overlapping concepts link stocks across industry boundaries in China’s A-share market. Starting from the relationship between concepts’ past and subsequent returns across horizons, we examine whether concept strength and stock–peer synchronization help select members with higher subsequent short-horizon returns, and whether investor attention can explain this return pattern. Using 5,228 stocks from December 2016 through August 2026, we combine recent peer returns with stock–peer return correlations in a ranking with transparent rules. Across 108 combinations of formation windows, holding horizons and list sizes, the joint ranking yields positive mean gross return spreads over the corresponding research pools; 65 remain significant after Holm correction across the declared 540 comparisons. Under the reference specification of five-day strength, ten-day synchronization and the top 20 stocks, the mean three-day spread from the next open is 0.2832 percentage points; the industry–size-matched spread is 0.1681 percentage points (95% interval, 0.1039–0.2311), also significant after multiplicity correction. Adding synchronization improves strength-only selection, whereas the increment over synchronization alone is not statistically established. Concept features improve some prediction tasks beyond conventional characteristics; the increment for the reference task is not statistically confirmed. Across the tested parameter range, the member-selection results support short-horizon return continuation associated with concepts. In the causal machine-learning diagnostics, neither the adjusted stock-popularity–return coefficients nor the paired specification contrasts survive the declared multiplicity correction. Return evidence from member selection does not identify an attention mechanism.

**Keywords:** Concept momentum; Return predictability; Investor attention; Quantitative investment strategies; Causal machine learning

**JEL classification:** G11; G12; G14; G41

## Introduction

China’s A-share market groups companies into concepts associated with technologies, policies and economic narratives. These labels cross industry boundaries, and a company can belong to several concepts at once. A strong concept does not identify which of its members will subsequently earn higher returns. We examine whether a stock’s recent alignment with its concept peers helps resolve this selection problem.

Related-security predictability motivates this question at two levels. At the category level, evidence allowing for overlapping memberships documents monthly Chinese concept momentum, stronger momentum among less-attended concepts, and stock-level lead–lag relations within concepts [@du_etal_2022]; stock momentum also varies with comovement with size and value styles [@wahal_yavuz_2013]. Across firms, customer–supplier links predict returns [@cohen_frazzini_2008], while evidence from China compares news-co-mention momentum with economic links that include concepts [@ge_li_zheng_2025]. Separating peer-average characteristics from a firm’s deviations within its peer group provides a complementary approach to return prediction [@avramov_etal_2026]. These studies connect group information to individual returns; our question is how that information guides short-horizon selection among overlapping concept members.

To answer this question, we distinguish the information in recent synchronization, the rules used to aggregate it and the increment over conventional predictors. The holding horizons examined range from 1 to 250 trading days, with returns from historical trading simulations serving as a reference for detailed comparisons. This differs from the three- to twelve-month stock momentum studied by Jegadeesh and Titman [@jegadeesh_titman_1993] and the short-lag dependence in monthly returns documented by Jegadeesh [@jegadeesh_1990]. Concept momentum here refers to testing the return continuation of strong concepts through their selected member stocks.

Investor attention offers an economic interpretation of concept-based trading (Fig. 1c). Category-based investment and limited-attention models link grouping to comovement and predictability [@barberis_shleifer_2003; @peng_xiong_2006], while laboratory evidence shows that attention constraints can shift information acquisition toward categories [@bhui_jiao_2023]. Observed attention measures, however, capture particular activities: searches [@da_engelberg_gao_2011] and platform-related buying [@barber_etal_2022] need not have the same return implications. We therefore treat Wind’s individual-stock popularity as a separate observable quantity and examine its associations with trading participation, returns and concept comovement. These associations are used to assess whether attention can explain the observed selection pattern.

The empirical design follows the selection problem. Recent peer returns measure concept strength, and stock–peer return correlations measure synchronization. Their rank-based combination uses no weights fitted to future returns. Peer portfolios use a leave-one-out construction to prevent a stock from mechanically entering its own peer return; measuring returns from the next open separates score formation from outcome evaluation in time. Industry–size matching tests whether composition accounts for the return spread, while component removal and alternative aggregation distinguish the roles of input information and combination rules. Fixed model comparisons assess the increment over conventional characteristics, drawing on machine-learning prediction in empirical asset pricing [@gu_kelly_xiu_2020]. Popularity analyses examine the attention interpretation through separate measurements; retrospective accounts show allocation outcomes under a different implementation framework.

## Data and empirical design

### Sample and empirical specifications

The historical sample contains 5,228 stocks and 8,835,000 stock–date observations on 2,353 signal dates, from 22 December 2016 through 31 August 2026. The daily research pool grows from about 1,900 stocks in 2017 to 5,100 in 2026. Daily membership snapshots contain about 150 concept labels in 2017 and 385 in 2026; pool stocks have a median of four scored memberships (Supplementary Table S1). The pool reflects research-data availability rather than a verified historical investable universe.

The specification grid crosses six F/W window pairs, three holding horizons and six list sizes, giving 108 specifications for each ranking rule and testing sensitivity to these choices. The other comparisons use five-day concept strength and ten-day stock–concept synchronization, denoted F5/W10, together with a Top20 list as a common reference specification. Returns are evaluated over three complete trading days from the next market-day open. Within each paired comparison, competing rules share eligible stocks, signal dates and return endpoints.

### Theme relations and the ranking rule

Concept strength F compounds recent daily returns of a stock’s concept peers. Synchronization W is the correlation between the stock’s daily returns and those of its peers, with the focal stock excluded from the peer portfolios. Both quantities are ranked across valid stock–concept membership records. FW multiplies the ranks within each membership and averages the products across a stock’s memberships (Fig. 1a). A high score thus identifies stocks connected to strong concepts whose recent movements they have followed closely.

[FIGURE_1]

### Information timing and attention measures

On signal day t, we use the latest membership snapshot dated no later than the preceding market day and prices through the t close. The ranked list is fixed before returns are attached, with no replacement selections when future returns are missing. We measure an H-day return from the adjusted open at t + 1 to the adjusted open at t + H + 1 (Fig. 1b), so the illustrative three-day reference ends at the t + 4 open; this design is denoted γ. Although the score uses lagged membership records, eligibility for the common comparison pool still requires valid scores under synchronous membership and both formation-window conventions (Supplementary Section S3). All source observations and prescribed outcome endpoints must fall on or before 31 August 2026; Supplementary Table S2 gives the mature sample by horizon. Recorded snapshot dates are not equivalent to the time of first public availability (Supplementary Section S1).

Table 1 separates membership timing from return timing while retaining prices through t close in every cell. The within-clock comparisons isolate the membership convention; the within-membership comparisons isolate the return clock. Close-based cells are diagnostic references: because the score already uses the signal-day closing price, trades cannot be assumed executable at that same close.

**Table 1 | Membership timing and return timing.** Here t is a signal date on the market calendar; H counts market trading days. O and C denote adjusted open and close prices. A recorded snapshot date is distinguished from its unverified first public availability.

| Design | Membership record used for scoring | Score price information | Return interval | Comparison role |
| --- | --- | --- | --- | --- |
| α | Date no later than t | Through close t | $C_t\rightarrow C_{t+H}$ | Same-day-membership, closing-return reference |
| β | Date no later than t | Through close t | $O_{t+1}\rightarrow O_{t+H+1}$ | Same-day-membership, next-open evaluation |
| γ | Date no later than the preceding market day | Through close t | $O_{t+1}\rightarrow O_{t+H+1}$ | Lagged-membership, next-open evaluation |
| δ | Date no later than the preceding market day | Through close t | $C_t\rightarrow C_{t+H}$ | Lagged-membership, closing-return reference |

Wind total popularity (T), watchlist popularity (S) and attention stickiness (K) record individual-stock prominence within the vendor’s measurement system but do not directly observe theme-level attention. Their transformations separate persistent levels from more recent changes, with samples determined by field coverage and historical-input requirements (Supplementary Sections S12–S13). Concept-level profiles and implementation accounts have their own construction rules and samples, described in Supplementary Sections S5 and S17.

## Results

### Short-horizon returns within overlapping concepts

Under the F5/W10 reference specification, the Top20 list has a mean three-day gross return of 0.3969% across 2,349 mature signal dates, compared with 0.1136% for the equally weighted research pool, a difference of 0.2832 percentage points (Fig. 2a).

To assess how the result depends on the formation windows, the sensitivity grid crosses six F/W pairs, F3/W5, F3/W20, F5/W10, F10/W20, F20/W10 and F20/W50, on the market-trading-day calendar. Within each pair's common eligible pool, we compare five scoring rules across three holding horizons, H3, H5 and H10, and six list sizes, Top5, Top10, Top20, Top30, Top50 and Top100. All 108 FW-minus-pool means are positive, and 65 pass Holm correction in the declared 540-comparison family (Fig. 2). These are related sensitivity comparisons: settings share stocks and dates, and longer windows can change the eligible pool. Concept-level formation–holding profiles provide a complementary view of its time scale (Supplementary Fig. S1 and Supplementary Data S9).

[FIGURE_2]

### Industry, size and variation over time

If selected stocks differ systematically from the research pool in industry and size composition, the spread may partly reflect those differences. We match each frozen Top5, Top10 and Top20 list to weighted group means that reproduce its industry composition, capitalization-decile composition or both (Fig. 3). The procedure retains all ten size groups and includes selected stocks in their comparison groups; it changes the benchmark rather than the selections.

[FIGURE_3]

With industry and size matched jointly, the mean spreads are 0.2053, 0.1882 and 0.1681 percentage points for Top5, Top10 and Top20, respectively. The Top20 spread has a 95% time-block interval of [0.1039, 0.2311] percentage points and a Holm-adjusted P value of 0.000900. All nine industry-, size- and jointly matched comparisons are positive and survive adjustment within this declared family. For Top20, the industry-only and size-only spreads are 0.2075 and 0.2522 percentage points. The spread therefore does not arise entirely from the observed industry and size composition.

Random portfolios matched on industry and size give a similar result. For Top20, the actual period mean exceeds the median of 1,000 matched portfolio means by 0.1680 percentage points. Retaining the intended list denominator and assigning zero to missing outcomes also leaves the joint-matched spread at 0.1681 percentage points. The random-portfolio distribution describes variation across matched draws; the confidence interval above describes uncertainty over time.

The spread varies across calendar years (Supplementary Fig. S2 and Table S6). Among the nine complete years from 2017 to 2025, the pool-relative mean is positive in seven and the joint industry–size mean in eight. Pool-relative spreads are −0.0045 percentage points in 2022 and −0.0704 in 2024; the joint-matched spread is −0.0317 in 2024.

In the fixed 2017–2021 and 2022–31 August 2026 periods, pool-relative spreads average 0.3364 and 0.2270 percentage points, and joint industry–size spreads average 0.1919 and 0.1426. All four contrasts pass their separate Holm4 families with 20- and 40-date blocks. Disjoint three-day formation offsets retain positive historical means, and the matched spread also remains positive in the common attention-data window. Supplementary Section S6 reports the date support, longer-block results, partial-year means and common-window estimates.

### Membership timing and return measurement

The timing of membership records and the return measurement clock may each affect the spread; the four information-timing designs separate the two. With market-calendar scores, the F5/W10 H3 spreads over the eligible pool are 0.3170, 0.2766, 0.2832 and 0.3213 percentage points for α, β, γ and δ, respectively (Table 1). All four use 2,349 common mature signal dates from 22 December 2016 to 25 August 2026. Within each membership convention, the close and next-open outcomes evaluate exactly the same intended selections.

The synchronous-minus-lagged membership contrasts are −0.0067 percentage points under the open clock and −0.0043 under the close clock. Neither passes the correction. By contrast, the close-reference spread exceeds its next-open counterpart by 0.0404 percentage points with synchronous labels and 0.0380 with lagged labels. Their pointwise 95% intervals with 20-day blocks are [0.0190, 0.0618] and [0.0165, 0.0597], with Holm20 P values of 0.0050 and 0.0076. Both clock contrasts also pass with 60-day blocks and when missing returns are set to zero at the original denominators. The interaction between membership timing and the return clock remains unconfirmed (Supplementary Tables S4 and S5).

Spreads are larger under the close clock. Because the score already uses signal-day closing prices, the close-based results are measurement diagnostics and do not provide evidence of execution at that close. The main selection result remains positive with lagged membership and next-open outcomes.

### Information in the inputs and their combination

The FW spread may arise primarily from one of its inputs. The two-input FW spread is 0.2832 percentage points, while strength-only (F) and synchronization-only (W) selection have reference pool-relative spreads of 0.0544 and 0.1267 percentage points, respectively. FW exceeds strength-only selection in all 108 parameter cells, with 81 corrected rejections. Its difference from synchronization-only selection is positive throughout, but none survives the same correction. The supported increment therefore concerns adding synchronization to strength; the increment from adding strength to synchronization is not statistically supported. The contrast between these two significance decisions is not itself a test of the difference between the two increments [@gelman_stern_2006]. The fixed-denominator sensitivity gives the same pattern of decisions (Supplementary Section S3).

The two inputs can be combined without the membership-level product. Holding both inputs and valid memberships fixed, A averages the separate stock-level strength and synchronization scores, while P multiplies their separate stock-level means. If f and w denote membership ranks, the distinction between P and FW is exact:

$$
FW_{i,t}=P_{i,t}+\operatorname{Cov}_{c}(f_{ic,t},w_{ic,t}).
$$

The covariance uses the number of memberships as its denominator. It records whether a stock's relatively strong concept memberships are also those with relatively high synchronization. This is the information preserved by multiplying before averaging rather than afterwards. Accordingly, comparing FW with P asks whether retaining that alignment changes the selection enough to improve outcomes; comparing FW with A additionally asks whether multiplication is needed to combine the two ranked inputs.

The aggregation choice changes few selections. Across all 2,353 signal dates, mean Top20 overlap is 96.74% between FW and A and 97.97% between FW and P (Fig. 4). On mature H3 windows, mean return differences are 0.000604 and 0.006048 percentage points, respectively (Supplementary Table S7). Both pointwise intervals span zero under the main and longer resampling blocks. In other words, the three rules select nearly the same concentrated portfolio from the two inputs. The six-pair F/W grid also shows no corrected aggregation advantage (Supplementary Section S3).

[FIGURE_4]

### Prediction conditional on conventional characteristics

Concept strength can coincide with recent stock returns, while synchronization can accompany differences in volatility or trading activity. Paired prediction models therefore test whether concept features provide incremental information beyond these conventional characteristics. Ridge regression and histogram gradient boosting use 19 characteristics and a missingness feature; enhanced versions add FW, F-only and W-only. Paired models share training rows, settings and evaluation pools. Annual predictions from 2020 to 2026 use the preceding three calendar years and matured outcomes (Methods).

The evaluation distinguishes two tasks: ranking the full research pool and selecting its highest-ranked stocks. Daily rank correlation between predictions and future returns, the information coefficient (IC), measures the former; Top20 and Top100 gross returns measure the latter. Better ordering across the pool need not improve its selected tail (Supplementary Fig. S6). Figure 5 reports enhanced-minus-conventional contrasts within each task.

[FIGURE_5]

At the reference three-day horizon, both learners have positive Top20 return and raw pool-IC increments, but all four Holm-adjusted P values equal 1. Using every eligible H3 training date also yields no corrected Top20, Top100 or raw pool-IC increment. Boosting’s industry-conditioned IC increases by 0.005414 and passes correction under 20- and 60-day blocks (Supplementary Section S9 and Table S8).

At H5, ridge’s Top100 return increment is 0.1397 percentage points, compared with 0.1837 for Top20. The direct Top100-minus-Top20 increment is −0.0440 percentage points (pointwise 95% interval with 20-date blocks, −0.1307 to 0.0395; Holm6 P = 1). The direct comparison does not establish a larger increment for the broader list. H5/Top100 support persists across longer blocks and the simulation-precision check; the Top20 decision varies with block length, missingness and simulation precision. Supplementary Section S8 reports the paired list-width and precision analyses.

After subtracting industry mean ranks from predictions and outcomes, three IC increments of 0.004114–0.004903 pass the original 10,000-replicate correction. Ridge H1 remains confirmed under both longer block choices; ridge H5 and boosting H20 depend on block length (Supplementary Section S8).

The component comparison does not confirm an increment from strength beyond synchronization. This motivates a conditional comparison: H3/Top20 boosting models fitted on all training dates retain W and conventional characteristics in both models and add F and FW jointly. The mean increment is −0.0225 percentage points, with Holm6 P = 1 under L20 and L60, leaving a positive increment from this block unconfirmed under the fixed specification (Supplementary Section S10 and Table S9).

Synchronization can reflect exposure to common returns or the magnitude of residual variation. A four-model H3/Top20 comparison adds W to conventional predictors and F, then repeats that contrast with lagged market beta and residual volatility also included. The W increments are 0.0742 and 0.0032 percentage points, respectively. Neither increment nor their direct difference passes the shared six-test Holm correction (Supplementary Section S11 and Table S10). Thus, the improvement from synchronization over strength-only ranking is not confirmed conditional on this richer predictor set.

Daily adaptations of published concept-momentum and comovement rules provide another set of benchmarks. FW has higher mean returns than all four specified daily rules on their matched pool. The monthly mapping gives a different pattern across 103 mature formation months, at most 17 nominal six-month blocks (Supplementary Figs. S4 and S5). These adaptations change the horizon, sorting and portfolio construction of the published strategies [@du_etal_2022; @wahal_yavuz_2013].

### Recorded popularity, participation and concept comovement

To examine the attention interpretation, we turn from concept-member selection to Wind’s stock-level popularity measures and compare their temporal relationships with returns and trading participation, then examine concept comovement before and after popularity is recorded. The three fields have distinct profiles (Fig. 6): watchlist popularity is persistent, stickiness decays, and total-popularity innovations show no corrected day-1 or day-3 endpoint response. The endpoint coefficients in Supplementary Section S12 describe persistence of the recorded fields.

[FIGURE_6]

The joint temporal criterion is met by eight of 72 attention–trading–horizon combinations, all involving turnover or trading value at 3–20 days (Fig. 6). It requires associations in both directions, lead–lag asymmetry, and within-stratum stock-identity permutations and 60-trading-day historical-misalignment controls. The passing associations have different signs across measures and horizons. None of the return combinations meets this joint criterion, although three one-way attention-to-return paths pass their familywise test: S at one and three days is negative, and T at three days is positive. Supplementary Section S14 reports the full path family and the sensitivity to signal-day trading controls.

The popularity–return comparison also involves two specification choices: the observations used for adjustment and the timing of control measurement. We compare four specifications on the same γ/H3 sample of 217,831 stock-days and 185 dates. Starting from the original role-specific fitting samples, the comparisons align auxiliary support and outcome-label maturity, remove four concept controls, and move the remaining 22 characteristics and industry to records strictly before the attention observation. No coefficient or paired specification difference passes its diagnostic Holm family: 36 coefficient tests and 27 contrast tests cover three declared HAC lengths (Supplementary Section S13.1, Fig. S12 and Table S12). These comparisons assess the sensitivity of conditional popularity–return associations to specification choices, rather than testing specification equivalence or decomposing the FW selection spread.

Concept comovement provides another temporal diagnostic: if attention drives concept-level trading, increases in popularity should be followed by stronger comovement between stocks and their concept peers. The analysis uses 123,530 complete stock-days across 202 dates, with industry benchmarks estimated from a disjoint stock subset and the focal stock excluded from its concept peers. On the same observations, mean correlations with real peers are 0.19861 in the earlier window and 0.19796 in the future window; the pseudo-network means are 0.04782 and 0.04983 (Fig. 7).

The pseudo network preserves coarse strata, and approximately 99.1% of nodes are exchangeable, but it fails the required paired size, degree and liquidity matching on all 202 dates. Its contrasts therefore remain descriptive diagnostics (Supplementary Section S15). The observed comovement precedes as well as follows the popularity measurement, leaving its source unresolved.

[FIGURE_7]

### Retrospective account simulations

The two-input idea is also evaluated in 48 retrospective account simulations from 1 December 2016 to 31 August 2026 (Fig. 8 and Supplementary Tables S13 and S14). The framework simulates trading at 09:55 through staggered sub-portfolios, with concept-return and ranking details that differ from the research construction. Its post-selection filter uses full-session information from the execution day, unavailable at 09:55; these accounts therefore describe retrospective allocation outcomes rather than implementable trading performance (Supplementary Section S17). In the untimed configuration closest to the reference, with three-day holding, F5/W10 and Pearson synchronization, the Top10 account grows to 3.92 times its initial capital: a 15.0% annual compound return after configured commissions and stamp duty. Maximum drawdown is −42.1%, from April 2017 to October 2018. Annual fees amount to about 8.0% of average assets, with annualized two-way turnover of about 129 times average assets. The simulations do not include slippage, market impact or capacity constraints.

Across the 48 archived simulations, compound annual returns range from 12.7% to 33.5% and maximum drawdowns from −59.6% to −14.6%. Paired configurations show how market-breadth timing rules change that risk profile. The one-level and two-level rules use observations available before 09:45 and are applied before the 09:55 trades. They raise annual compound returns by paired medians of 4.1 and 7.3 percentage points, and improve maximum drawdowns by 22.2 and 24.4 percentage points. Zero target-exposure signals occur on approximately 40%–45% of signal dates for scheduled sub-portfolios, while other sub-portfolios may remain invested. These paired comparisons hold the remaining archived framework fixed. Supplementary Section S17 reports the effect of removing the full-session exclusion across the same 48 configurations.

[FIGURE_8]

## Discussion

The member-selection result clarifies the relation between concept information and short-horizon returns. Existing work links concepts to constituent returns and stock momentum to comovement [@du_etal_2022; @wahal_yavuz_2013]. Here synchronization improves strength-only selection, while alternative joint rankings select nearly the same stocks. When synchronization already enters the fixed prediction model, the additional strength-related block has no confirmed increment. The information comparison, rather than the multiplication formula, is the substantive result: the value of combining the inputs depends on the information already included in the benchmark rule.

Theme-based attention is one candidate explanation for this return pattern. Style-based trading generates comovement and momentum in Barberis and Shleifer’s model [@barberis_shleifer_2003], and stock-search attention is associated with short-run price increases and subsequent reversal in Da et al. [@da_engelberg_gao_2011]. Our concept-level profiles have a related temporal shape across different aggregation levels. However, common thematic news, shared risk and correlated trading demand can also produce return alignment. Wind popularity supplies individual-stock measurements: the joint temporal comparisons support particular participation relationships, while the common-support and control-timing diagnostics examine conditional popularity–return associations in a separate panel. Distinguishing these interpretations requires variation in thematic exposure that is independent of common news and return-driven changes in popularity. Exogenous variation in news positioning, as studied by Fedyk [@fedyk_2024], provides one design precedent.

The reconstructed data define the historical scope of these comparisons. Concept and industry records lack certified first-publication and revision histories; accounting fields are excluded because their release chain is unverified. The research pool follows data availability, with survivor coverage and non-random endpoint missingness unresolved. Missing-outcome bounds remain positive under a restricted return range and include zero when that range widens. Supplementary Section S1 gives these bounds, the source corrections and the common 31 August 2026 cutoff. Establishing what information was actually available to investors at the time requires certified records of first publication and subsequent revisions.

The ranking uses no return-fitted weights, and rolling training enforces label maturity. However, later comparison designs were informed by earlier historical results, so the reported multiplicity adjustments apply within declared comparison families and do not account for the full sequence of earlier specification choices [@harvey_liu_zhu_2016; @white_2000]. A protocol fixed before new observations arrive would provide a prospective test of persistence. Extending this test to implementable returns requires target formation based on decision-time information and explicit market-impact assumptions. Aligning the account rules with the research lists would connect that evaluation directly to the reference selection.

## Conclusion

Starting from intertemporal relationships in concept returns, this study examines member selection within overlapping concepts. In this historical A-share sample, stocks aligned with recently strong concepts subsequently earn higher mean short-horizon gross returns. This return continuation is what we call concept momentum: its spread is positive across all 108 parameter combinations and persists after industry–size matching. Synchronization adds to strength-only selection, while several simple joint rankings select nearly the same stocks, indicating that the gain comes from the information in synchronization rather than a particular combination formula. The full concept-feature block improves conventional-model predictions in some ranking and selection tasks, whereas the return increment for the reference selection task is not confirmed. Some temporal associations of recorded popularity with turnover and trading value meet the joint criteria, but neither the adjusted conditional popularity–return associations nor their differences across fitting-sample and control-timing specifications survive correction. Identifying whether investor attention accounts for the member-selection return pattern requires exogenous variation in thematic exposure.

## Methods

### Data provenance

The price inventory used for the analysis contains 5,907 files and 10,447,754 stock-day records through 31 August 2026. Eligibility depends on historical inputs rather than future outcome availability. Supplementary Section S1 records the source corrections and corresponding recomputations. The archived covariate panel supports recomputation of conditional-attention results (Supplementary Section S1).

Different experiments impose additional coverage and historical-input requirements. The attention and mechanism analyses use the common 31 August cutoff; the sampled conditional-attention panel ends on 27 August 2026. Market-calendar formation windows are used for the main research scores. Earlier observation-row sensitivities retain their stated windows, which may span more market days when observations are missing (Supplementary Sections S3 and S4.1).

### Concept scores and aggregation

For each stock–concept membership, concept returns exclude the focal stock and require at least one valid peer. F compounds returns over $L_F$ market days, requiring all observations and retaining the daily floor of −0.999. W is the Pearson correlation between the focal-stock returns and untruncated peer returns over $L_W$ market days. It requires non-zero variation and the declared number of valid pairs. F5/W10 requires five complete concept-return observations and at least five paired returns within ten market days.

Each quantity is ranked across finite membership records on that date, using the minimum tied rank divided by its finite-record count. Memberships valid for both ranks are then retained. A stock with several memberships contributes several records to this ranking universe. With f and w denoting the resulting ranks, the stock-level scores are

$$
\begin{aligned} F^{\mathrm{only}}_{i,t}&=\operatorname{mean}_c f_{ic,t},\\ W^{\mathrm{only}}_{i,t}&=\operatorname{mean}_c w_{ic,t},\\ FW_{i,t}&=\operatorname{mean}_c(f_{ic,t}w_{ic,t}),\\ A_{i,t}&=\tfrac12(F^{\mathrm{only}}_{i,t}+W^{\mathrm{only}}_{i,t}),\\ P_{i,t}&=F^{\mathrm{only}}_{i,t}W^{\mathrm{only}}_{i,t}.\end{aligned}
$$

The mean and population covariance use the same memberships, giving FW = P + Cov(f, w). This identity isolates the within-stock alignment retained by multiplying before averaging. Integer rank sums and products are compared as exact ratios; ascending stock code resolves mathematical ties. Supplementary Section S3.1 records this numerical refinement and its timing. The six-pair sensitivity reconstructs all scores from the corrected sources under common within-pair eligibility; its windows and support rules appear in Supplementary Section S2.1.

### Outcomes and comparison designs

Daily returns are close/previous close − 1. Adjusted closes compound these returns, and adjusted opens equal adjusted closes multiplied by the same-day open/close ratio. The H-day outcome is adjusted open(t + H + 1)/adjusted open(t + 1) − 1. Both endpoints must be finite and positive on the prescribed market dates; the next available observation cannot replace a missing endpoint. Closing-return diagnostics use close(t + H)/close(t) − 1. Table 1 defines the four timing cells, and Supplementary Section S4 reports their paired contrasts.

Top-N lists are fixed before outcome linkage. Primary list returns average observed outcomes; a separate sensitivity retains denominator N and sets missing returns to zero. The latter does not model suspension or delisting proceeds or define an economic lower bound. Pool-relative spreads subtract the equally weighted research-pool return. Direct contrasts compare frozen lists on the same date. All research returns are gross, complete-H cohort returns, averaged equally across formation dates. Overlapping daily cohorts do not constitute a rebalanced wealth series, and their means are not daily trading profits.

Industry–size comparisons preserve the frozen lists and match their composition using weighted group means. The full daily pool is divided into ten capitalization groups, with all groups retained and selected stocks included in their comparison groups. Industry-only, size-only and joint controls are evaluated alongside 1,000 within-group random portfolios. Actual lists are never replenished for missing outcomes. Exact grouping, paired-date eligibility and missingness rules are given in Supplementary Section S2.4; temporal checks appear in Section S6.

Daily concept-momentum and comovement benchmarks use common historical eligibility and identical endpoints. A monthly mapping uses independent slope-tercile and momentum-decile sorts. These comparisons draw on concept-momentum and comovement studies [@du_etal_2022; @wahal_yavuz_2013], with different market, category and return conventions from the published strategies. Supplementary Section S7 specifies their construction, including the monthly skipped interval and six-month holding period.

### Conditional prediction

To test whether concept features add to conventional information, conventional models contain 19 price, volatility, trading and size characteristics plus a missingness feature. Enhanced models add FW, F-only and W-only, testing the three concept features jointly. Characteristics become within-pool average-tie percentiles minus 0.5, with missing values set to zero. Neither industry nor accounting characteristics enter training. The complete feature definitions and fixed learner settings are given in Supplementary Section S2.2.

Ridge regression and histogram gradient boosting predict each year from 2020 to 2026 using the preceding three calendar years. Training retains every twentieth market-calendar date, with outcomes matured by the preceding year-end and at least two signal years represented. Each fit consequently uses 23–37 dates rather than every trading day. Paired ordinary and enhanced models use identical rows and settings. Training targets are clipped to [−0.8, 2.0]; evaluation returns are not clipped. Seven years, eight horizons, two learners and two feature sets produce 224 fits.

Predictions cover the common pool, with Top100 lists saved before outcome linkage and Top20 taken as their prefixes. We evaluate date-weighted returns and daily Spearman rank information coefficients (ICs). Industry-conditioned IC correlates prediction and outcome ranks after subtracting industry mean ranks on the same valid subset, without reranking or retraining. Industry labels are not certified point-in-time. A separate H3 sensitivity uses every eligible training date, retaining features, settings and endpoints, and produces 28 additional fits (Supplementary Section S9).

A targeted H3/Top20 comparison holds W in both feature sets and adds F and FW jointly, using all eligible training dates and the fixed boosting settings (14 fits). Maturity and selection rules are unchanged; the stress-state definition and six-test family are specified in Supplementary Section S10. A separate four-model comparison adds W to conventional characteristics and F, with and without past market beta and residual volatility. The risk measures use the preceding 60 trading days, excluding the signal day; seven annual fits of each model give 28 fits. Section S11 defines the three paired contrasts and their joint six-test family.

### Statistical inference

Inference uses paired daily contrasts to retain the common market exposure and overlap between selected lists. Circular moving-block resampling preserves dependence across consecutive dates. The baseline block length is L = max(20, H), with pointwise 95% percentile intervals. Two-sided P values compare the observed absolute mean with centred bootstrap absolute means and use a plus-one correction. Declared block lengths and longer-block sensitivities assess the dependence assumptions (Supplementary Section S2.3). Comparisons with floor(n/L) < 20 remain descriptive and enter correction with P = 1 as a placeholder, not an estimated P value. Monthly comparisons have at most 17 nominal six-month blocks and receive no inferential intervals or P values.

Holm adjustment applies within the declared families in Supplementary Table S3. Resampling counts, module-specific block choices and longer-block sensitivities are consolidated in Supplementary Section S2.3. The prediction family comprises 80 primary contrasts and 16 industry-conditioned IC contrasts. Parameter sensitivity uses separate 540-test complete-outcome and missing-as-zero families. The five timing contrasts share a Holm20 family across two missingness conventions and two block lengths. Adequate temporal support is a prerequisite for formal inference, rather than an indication of significance after correction.

The longer-block analyses were added after the main results. The 192-comparison aggregation grid was fixed before execution after pilot results had been inspected. These comparisons use previously examined history; their family-level corrections do not provide an untouched holdout or cover the full research sequence. Calendar-year, leave-one-year-out and non-overlapping H3-offset comparisons are descriptive; fixed-period tests retain their separate families (Supplementary Section S6).

### Attention and participation

To distinguish persistent popularity from its changes, attention endpoints use T innovations, five-calendar-day S changes and weekday-standardized K, expressed as contemporaneous cross-sectional ranks. The future attention endpoint is t + H + 1; endpoint/current coefficient ratios use identical support. Supplementary Section S12 specifies measurement and reference-set sensitivities. Endpoint inference uses 1,000 circular block draws of 120 trading dates and a Holm24 family; H250 is descriptive with P = 1. HAC estimates are secondary diagnostics.

Expanding-time attention models fit on earlier observations, with outcome labels required to mature before each test fold. AIPW combines propensity weighting with outcome regression [@bang_robins_2005], and Tree-DML uses flexible regressions to residualize exposures and outcomes [@chernozhukov_etal_2018]. The retained Tree-DML models use role-specific finite and trimmed fitting samples; their original date-HAC intervals and adjusted P values are therefore reported as implementation diagnostics rather than standard orthogonal-score inference. Full-period preprocessing, past-only calibration and the S-coverage window remain separate protocols (Supplementary Figs. S10 and S11 and Table S11). Section S13 specifies the exposure definitions, support and overlap gates, and the timing of controls and concept-peer construction.

The focused γ/H3 diagnostic holds evaluation observations fixed across four specifications and examines fitting support, outcome-label purging, concept controls and the date of the same conventional controls. Earlier historical results were already known when this diagnostic was specified. All 64 nuisance fits use the existing learner settings and four evaluation folds. Date-aggregated score contributions supply diagnostic coefficient and paired-difference intervals at three fixed HAC lengths; Section S13.1 defines the 36- and 27-test families. These conditional-association analyses do not establish a well-defined exogenous exposure or conditional exchangeability.

Participation analyses use trading endpoints at t + H. Their joint criterion requires associations in both directions, temporal asymmetry and negative-control checks. The max-|t| reference covers all 144 directional paths. Prediction of next-session popularity-list visibility provides a related-measure comparison. Supplementary Section S14 reports these designs and the excluded order-size module, whose inputs failed reconciliation.

### Attention and concept comovement

Concept memberships are taken from the latest stored snapshot strictly preceding the attention observation. Peer returns exclude the focal stock, require at least five observed peers per concept and average eligible concepts equally. Correlations require 16 paired observations within each 20-trading-day window. Even-numbered stock codes estimate industry benchmarks and odd-numbered codes supply focal observations, removing indirect focal inclusion through demeaning. This split was introduced after an initial conventional-demeaning analysis, which remains archived.

To assess whether comovement depends on the observed links, fifty fixed-seed membership permutations within industry, size-tercile and degree-tercile strata define one pseudo-network benchmark. Regressions include T/S/K ranks, industry indicators and observed slow-state, momentum, risk, size and liquidity controls. Matching requires at least 60% exchangeable nodes and mean absolute standardized paired distances no greater than 0.10. Failure withholds formal inference while retaining diagnostic estimates. Supplementary Sections S15, S16 and S2.4 report the estimates, gate checks and regression details. The final sampled signal is 30 July 2026; its complete outcome ends on 27 August, within the common cutoff.

### Account simulation design

The separate event-driven framework, select-stock-xhzheng 2.5.0, implements the same two-input idea with three differences: concept returns include the focal stock, F ranks concepts, and minimum-rank ties can extend selection beyond the nominal list size. Membership records and ranking inputs follow the γ information boundary, with execution simulated before 09:55 on the next market day. A separate post-selection filter uses the execution day’s full-session one-price upper-limit indicator; this information is unavailable at 09:55 and can change target holdings and sales of existing positions. These backtests therefore evaluate implementation variants rather than execution of the frozen research lists.

All 48 runs cross three- or five-day holding periods, F5/W10 or F10/W20, Pearson or concordance correlation, Top5 or Top10, and three timing settings. Runs begin with CNY 30,000,000 on 1 December 2016 (8 December for F10/W20), exclude Beijing Stock Exchange listings and end on 31 August 2026. Equally capitalized staggered sub-portfolios implement the holding periods. Upper-limit-blocked buys are retried and cancelled if still unfilled at the close; lower-limit-blocked positions are retained and reconsidered. Missing closes are carried forward for valuation, without establishing executable prices.

Commission is 1.2 basis points and stamp duty 0.1% on sales. Slippage, market impact and capacity are not modelled. Daily net values determine annualized returns, maximum drawdowns, yearly returns, invested shares, fees and commission-implied two-way turnover. All runs use the corrected frozen inputs; Supplementary Section S17 gives the execution and timing rules. CSI 300, CSI 500 and CSI 1000 price indices exclude dividends and serve as descriptive references. The backtest comparisons carry no formal inference.

## Abbreviations {-}

A, arithmetic combination of the stock-level strength and synchronization scores; AIPW, augmented inverse-probability weighting; CAGR, compound annual growth rate; CCC, concordance correlation coefficient; CSI, China Securities Index; F, concept strength; FW, membership-level product score of concept strength and synchronization; H, holding horizon in trading days; HAC, heteroskedasticity- and autocorrelation-consistent; IC, information coefficient; K, attention stickiness; N, selection size; P, product of the stock-level strength and synchronization scores; PR-AUC, area under the precision–recall curve; S, watchlist popularity; T, total popularity; Tree-DML, tree-based double/debiased machine learning; W, stock–concept synchronization.

## Data availability {-}

The training code and experimental-results data for this study are publicly available at https://github.com/xinhao-zheng/concept-momentum-and-investor-attention, together with a concept-membership archive and factor and strategy modules for the trading implementation rules. These modules require the host backtesting framework and separate price inputs; their rules can also be extracted for independent use. Wind popularity measures and other raw vendor data, including licensed price and industry data, are not included.

## Declaration of competing interest {-}

The author trades Chinese A-share equities in a personal account, including with rules related to those studied here, and declares no other competing interests.

## Funding {-}

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process {-}

During the preparation of this work, the author used OpenAI Codex and Anthropic Claude in the Cursor editor to review analysis code, run statistical calculations, check the literature, and draft and proofread the manuscript. After using these tools, the author reviewed and edited the content as needed, verified the reported results against the saved computation records and takes full responsibility for the content of the published article.
