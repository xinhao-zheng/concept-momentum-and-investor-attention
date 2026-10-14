# Supplementary information

Concept momentum and investor attention: Evidence from China’s A-share market

## Historical availability

Research observations and prescribed outcome endpoints are restricted to 31 August 2026 or earlier. The pre-correction price inventory contains 10,447,809 stock-days; the membership and research-calendar inventories contain 82,392,428 records and 2,373 dates, respectively. External checks identified four erroneous closes on 9 June 2025: sh603207, 29.64 to 31.14; sh603341, 38.70 to 39.19; sz301539, 18.33 to 18.43; and sz301662, 79.01 to 80.01. The correct ex-dividend previous-close fields on 10 June are retained. A subsequent review of positive-volume records with missing execution quotes identified a security-code transition, stale daily records and six observations with recoverable quotes. The repaired source links the former code 300114 to 302132, removes 55 invalid economic stock-days and leaves 10,447,754 records in the price inventory. The invalid records comprise one copied IPO record on a suspended date for sh603336 and 54 stale observations after the last trading date under their recorded codes.

All 48 implementation backtests use the same frozen corrected sources, retaining their original protocols and execution settings. Their configurations, account paths and independently compiled summaries are bound to the current report. They remain implementation variants distinct from the research Top20 lists; Section S17 specifies the differences.

Financial files contain publication-date fields, but reporting periods, announcements, revisions and collection times have not been linked into a verified release history. They are therefore not used as strict point-in-time accounting controls. A researcher-defined selection threshold is not a verified exogenous display discontinuity. Fedyk studies exogenous variation in prominent news positioning [@fedyk_2024]. The present records supply no comparable source of exogenous exposure.

A complementary sensitivity assigns each missing stock–date return a support interval [−1, U]. The same unknown return enters both the portfolio and any comparison group that contains that stock; it is not varied independently on the two sides. Frozen selections and intended denominators give a linear spread in these unknown returns. Minimizing and maximizing that expression produce support bounds for the equal-date-weighted mean. The pool comparison covers 2,349 mature dates; joint industry–size matching uses its 2,346 paired complete-comparison dates.

With U = 0.20, the pool-relative bounds are [0.0992, 0.5224] percentage points and the joint-matched bounds are [0.0141, 0.3779]. With U = 1.00, they become [−0.0715, 0.6339] and [−0.1360, 0.4701], respectively. These are bounds conditional on the assumed return support, not confidence intervals. The upper endpoints are assumptions, not universal return limits; widening the support admits zero and negative mean spreads.

## Detailed research protocols

This section records execution details supporting the main Methods. Existing sections give benchmark construction (S7), attention measurement and temporal models (S12–S13), data provenance (S1), participation analyses (S14), timing comparisons (S4) and implementation backtests (S17).

### Score windows and comparison grids

The six $(L_F,L_W)$ pairs are (3,5), (3,20), (5,10), (10,20), (20,10) and (20,50). Strength requires all $L_F$ market-day observations. Synchronization requires at least 5, 5, 10 and 25 valid pairs for $L_W$ = 5, 10, 20 and 50, respectively. Each pair's historically eligible pool is intersected with the original common research keys. FW, F-only, W-only, A and P share eligibility and endpoints within a pair. Selection is fixed before future returns are joined. H = 3, 5, 10 and N = 5, 10, 20, 30, 50, 100 define 108 cells, each with five FW-minus-baseline contrasts. The complete-outcome and missing-as-zero versions receive separate 540-test corrections, without replacing selected stocks.

The earlier eight-horizon comparisons use N = 5, 20 and 100 and H = 1, 3, 5, 10, 20, 60, 120 and 250 market days. Common historical eligibility and identical return endpoints are retained within comparisons. The original common pool requires both β and γ scores under market-calendar and observation-row formation conventions. Section S3 reports the γ-only support sensitivity. Exact arithmetic governs rank aggregates and selection, while prices and returns retain double precision.

### Prediction features and fixed learners

The 19 ordinary characteristics are 1/3/5/10/20/60/120/250-day momentum; negative five-day momentum; 20/60-day sample return standard deviations; current and 20-day mean turnover value; ratios of five- to twenty-day and twenty- to sixty-day mean turnover value, minus one; 20-day mean absolute return/turnover value (Amihud illiquidity); free-float and total capitalization; and the 20-day mean turnover value/free-float capitalization ratio. Market-calendar windows require exact momentum endpoints and complete rolling observations. Non-positive turnover value or capitalization is missing. After percentile transformation, negative five-day momentum reverses the ordering of five-day momentum; it remains in the original feature list.

Every characteristic becomes its within-pool average-tie percentile minus 0.5, with missing values set to zero. Missing count/19 − 0.5 supplies a twentieth input. The enhanced model transforms γ/calendar FW, F-only and W-only identically, giving 23 inputs. Neither model includes industry or accounting characteristics. All rows receive equal fitting weight. Training dates have global market-calendar indices divisible by 20, and matured outcomes must leave at least two signal years before the prediction year. The main fits therefore contain 23–37 training dates, including 23–24 at H250.

Ridge regression includes an intercept and penalty 100. Histogram gradient boosting uses learning rate 0.06, 40 iterations, at most 15 leaves, at least 100 observations per leaf, L2 penalty 10, 63 bins, no early stopping and seed 20260923. Training targets are clipped to [−0.8, 2.0]; evaluation returns are unchanged. Ordinary and enhanced fits use identical training rows. The annual three-year training window, feature construction and learner settings remain fixed in the full-training-date H3 sensitivity (Section S9).

Industry-conditioned IC first ranks predictions and returns globally within the same valid subset, subtracts industry mean ranks and correlates the residual ranks. It requires known industry and at least five valid stocks per industry; these conditions exclude no additional observations here. Industry labels are not certified point-in-time, and conditioning does not retrain the models.

### Resampling and multiplicity

For the paired cohort-return analyses below, circular moving-block draws are shared across paired statistics. Intervals are pointwise bootstrap percentiles. Two-sided P values compare absolute observed means with absolute centred bootstrap means and use a plus-one correction. Fewer than 20 nominal complete blocks, floor(n/L) < 20, withholds formal inference while retaining P = 1 in the correction family. Finite-replicate resolution is retained when computing adjusted P values. Automatic block-length selectors provide a data-dependent alternative [@politis_white_2004; @patton_politis_white_2009]. The present analyses instead use declared lengths and longer-block sensitivities, as consolidated below; Supplementary Table S3 identifies the complete family compositions.

| Analysis | Bootstrap replicates | Main block length | Family and longer-block scope |
| --- | ---: | --- | --- |
| Components, aggregation and daily benchmarks | 100,000 | max(20,H) | Separate declared families; daily-benchmark longer-block checks cover the four γ/Top20/H3 contrasts |
| Conditional prediction | 10,000 | max(20,H) | Holm96; the same family repeats with base blocks of 40 and 60 days |
| Six-pair parameter sensitivity | 100,000 | 20 dates | Separate Holm540 families by missingness; 40- and 60-date checks cover six pairs × five contrasts at H3/Top20, with Holm30 separately at each length |
| Paired membership and return timing | 20,000 | 20 and 60 dates | One Holm20 covers five contrasts × two missingness conventions × two block lengths |
| Industry and size matching | 10,000 | 20 dates | Holm9 covers three list sizes × three matching schemes |
| Fixed-period H3 contrasts | 20,000 | 20, 40 and 60 dates | A separate Holm4 at each length covers two periods × two benchmarks |
| Full-training-date H3 sensitivity | 10,000 | 20 and 60 dates | One Holm24 covers twelve contrasts × two block lengths |

The prediction family contains eight horizons × two learners × [pool IC + two selection sizes × two missingness conventions] = 80 primary contrasts. Sixteen industry-conditioned IC contrasts bring the total to 96. The full-training-date H3 family instead includes raw IC, industry-conditioned IC, Top20 and Top100 under both missingness conventions for each model. Its protocol was frozen before that computation. Monthly mappings have at most 17 nominal six-month blocks and receive no inferential P values or intervals. Attention and comovement analyses use their distinct inference procedures in Sections S12–S13, S14 and S2.4.

### Matched controls and comovement regressions

Industry–size matching intersects the frozen research pool with known same-day industry and positive finite float capitalization, retaining all 8,835,000 observations. Corrected-price, double-precision F5/W10 scores are ordered descending, with ascending stock codes resolving ties. Top5 and Top10 are prefixes of the frozen Top20 list. Capitalization uses its stored single-precision values. Stocks are ordered by capitalization ascending and code ascending; zero-based rank i among n candidates receives size group floor(10i/n)+1. All ten groups are retained.

The analytic comparator weights each group's mean available H3 return by k_g/N, where k_g is the selected count in that group. Selected stocks remain in the groups. Each of 1,000 fixed-seed random repetitions draws k_g stocks uniformly without replacement from each group, with the actual selected stocks still eligible. Draws depend on candidate identities and groups, not outcomes. Random-portfolio period-mean distributions use the corresponding paired dates and are conditional references rather than time-series confidence intervals.

H3 outcomes join exact adjusted opens at t + 1 and t + 4, with endpoints no later than 31 August 2026. A primary date requires the observed actual-list mean and every selected group's mean. Joint matching retains 2,348 Top5 dates and 2,346 Top10/Top20 dates, compared with 2,349 under industry-only and size-only matching. Missing selected outcomes number 12, 31 and 66, respectively. A descriptive sensitivity fills missing stock outcomes with zero in both actual lists and groups, retains N and uses all 2,349 mature dates.

Temporal comparisons recompute scores, lists, adjusted-open outcomes and analytic group means from corrected prices. Calendar-year, leave-one-year-out and three disjoint H3 formation-calendar offsets are descriptive. The fixed periods are 2017–2021 and 2022–31 August 2026, with four paired contrasts across pool and joint-group benchmarks. The corresponding block procedures are listed above; gaps, support and results appear in Section S6.

For concept-comovement regressions, even-numbered stock codes estimate industry benchmarks with at least 20 observed reference stocks per industry-day; odd-numbered codes supply focal stocks. Daily cross-sectional regressions jointly include T/S/K ranks, industry indicators and slow-state, momentum, risk, size and liquidity controls timestamped no later than attention observation. Fifty fixed-seed membership permutations within industry, size-tercile and degree-tercile strata define a single pseudo-network benchmark. Coefficients are averaged equally across dates. The 2,000 shared circular draws use blocks of eight sampled dates, corresponding to 40–45 market trading days; 202 dates supply 25 nominal complete blocks. All nine hypotheses remain in their Holm family. The required mean absolute standardized paired distance is no greater than 0.10 for each of size, degree and liquidity, alongside at least 60% exchangeable nodes. Failure closes formal inference without deleting diagnostic estimates (Sections S15 and S16).

## Common support and score sensitivity

The score allows separate formation windows $L_F$ and $L_W$. F5/W10 denotes the reference pair $(5,10)$ carried over from the original design and retained for component comparisons, rolling prediction models, and H3 timing and matched-control analyses. A separate parameter sensitivity crosses six window pairs, three horizons and six list sizes. The full-training-date check changes training density at H3 with F5/W10 features fixed. These scopes distinguish the analyses throughout the supplement. All observations end by 31 August 2026.

The original common pool requires valid β and γ scores under both formation-window conventions (market-calendar days and valid observation rows), together with valid price benchmarks. Thus γ eligibility partly depends on β validity. Requiring valid γ scores under both window conventions and the necessary historical prices adds 8,045 stock-days on 197 dates. Mean Top20 overlap is 99.84%; FW-minus-F-only and FW-minus-W-only return differences on this γ support are approximately 0.2271 and 0.1555 percentage points. The selected lists therefore remain close to the originals after eligibility is relaxed. List overlaps use all 2,353 signal dates, while H3 return contrasts use 2,349 mature dates (Supplementary Table S2).

The aggregation comparisons retain membership ranks and selected-list sizes (Supplementary Fig. S3). The population-covariance identity was checked on four dates under both β and γ, with maximum error below $10^{-12}$. The direct W-only-minus-F-only mean was approximately 0.0723 percentage points, with pointwise interval [−0.0956, 0.2356]. Annual FW-minus-W-only means were approximately −0.0807 and −0.0038 percentage points in 2022 and 2025. The annual values describe variation within the same historical sample.

Assigning zero to missing future outcomes at the original selection denominators yields 64 corrected pool-relative rejections and 80 corrected FW-minus-strength-only rejections across the 108-cell sensitivity grid. No FW-minus-synchronization-only contrast passes the same correction. These results preserve the frozen lists.

FW minus the arithmetic combination A is positive in 75 of 108 grid cells, and FW minus the product of stock-level means P is positive in 55 of 108. Neither set contains a rejection after Holm540 correction. The separate eight-horizon comparison has no corrected aggregation advantage among its 144 temporally supported tests (Supplementary Fig. S3).

### Parameter-score precision reconciliation

All six specifications are recomputed from the corrected sources in Section S1. The final implementation aggregates integer membership ranks and compares the resulting ratios by integer cross-multiplication, with stock code resolving exact ties. This refinement followed a reproducibility check in which floating-point summation changed the order of mathematically equal component scores. It retains the economic definitions, eligibility, seeds and inference families. Reported F5/W10 FW lists are unchanged from the preceding corrected-source version; affected component, aggregation, parameter and predictive comparisons are recomputed.

## A paired separation of membership dates and return clocks

The paired comparison fixes F5/W10, H3 and Top20 and uses corrected-price market-calendar scores through 31 August 2026. The historical-input pool remains defined by valid scores under both membership conventions and both earlier window conventions, without using future outcome availability. It contains 8,835,000 stock–date records on 2,353 signal dates. Daily β and γ lists are fixed by descending score and ascending stock code before future outcomes are linked. Earlier historical results had already been examined; this computational ordering does not create original preregistration or an untouched holdout.

The synchronous β membership snapshot is dated no later than signal day t; the γ snapshot is dated no later than the preceding market day. Both scores use price information through t close. α and β share the synchronous list, while γ and δ share the lagged list. α and δ use adjusted close(t) to close(t+3); β and γ use adjusted open(t+1) to open(t+4). All endpoints are exact global-market-calendar joins for the same stock code, retaining missing values rather than shifting to the next observed row or substituting another selected stock. The price cutoff is 31 August 2026. The close-based cells are diagnostic references: a score using t close does not imply a feasible fill at that same close.

All four cells share the same eligible-pool identities on each date. Each selected-list return is compared with that pool evaluated under the corresponding close or open clock. The 2,349 common mature dates span 22 December 2016 to 25 August 2026, with no calendar gaps and no additional date excluded by the four-cell estimability requirement. There are 8,814,334 mature pool stock–date records. Of 46,980 intended observations per selected list, α, β, γ and δ have 45, 68, 66 and 40 missing outcomes, respectively. Pool outcomes are missing on 9,796 close-clock and 14,234 open-clock records. Complete-observation means therefore have different observed-stock denominators across clocks. A second convention sets missing outcomes to zero while retaining the intended selected and pool denominators; it uses the same dates and neither models delisting proceeds nor guarantees an economic lower bound.

Five paired daily spread contrasts isolate the two membership comparisons (β−γ and α−δ), two clock comparisons (α−β and δ−γ), and their interaction, (α−β)−(δ−γ). The interaction also equals (α−δ)−(β−γ), an identity verified numerically. Equal-date means are evaluated using 20,000 paired circular moving-block replicates at L=20 and L=60, with bootstrap starts shared across contrasts and missingness conventions within each block length. Pointwise 95% intervals use percentiles. Two-sided P values compare the observed absolute mean with centred bootstrap absolute means and include a plus-one correction. One Holm20 family covers five contrasts, two missingness conventions and two block lengths. The two lengths provide 117 and 39 nominal complete blocks; these counts are support criteria, not a claim of independent blocks.

Under complete observations and L=20, α−β is 0.040421 percentage points (pointwise 95% interval 0.018988–0.061803; Holm20 P=0.005000), and δ−γ is 0.038018 (0.016461–0.059703; P=0.007650). The corresponding L=60 adjusted P values are 0.007650 and 0.016249. The missing-as-zero clock contrasts are 0.040988 and 0.038565 percentage points and also pass under both lengths. Both membership contrasts and the interaction remain unconfirmed throughout, with adjusted P=1. The eight passing rows in the twenty-row family represent two related clock contrasts evaluated under two missingness and two dependence assumptions. They do not represent eight independent findings. Nonrejection of the membership contrasts is not an equivalence test.

The γ/open cell uses the same corrected-price score, list and outcome definitions as the main specification on all 2,349 mature dates. Its source manifest preserves the calculation version and paired endpoints. Earlier spot checks of three stock–date windows against the raw adjustment chain remain archived as checks of the previous snapshot. This experiment uses market-calendar scoring and differs from the observation-row sensitivity below. The concept-level analysis in Supplementary Fig. S1 shares the γ information boundary while comparing concept portfolios rather than selected-stock returns.

### Observation-row formation-window sensitivity

A separate timing sensitivity constructs F5/W10 from valid stock–concept observation rows under the same β and γ membership conventions. Its common pool is the intersection of finite positive scores, fixed before outcome linkage. The grid covers eight horizons and Top20/Top30, with separate Holm80 families for complete observations and missing-as-zero returns. Outcomes use exact market-calendar endpoints no later than 31 August 2026; inference uses date-HAC lag max(H−1,5). At H3/Top20, the four cell spreads on 2,364 common dates from 1 December 2016 to 25 August 2026 are 0.3153, 0.2747, 0.2748 and 0.3135 percentage points for α, β, γ and δ. The α−β clock contrast is 0.04061 percentage points (pointwise 95% interval 0.01709 to 0.06412; Holm80 P=0.04071); δ−γ is 0.03866 (0.01535 to 0.06198; P=0.06458). The two membership contrasts and their interaction have adjusted P=1. With missing returns set to zero, the two clock contrasts are 0.04121 and 0.03926, with adjusted P=0.03386 and 0.05347, respectively. The clock-contrast point estimates are therefore similar under both membership conventions; under this broader correction, the synchronous-membership clock contrast passes, whereas the lagged-membership contrast does not. The formation window, pool, dependence estimator and correction family differ from the market-calendar comparison; these related analyses are reported separately.

## Relation to earlier analyses

Earlier work compared concept and industry rankings, then introduced stock–concept synchronization and distinguished entry into a candidate pool from additional ranking within it. Raw IC and standardized IC-z answer different questions; standardization does not remove overlapping memberships. Permutations, industry/capitalization adjustment and matching examine score relationships and competing explanations rather than exogenous treatments. Earlier entry-spread and attention event-matching analyses retained material imbalance. They provide background to the research sequence; the formal quantitative results reported here are drawn from the current analyses restricted to the stated cutoff.

An information coefficient computed against future returns becomes available only after those outcomes mature. Retrospectively selecting its peak cannot supply an ex ante timing signal. The implementation overlays instead use market breadth observed before execution (Section S17).

Later comparisons, developed after inspection of earlier results, address aggregation on common samples, attention measurement, daily benchmarks, conditional prediction, monthly mapping and timing. They estimate different quantities: selected-list returns, properties of rank-based proxies and covariate-adjusted return associations. Their samples and outcomes remain separate, and shared observations are treated as related evidence rather than independent replications.

Supplementary Fig. S1 compares concept-level return associations across formation and holding horizons under α, β and γ. On each market day u, the concept return is the equal-weight mean of finite member-stock returns, requiring at least one observed stock. α and β use the latest recorded membership snapshot dated no later than u; γ uses the latest snapshot dated no later than the preceding market day, u−1. Formation strength compounds F complete market-day concept returns through t in all three designs. Thus γ delays membership while retaining formation returns through t, as in the stock-level design. In α, the future outcome compounds daily concept returns over t+1 to t+H, with memberships updated on each day. In β, it averages individual-stock adjusted opening returns from t+1 to t+H+1 within the eligible concept basket fixed at t. The γ basket uses membership dated no later than t−1, retains members with an observed stock return at t and uses the same opening-return interval. The α–β comparison therefore changes portfolio construction as well as the return clock. These are concept portfolios without a focal stock; the leave-one-out construction applies to the stock-level selection tests.

Past formation eligibility is determined without future outcomes. The α index requires a valid concept return on every day of its H-day outcome. In β and γ, stocks lacking either prescribed opening endpoint have missing returns; the primary outcome averages observed returns within the basket fixed at the information date. Each design uses its own valid concepts and dates within an F/H cell, with at least 20 concepts and nonconstant ranks for each daily correlation. All source observations and prescribed endpoints are no later than 31 August 2026, and holding periods are not shortened. Supplementary Data S9 also reports results on the same concepts and dates across the three designs, a fixed-member closing-return sensitivity for α, and fixed-denominator missing-as-zero sensitivities for β and γ.

For each date t, we calculate the cross-concept Spearman correlation $\mathrm{IC}_t$ and denote its value clipped to [−0.999999, 0.999999] by $\widetilde{\mathrm{IC}}_t$. The scaled value is $z_t=\operatorname{atanh}(\widetilde{\mathrm{IC}}_t)\sqrt{N_t-3}$, where $N_t$ is the number of concepts entering that day's correlation. The figure reports equal-date means of $\mathrm{IC}_t$ and $z_t$. This count-based scaling is descriptive and does not account for dependence arising from overlapping memberships or repeated observations over time. Supplementary Data S9 reports all 486 signed values, the date support of each cell and the common-support sensitivity. Among the 25 cells with F and H no greater than ten days, mean raw IC is positive in every α cell and in 21 cells under each of β and γ. For the 12 cells with F of at least five days and H of 60 or 120 days, it is negative in 10 α cells and all 12 cells under each of β and γ. These counts describe direction rather than statistical significance.

## Reconstruction, annual variation and fixed-period sensitivity

The source-corrected analysis retains common-pool eligibility rules and joins all 8,814,334 mature stock–date records to exact H3 endpoints. Scores, selections, daily group means and paired spreads are recomputed under the same definitions. Across 2,349 mature dates, 66 of 46,980 intended Top20 outcomes remain missing. Earlier frozen-score and raw-source replay checks are archived separately from the current corrected-price estimates. Section S1 states the data-provenance scope of these computational checks.

Annual means are reported descriptively, with 2016 and 2026 explicitly identified as partial years (Supplementary Fig. S2 and Table S6). Pool-relative and joint industry–size spreads average 0.3364 and 0.1919 percentage points in 2017–2021, and 0.2270 and 0.1426 in 2022 through 31 August 2026. All four comparisons pass their respective Holm4 families with 20- and 40-date blocks; the later pool comparison is close to the significance threshold, with adjusted P values of 0.0343 and 0.0415. At block length 60, the later period has 18 nominal complete blocks, falling short of the sample-support requirement, so formal inference is not reported. Different block lengths do not supply independent findings. Dropping missing paired dates leaves two calendar gaps in the early joint-group series. Omitting any single calendar year leaves the pool-relative mean between 0.2496 and 0.3238 percentage points and the joint industry–size-matched mean between 0.1537 and 0.1911 percentage points.

A contemporaneous benchmark restricts the H3 Top20 comparisons to the common window of the attention analyses. The first eligible signal is 25 March 2022, after all three T/S/K fields become observable under the two-calendar-day attention lag; outcomes must mature by 31 August 2026. The resulting 1,072 dates end on 25 August 2026. FW exceeds the pool by 0.218 percentage points (pointwise 95% interval 0.010 to 0.440), F-only by 0.264 (0.083 to 0.444), W-only by 0.120 (−0.081 to 0.337), and joint industry–size peers by 0.131 (0.031 to 0.235), using 20,000 circular 20-date-block draws. Field coverage and information timing determine the window; these intervals are descriptive and enter no correction family.

The 2026 partial year contains 156 signal dates through 25 August. Its pool-relative and joint industry–size means are 0.6040 and 0.2783 percentage points, respectively (Table S6). The nine complete calendar years are unaffected by the final observation cutoff.

Selecting each of the three market-calendar offsets separately yields non-overlapping H3 holding intervals within an offset. Pool-relative means range from 0.2759 to 0.2955 percentage points; joint-group means range from 0.1573 to 0.1742. These are descriptive historical contrasts. Adjacent cohorts may still share broader market regimes, and offsets are not independent replications.

## Daily and monthly benchmark implementation

CM5 and CM20 compound leave-one-out concept returns over five and twenty market days, requiring at least 5 and 15 observations, respectively, and valid ten- and sixty-day comovement slopes. Compounding retains the −0.999 daily floor; slope estimation does not truncate returns. BC10 and BC60 average membership-level OLS slopes with intercepts, requiring at least 5 and 40 paired returns and demeaned sums of squares above $10^{-15}$ for both series. Unlike W10, these statistics are slopes rather than correlations. Order k = 1, …, n follows descending mean slope and ascending stock code. Groups are defined by floor[3(k − 1)/n]; the highest-slope group is prioritized, with descending five- or twenty-day stock momentum within groups. Its size always exceeds 100, so every Top-N list comes from that group. This is a slope-priority sort, not an independent three-by-ten sort.

The daily-benchmark bootstrap uses base seed 230923. Of 384 comparisons, 288 meet time support and 216 pass Holm correction; 96 long-horizon comparisons are descriptive. All point differences are positive. The four benchmarks were developed after earlier research and define the scope of this comparison. In their stricter common pool, main-specification FW gross return is 0.4027% and its pool-relative spread is 0.2814 percentage points. Missing future returns retain their positions in the fixed selections, and complete-endpoint means use the observed outcomes in this pool.

For formation month m, monthly stock momentum uses complete months m − 5 to m, requiring every market-day return to be valid and above −1. Membership-level OLS uses daily observations in months m − 2 to m, at least 20 valid pairs and both demeaned sums of squares above $10^{-15}$. Slopes are averaged arithmetically within stocks. Eligible memberships satisfy γ at formation; historical concept returns use each historical day's membership rather than backfilled formation-date membership. A membership need not exist on every estimation day. Independent slope terciles and momentum deciles use deterministic code-based ties, pools of at least 300 stocks and at least ten stocks per intersection. July 2017 is excluded before outcome evaluation because its smallest cell contains eight stocks. Across the 110 retained formation months from May 2017 to July 2026, the smallest pool contains 1,419 stocks and the smallest intersection 13.

Scores and lists are fixed at month-end, skip m + 1, enter at the first adjusted open in m + 2 and exit at the first adjusted open in m + 8. Neither FW nor the slope-priority BC comparator is updated in the skipped month. There are 103 mature formation months from May 2017 to December 2025, all exiting by 31 August 2026. The outcome is a six-month cohort return, not a monthly return series of overlapping portfolios; annualized returns and Sharpe ratios are therefore not reported. Overall endpoint coverage is 99.424%, falling to 76.923% in the worst month–cell. Missing-as-zero FW-minus-pool spreads are −0.041 and +0.215 percentage points for Top20 and Top100, versus complete-endpoint spreads of −0.082 and +0.215. Monthly comparisons are reported descriptively under both conventions (Supplementary Fig. S5).

In the monthly independent sort, the high-minus-low comovement difference in winner-minus-loser six-month returns is −0.782 percentage points. The 103 mature formation months provide at most 17 nominal six-month blocks, so these values are not used for statistical inference. Some daily adapted rules themselves have low mean returns; FW's advantage over these constructions therefore cannot be extrapolated to an advantage over the original published strategies.

## Model support and longer-block sensitivity

Ordinary and enhanced models share training rows, prediction pools and learner settings. Parameters were not searched against the reported results. After the source corrections, all 224 models were refitted across 56 year–horizon cells. The model bootstrap uses seed 20260923. H120 and H250 supply 12 and 5 nominal complete blocks, respectively, below the sample-support requirement for formal inference. Industry conditioning leaves the valid stock–date set unchanged, as checked across 62,215 date–horizon–method counts under the same subset rule. Ordinary ridge and boosting H3 pool ICs are 0.055210 and 0.053621, versus 0.037238 for FW; these level differences were not separately tested. The prediction loss is not tailored to extreme Top20 selection.

At H3, the concept-feature increments in daily rank correlation are 0.001656 for ridge regression and 0.000392 for boosting. The Top20 return increments are 0.057239 and 0.030821 percentage points, respectively. All four Holm-adjusted P values equal 1. Negative long-horizon estimates with inadequate temporal support are reported descriptively.

For ridge H5/Top100, the mean increment is 0.139702 percentage points, with a pointwise 95% interval of [0.074263, 0.206752] and Holm P = 0.009599. The adjusted P values under 40- and 60-day base blocks are 0.009599 and 0.028497. Assigning zero to missing outcomes at the original list denominator gives 0.139632 percentage points. The H5 Top20 comparison has Holm P = 0.054595 despite a positive pointwise interval.

A post hoc paired diagnostic directly compares list widths on the same 1,609 H5 ridge evaluation dates, from 2 January 2020 to 21 August 2026. Let D_N be the enhanced-minus-ordinary mean return of each frozen Top-N list. The mean D_100 minus D_20 is −0.044035 percentage points. Its pointwise 95% intervals are [−0.130721, 0.039462], [−0.129415, 0.040815] and [−0.129814, 0.040157] for blocks of 20, 40 and 60 dates. The corresponding fixed-denominator missing-as-zero mean is −0.043939, with intervals [−0.131640, 0.040094], [−0.130453, 0.041583] and [−0.131125, 0.041516]. All six Holm-adjusted P values are 1. The design uses 100,000 circular moving-block draws per length, shared across paired quantities, seed 2026100823, centred two-sided P values and a plus-one correction. The six-test family covers both missingness conventions and all three block lengths; it is separate from the original Holm96. Saved predictions and selections are unchanged, and no learner is refitted. An independent exact-endpoint reconstruction from the frozen price cache reproduces the saved daily list returns within 1.5e-8 in return units. The last H5 entry is 24 August and the exit is 31 August 2026. Because this diagnostic was specified after the original width-specific estimates were known, it is not an untouched holdout test. Its results do not support a larger increment for the broader list or establish equivalence between widths.

In the original 10,000-replicate analysis, no primary Top20 or unconditioned pool-IC increment passes Holm96. The five passing cells represent the same ridge H5/Top100 contrast under two missingness conventions and three industry-conditioned IC increments, not five independent findings. The ridge H5/Top100 mean is unchanged under 40- and 60-day base blocks, with pointwise intervals [0.069332, 0.208331] and [0.064716, 0.209258] percentage points. For industry-conditioned IC, ridge H5 adjusted P values become 0.134987/0.224978, and boosting H20 values become 0.027897/0.074393; ridge H1 passes under both longer blocks, whereas boosting H20 passes at 40 days but not at 60 days. Ridge Top20 means at H60/H120/H250 are approximately −0.351/−0.253/−0.878 percentage points. The latter two lack time support, and none provides formal negative confirmation. These fixed models test a conditional increment without exhausting competing information sets.

A subsequent simulation-precision sensitivity retains all forecasts and the full 96-test family. With 1,000,000 circular block draws per horizon and base length (20, 40 and 60 dates), centred two-sided plus-one P values and seed sequences [2026100824, H, L], ridge H5/Top20 complete-endpoint Holm P values are 0.046592, 0.057776 and 0.094116, respectively. Under the missing-as-zero convention, the value is 0.051570 at the 20-date base length. Mapping the complete-endpoint simulation tail-count intervals through Holm gives a sensitivity range of [0.042557, 0.050712] at that base length. These ranges describe finite simulation error; they are neither effect intervals nor simultaneous confidence intervals. The original Top20 significance decision therefore changes with block length, missingness convention and simulation precision; a Top20 increment robust to block length has not been established. H5/Top100 has adjusted P values of 0.002444, 0.010434 and 0.021056 at the three base lengths; the H3 primary-task conclusion is unchanged. Each base length provides a sensitivity analysis of the same 96-test family, retaining placeholders for non-estimable cells, rather than a newly pooled family of 288 confirmatory tests. This post hoc protocol and its fixed simulation budget were frozen before computation. They changed neither forecasts, outcomes nor observed effects, and cannot remove prior research selection. The original figures and preceding numerical summaries retain their provenance from the original 10,000 draws.

## Full-training-date H3 results

The full-training-date sensitivity fits 28 models over the seven annual evaluation windows, using 722–727 training dates in each fit and the same 1,611 mature evaluation dates (2 January 2020–25 August 2026). The maximum training-label endpoint is verified to precede each evaluation year. Fixed ridge and boosting settings, all ordinary features and the three added concept scores are unchanged. This increases the number of training dates while retaining the model settings and target.

Top20 concept-minus-ordinary spreads are 0.0992 percentage points for ridge and 0.0101 for boosting. Neither passes Holm24, nor do Top100 spreads or raw pool ICs (Supplementary Fig. S7 and Table S8). Boosting's industry-conditioned IC increment is 0.005414, with pointwise intervals [0.002725, 0.008111] at length 20 and [0.002388, 0.008413] at length 60. Adjusted P values are 0.0024 and 0.0230. Both dependence specifications support the same conditional-ranking increment at H3. The dense-training exercise is confined to this horizon.

## Strength-related information conditional on synchronization

This targeted diagnostic (E1) asks whether the F and FW feature block improves H3/Top20 selection given W and conventional predictors. The baseline includes 19 ordinary characteristics, their shared missing-count summary and W (21 inputs); the enhanced model adds F and FW (23 inputs). The estimand is the joint predictive increment of these two additions. We retain γ/calendar F5/W10 scores, the feature transformations and fixed histogram-boosting settings in Section S2.2, including no early stopping. Each year from 2020 to 2026 uses all eligible dates in the preceding three calendar years, with training outcomes matured by the previous year-end and before fitting. The models share training and prediction rows, yielding 14 fits. All annual predictions and Top20 lists are fixed before evaluation outcomes are joined. H3 returns use the exact adjusted opens at t + 1 and t + 4; 1,611 formation dates span 2 January 2020–25 August 2026, and all outcome endpoints are no later than 31 August 2026.

The specified endpoints are the equal-date-weighted augmented-minus-baseline return difference over all dates, its mean in a stress state, and the stress-minus-nonstress difference. Stress is defined at formation-day close by the CSI300 trailing 20-trading-day sample standard deviation of simple close-to-close returns exceeding the 90th percentile of that measure during the preceding three calendar years. Each annual threshold uses market observations through the previous 31 December; it requires no future H3 labels. Rows after 31 August 2026 are excluded before constructing index returns or volatility. The stress state contains 174 dates across five evaluation years but occupies only 17 nonoverlapping 20-day blocks anchored at the first evaluation date. It fails the fixed support requirement of at least 120 stress dates, four years and 20 occupied blocks. Both stress endpoints are therefore not estimable for inference; their four block-length cells receive no intervals or P values.

The six-cell family combines three endpoints and block lengths L20/L60. Paired circular moving-block bootstrap uses 100,000 replicates and seed 20261008, with centred two-sided P values and a plus-one correction. Four inestimable cells retain correction placeholders of one. Intervals are pointwise 95% percentiles; a positive-increment claim requires Holm6 confirmation under both block lengths. Resampling conditions on the fitted prediction path, without refitting the learners. This newly fixed comparison uses previously studied history and does not create an untouched holdout or establish an individual F effect.

The full-period increment is −0.0225 percentage points and is not confirmed under either block length (Table S9). Missing outcomes affect 355 of 64,440 intended stock slots (0.5509%); lists are unchanged, and all 1,611 dates retain observed means for both models. Assigning missing outcomes zero on the same lists gives a descriptive difference of −0.0218 percentage points. Annual estimates vary in sign; excluding 2024 changes the full-period mean to +0.0148 percentage points. Neither the negative full-period point estimate nor the failure to reject establishes a stable disadvantage or equivalence.

## Synchronization conditional on observed market risk

A simple projection clarifies the interpretation of synchronization. Write $r_i=\beta_{ic}f_c+\varepsilon_i$, with $\operatorname{Cov}(f_c,\varepsilon_i)=0$, finite second moments and positive variances for $f_c$ and $r_i$. Then

$$
\operatorname{Corr}(r_i,f_c)=\frac{\beta_{ic}\sigma_f}{\sqrt{\beta_{ic}^{2}\sigma_f^{2}+\sigma_{\varepsilon,i}^{2}}}.
$$

Correlation depends on common exposure and residual variation. For positive exposure and positive residual variance, increasing exposure or reducing residual variance raises correlation. Across stocks, a higher correlation need not imply a larger exposure coefficient. This projection motivates testing a risk-conditioned increment; it neither prescribes the aggregation rule nor identifies attention as the source of persistence. The measured W remains a short-window sample statistic over overlapping concepts.

The diagnostic fixes H3/Top20 and four feature sets. Model A contains the 19 conventional characteristics, their original missing-count summary, F, and a shared missing-count summary for two risk measures. B adds W to A; C adds market beta and residual volatility to A; D adds W to C. Risk measures use paired stock and CSI 300 returns over t−60 through t−1, require at least 45 valid pairs, and include a regression intercept. Residual volatility is the residual standard deviation with n−2 degrees of freedom. The single-market residual can retain industry and concept risk. The existing within-pool rank transformation and common missing-value encoding retain the full prediction pool.

The four models share annual three-calendar-year training windows, all eligible training dates, matured labels and the fixed boosting settings in Section S2.2. Seven prediction years, 2020–2026, give 28 fits. Predictions and Top20 lists are sealed before evaluation outcomes are joined. The 1,611 evaluation dates end on 25 August 2026, with all exits by 31 August. Three paired contrasts—B−A, D−C and (D−C)−(B−A)—share 100,000 circular moving-block draws at L20 and L60, seed 20261009. The six tests share Holm correction; intervals are pointwise 95% percentiles conditional on the fitted predictions. D−C is the primary contrast. The specification was frozen before these results were calculated, using previously studied history.

The W increment is 0.0742 percentage points before adding the two risk measures and 0.0032 afterwards. Neither increment nor their direct change is confirmed by the six-test family (Table S10). The reduction in the point estimate is insufficient to establish that market risk explains or absorbs W’s contribution. Missing outcomes affect 744 of 128,880 intended model–stock–date slots; all four models retain observed outcomes on every evaluation date. The descriptive fixed-denominator, missing-as-zero means are 0.0728, 0.0012 and −0.0715 percentage points for the three contrasts.

## Attention measurement and ranking reference sets

The endpoint analysis in main Fig. 6 describes the temporal profiles of the three recorded fields. We use future endpoint levels so that current attention does not enter both the predictor and a future-minus-current outcome. On identical support, S’s future-endpoint regression coefficient is approximately 103–119% of its contemporaneous coefficient over 1–120 trading days. The corresponding K ratio is about 10–11% at days 1–3, declining to 5.1% at day 20 and 1.9% at day 120. T endpoint responses are about −7.5% and −7.4% of the contemporaneous response at days 1 and 3, with endpoint-family Holm-adjusted P values of 0.3866 and 0.4555. The positive day-120 T coefficient is about 5.1% of the contemporaneous coefficient and passes correction (P = 0.0300). These ratios describe persistence in the recorded fields, not the retention of individual users.

The attention-measurement analysis uses Wind's stock-popular-concept-detail fields for total popularity (T), watchlist popularity (S) and stickiness (K). The jointly valid trading-day panel contains 5,682,319 stock-days from 23 March 2022 to 31 August 2026. Extraction retains 15,743,951 of 15,871,852 raw records after excluding 127,901 records beyond the cutoff. On 30 April 2026 no stocks have jointly valid T/S/K values; the date remains missing. These vendor fields define the measurement system studied here. Their interpretation differs from evidence based on Google searches [@da_engelberg_gao_2011].

All three fields use a common daily percentile-rank transformation. At H3, current ranks correlate with their own future ranks by approximately T = 0.8918, S = 0.9998 and K = 0.4542. Replacing current levels with five-trading-day rank changes gives 0.0824, −0.0801 and 0.0101. These quantities distinguish persistence in rank levels from prediction using rank changes. Their target is future attention rank, not future stock return, and neither directly observes individual attention. To isolate reference-set sensitivity, we rerank raw current and past values within the stocks present both today and five trading days earlier, then subtract the two common-set ranks. Only this explanatory change is replaced; the future target is unchanged. Stocks must survive in the common set, so this transformation does not remove selection.

For S at H3, 1,067 paired dates yield mean correlations of −0.080145 under the original definition and +0.026331 under the common-set definition (Supplementary Fig. S8). The paired difference is 0.106476. Pointwise 95% intervals are [0.081422, 0.133429] with 20-day blocks and [0.073230, 0.143213] with 40-day blocks; each has Holm24 P = 0.000240. The family contains three fields × eight horizons, corrected separately by block specification; H60 and longer are descriptive. Requiring defined correlations for all three fields under both definitions removes no further dates. The sign reversal identifies sensitivity of the attention-rank association to its ranking reference set.

For H = 1, 3, 5, 10, 20, 60, 120 and 250, paired reranking uses 1,069, 1,067, 1,065, 1,060, 1,050, 1,010, 952 and 822 dates. All begin on 30 March 2022 and end, respectively, on 28 August 2026, 26 August 2026, 24 August 2026, 17 August 2026, 3 August 2026, 5 June 2026, 9 March 2026 and 19 August 2025, so that every future rank is observed by 31 August 2026. H3 has 1,068 endpoint-mature dates before requiring at least 30 valid future common stocks, which removes one date. Missing observations are not zero-filled. The composition analysis contains 1,071 dates and 5,648,305 common stock-days.

$$
D_{\mathrm{full}}=D_{\mathrm{common}}+E_{\mathrm{reference}}.
$$

The reference term is the exact residual between the two rank-change definitions. For S, its mean daily cross-sectional standard deviation is 0.000228, compared with 0.008236 for common-set change. Strictly opposite non-zero signs occur for an average of 7.101% of all common stocks; a larger absolute reference term occurs for 15.118%. The two attention-change Top20 and Top100 lists overlap by 99.556% and 99.323% (Supplementary Fig. S9). Perturbations around zero can therefore coexist with stable extreme attention-change lists. These lists differ from the FW portfolios evaluated for returns. Integer-rank and rational-number checks reproduce stock-level and variance identities within $3\times10^{-16}$.

The common five-trading-day transformation differs from earlier fresh measures: historically standardized one-calendar-day change for T, five-calendar-day log change for S, and same-weekday historical robust standardization for K, followed by cross-sectional ranking. For the one-calendar-day T change and five-calendar-day S change, source records must have consecutive calendar dates across the lag; even when both endpoint levels exist, a missing intermediate date makes the change unavailable. K's same-weekday standardization uses its separate historical reference.

The earlier Δ20/H1 comparison of S minus K now uses the same 1,040 dates under 20- and 40-day blocks, because its 1,054 available dates are truncated to complete blocks. The mean is −0.056901. Holm96 P values are approximately 0.026 and 0.481, from unadjusted P values of approximately 0.00066 and 0.01233. On identical dates, the conclusion therefore depends on the assumed dependence length. The complete-block prefix ends with the 10 August 2026 signal and 11 August outcome.

An additional post-audit circular moving-block sensitivity retains all 1,054 dates through the 28 August signal and 31 August outcome. It gives a mean of −0.057794 and Holm96 P values of 0.022 and 0.345 at 20- and 40-day blocks, respectively. Each block specification re-estimates the full 96-comparison family with 100,000 draws. This supplementary resampling changes both tail inclusion and block construction; it preserves the qualitative dependence-length sensitivity without replacing the primary estimates. These transformations and inferential sensitivities should not be merged into a single attention-decay estimate.

## Temporal preprocessing in causal machine learning

This analysis estimates the association between recorded attention and subsequent panel-relative returns after flexible adjustment for observed characteristics. It does not estimate mediation of the FW selection spread. The conditional-attention panel contains 448,660 stock-days, 474 sampled dates and 5,220 stocks in each of α, β and γ. Signal dates run from 1 December 2016 to 27 August 2026. Every fifth eligible market date is sampled; stock-days are then selected by the fixed hash rule on common finite, positive β/γ score support. The F5/W10 network uses valid stock–concept observation rows: five returns for F and at least five valid pairs within the latest ten observations for W. Membership records are dated no later than t under β and no later than the preceding market date under γ; both retain price information through t. α uses the β network. Own and peer attention records are dated no later than t minus two calendar days. Delayed attention measurement and delayed membership information impose two separate timing constraints. Peer exposure is the winning concept’s share of raw market-wide popularity, with overlap allocated by membership degree. The focal stock’s allocated contribution is removed from the concept numerator and its raw popularity from the market denominator. It is not the mean popularity of the peers. The winning concept is selected by the highest signal-day F5/W10 membership score, after the naturally lagged attention observation; lagged membership records do not make that selection a pre-exposure network. This branch differs from the main FW score, which averages products across memberships.

The first 40% of sampled dates form the initial training period; the remaining dates form six expanding-window test folds under the fixed 40% plus six 10% partition rule. Their test ranges are 22 October 2020–28 September 2021, 12 October 2021–23 September 2022, 30 September 2022–12 September 2023, 19 September 2023–9 September 2024, 18 September 2024–29 August 2025 and 5 September 2025–27 August 2026. Training uses earlier signal dates and excludes labels whose endpoints have not matured strictly before the test start. All inputs and complete-H outcomes are restricted to 31 August 2026. The target is the complete-H return minus the same-design, same-date panel mean, calculated before state grouping and model-validity filtering. α uses C(t) to C(t+H); β and γ use O(t+1) to O(t+H+1). The target is a panel-relative return, not an index excess return.

Tree-DML residualizes continuous own-stock and leave-one-out category-peer exposures for T, S and K and the return outcome, using the residualization construction in the DML framework [@chernozhukov_etal_2018]. Own T is a historically standardized one-calendar-day change in log(1 + T); own S and K are observation-date percentile ranks of their log levels. These quantities then undergo the stated historical calibration and scaling. The past-calibrated protocol estimates exclusion thresholds at the 1st and 99th percentiles using only dates before 22 October 2020; observations outside them are removed, not winsorized. Remaining doses are demeaned within date and scaled by a fixed historical population standard deviation. Calibration requires at least 1,000 finite rows and 20 dates; otherwise estimation is deferred. S lacks sufficient early calibration data and does not borrow later information. AIPW uses an augmented inverse-probability-weighted contrast [@bang_robins_2005] and compares states defined by within-date percentile ranks of the change in T and the log levels of S and K: low ≤1/3, high ≥2/3, with middle observations excluded. The joint comparison requires all three fields high or all three low. Continuous doses, rank changes and these states are distinct exposures, not interchangeable attention shocks. In the retained Tree-DML implementation, exposure and outcome regressions are fitted on their own finite and trimmed supports, then evaluated through paired residuals on their intersection. They need not estimate the two conditional means under the same target distribution. The retained coefficients and date-HAC calculations are residual-association diagnostics; their reproducibility does not establish the usual orthogonal-score inference conditions. Section S13.1 reports a common-support diagnostic for own T/S/K at γ/H3.

Controls comprise 26 numerical characteristics and Shenwan level-1 industry: one-day return; 5/20/60-day momentum; log free-float and total capitalization; log current and 20-day mean turnover value; 20-day turnover value/capitalization and Amihud illiquidity; 60-day CSI 300 beta, idiosyncratic volatility and drawdown from the high; current market return; 20-day suspension and zero-turnover counts; 20-day institutional, large-order and retail net inflow/turnover; CSI 300/500/1000 membership; F5/W10 score; concept degree; network membership count; and peer count. Non-finite numeric inputs use native missing-value handling. Industry vocabulary is learned from actual fitting rows and unseen categories are missing. Some signal-day controls follow the naturally lagged attention observation, so not all can be interpreted as pretreatment confounders. Flow variables are vendor-defined order-size proxies, not verified investor identities or an exhaustive decomposition of turnover.

Nuisance regressions and propensity models use histogram boosting with learning rate 0.05, 31 leaves, at least 100 rows per leaf, L2 = 1 and 80 fixed iterations, without random internal early stopping. DML deterministically retains at most 30,000 training rows per role; AIPW retains at most 15,000 per state and restores presampling state proportions through propensity-model weights. DML nuisance fits require at least 1,000 training and 100 test rows. Reported diagnostics require at least 10,000 usable rows, 100 test dates and all six completed folds. AIPW additionally requires at least 1,000 high and low observations in every training and test fold, with at most 20% of pooled test propensities clipped to [0.05, 0.95]. These thresholds assess estimability and overlap, not unconfoundedness.

DML divides the sum of exposure–outcome residual products by the sum of squared exposure residuals. AIPW averages pseudo-outcomes combining state-specific outcome predictions and inverse-propensity-weighted residuals. Both weight stock-day contributions rather than daily coefficients equally. Diagnostic standard errors aggregate scores by date and apply HAC lag max(1, ceil(H/5)) sampled dates; two-sided tests use a normal approximation and intervals are pointwise 95%. DML coefficients are complete-H percentage points per fixed historical standard deviation; AIPW coefficients are high-minus-low percentage points. Six Holm24 families separate own and peer exposures within each design; three Holm32 families cover four states × eight horizons within each design. Deferred cells retain P = 1 as a correction placeholder, with coefficient estimates withheld. Supplementary Table S11 gives implementation support and diagnostic Holm and secondary BH counts; Supplementary Figs. S10 and S11 display estimates together with their horizon-specific support.

Under past-only calibration and fixed training, none of the 84 estimable continuous-exposure coefficients or 42 estimable state contrasts passes its diagnostic Holm family. For K high minus low at H20, β gives −0.474102 percentage points, pointwise 95% interval [−0.814250, −0.133954] and Holm32 P = 0.201555; γ gives −0.454133, interval [−0.796372, −0.111894] and P = 0.297651. Each uses 215,670 stock-days and 281 evaluation dates. The T and K cells are estimable through H120. H250 has no mature test outcomes in the final fold, whose first signal is 5 September 2025, and therefore fails the requirement that all test folds contribute. S also lacks early calibration or training support.

The global-preprocessing diagnostic has one Holm-significant AIPW contrast: α, K high minus low at H5, −0.228626 percentage points (Holm32 P = 0.024395). Its preprocessing uses information beyond the training period, so it is reported separately from the past-only conditional comparisons. The analyses share observations and estimate conditional associations; causal interpretation additionally requires the identification assumptions stated in the main text.

The S-coverage analysis starts on 30 March 2022, the first sampled date with finite S. Calibration uses dates before 30 September 2022, the first test-fold start preceded by at least 1,000 finite S rows and 20 sampled dates; before the six-exposure intersection, this coverage rule has 26,936 finite own-S rows on 25 dates per design. The ensuing dose calibration uses 26,846 common rows on those 25 dates. The four subsequent folds evaluate 30 September 2022–27 August 2026. The S-coverage branch first restricts the panel to the intersection of all six finite raw own and peer exposures: 262,877 stock-days on 215 dates per design, including calibration. Subsequent exposure-specific trimming and outcome availability determine the role-specific fitting and evaluation supports. The same models, seeds and support gates apply, with the complete-fold requirement evaluated over these four folds.

All six continuous exposures are estimable at H1–H120 in each design, giving 126 of 144 Tree-DML cells on 166–190 evaluation dates; no coefficient passes the diagnostic Holm24 correction (smallest adjusted P = 0.316). H120 has sufficient mature training labels before the first test fold. H250 has none before that fold and also lacks mature outcomes in the final test fold. AIPW remains estimable for T and K at H1–H60 (36 of 96 cells), with none passing Holm32. At H120, the first fold fails the requirement of at least 1,000 training observations in each high and low state group. S and joint-state AIPW contrasts fail the overlap requirement even at H1–H60: pooled test propensities are clipped in approximately 42–45% and 51–53% of observations, respectively (Supplementary Fig. S11c). Thus longer field coverage permits estimation of continuous S exposures, but a high-versus-low S comparison meeting the support requirements has not been established. This coverage-defined sensitivity shares data with the primary protocol and is neither preregistered nor an independent test.

### Common support and control timing

A focused diagnostic (N1) examines whether the residual-association estimates change when auxiliary-model support and the timing of controls are aligned. The design was fixed before this diagnostic was run, after the earlier historical results had been examined. It retains γ, H3, own T/S/K, the S-coverage window and four expanding evaluation folds. The exposure definitions, original historical trimming thresholds, within-date demeaning and fixed historical standard deviations are unchanged. The outcome remains the same γ three-day panel-relative return. The final evaluation signal is 20 August 2026, with its return endpoint on 26 August 2026.

Four specifications share 217,831 evaluation stock-days across 185 dates. M0 retains the original role-specific fitting supports and evaluates the resulting residuals on this common sample. M1 restricts both nuisance regressions to the same eligible training rows and requires H3 outcomes to mature before the test fold. M2 removes four concept controls—F5/W10 score, concept degree, network membership count and peer count—while retaining 22 conventional characteristics and industry at their signal-day values. M3 dates those same 22 characteristics and industry strictly before the recorded attention observation. M1−M0 therefore changes both support and label purging; M2−M1 changes the concept-control block; M3−M2 changes the timing of an otherwise fixed control set. These comparisons do not decompose causal pathways.

Pre-attention records are matched before outcome values are loaded. For each stock, the latest retained covariate record must precede the natural attention-observation date by no more than seven calendar days; its feature and source-trade timestamps must not exceed that record’s date. Of 262,877 candidate stock-days, 257,880 pass this match and 4,997 do not. The common evaluation sample additionally requires all three calibrated own exposures and H3 outcomes to be finite and to enter completed test folds. The retained panel supplies historical records, without certifying first-publication vintages. M0–M3 retain the existing boosting settings, deterministic training cap and fit-only industry encoding, giving 64 nuisance fits. Sixteen additional fits of the original implementation verify the M0 residuals and are not a separate scientific comparison.

Each exposure has its own residual-projection coefficient, calculated as the ratio of the exposure–outcome residual product sum to the exposure-residual sum of squares. These are single-exposure projections, not joint effects of T, S and K. Coefficient units are full three-day return percentage points per corresponding fixed historical standard deviation; the fields have different exposure definitions and scales. Diagnostic standard errors aggregate score contributions by date. Paired specification differences subtract the two estimators’ contributions on the same dates. Three fixed HAC lags—1, 8 and 12 sampled dates—cover short and longer dependence assumptions. A Holm27 family contains three adjacent specification differences × three fields × three lags; a separate Holm36 family contains four specifications × three fields × three lags. All intervals are pointwise 95% score-based diagnostics conditional on the fitted nuisance functions and the stated date-dependence specification. They omit uncertainty from re-estimating the nuisance functions; aligned support alone does not establish nuisance convergence or coverage in this dependent panel.

Table S12 reports the twelve coefficients. No test passes either diagnostic family: the smallest adjusted P values are 0.100605 for specification differences and 0.057180 for coefficients. Figure S12 displays the nine paired differences at HAC8; the complete 27 contrasts and 36 coefficient calculations accompany the source data. Point estimates change when the controls move before attention, but the paired tests do not establish specification differences under the declared correction. Nonrejection does not establish equivalence, zero attention effects or valid orthogonal-score coverage. The diagnostic addresses observable support and timing choices within one γ/H3 branch; exposure assignment, persistent popularity and unobserved common news remain separate identification questions.

## Attention measures, list visibility and participation

Current attention information improves prediction of continued appearance on supplier-labelled one-hour and 24-hour popularity lists at the next market session. Compared with strictly historical slow-state models, adding current T/S/K increased the out-of-time area under the precision–recall curve (PR-AUC). The increments were 0.001274 for retention on the one-hour list (95% interval, 0.000496–0.002037) and 0.000933 on the 24-hour list (0.000539–0.001340). Both headline-family Holm8-adjusted P values were 0.015984, and the increments were positive across all eight available folds.

The increment in top-100 recall for new entries on the one-hour list is 0.005235 (pointwise 95% interval, 0.001053–0.009669; Holm8-adjusted P = 0.083916), with positive increments in six folds, but it does not survive family correction. Its code-rotation increment is −0.002613 (95% interval, −0.005029 to −0.000245); the interval is not contained within the prespecified ±0.0025 equivalence band, whereas both retention controls satisfy that bound. Among the eight primary comparisons, the two retention comparisons meet all implemented primary criteria. These results provide a related-measure test of persistent list visibility rather than independent validation of latent attention.

The one-hour and 24-hour labels denote list construction windows; both predictions target the next market session. Complete current and next-session Top100 anchors and complete current and past 5-, 20- and 60-session T/S/K vectors are required. Each headline uses 797 evaluation dates in eight folds. The last evaluated one-hour signal is 27 July 2026, with its outcome on 28 July; the corresponding 24-hour dates are 20 and 21 August. The one-hour candidate panel extends to 13 August, but later candidates fail the historical-input gate: a missing 30 April source record interrupts the consecutive-calendar-day S-change window and propagates through the 60-session lag; incomplete list anchors exclude additional dates. All inputs and outcomes remain within the 31 August cutoff.

The last partial fold contains 15 and 16 evaluation dates, respectively. Excluding it in a diagnostic reaggregation leaves the same two retention comparisons passing the implemented Holm8, fold-sign and observed-versus-rotation criteria. After excluding that fold, the one-hour new-entry increment is 0.004424 (pointwise 95% interval, 0.000540–0.008337; Holm8-adjusted P = 0.191808), with positive increments in five of seven folds. The available records also do not establish official platform identity, exposure levels or independence between upstream sources.

Eight of 72 attention–trading–horizon combinations meet the joint reporting requirements: T with trading value at H5 and turnover at H5/H10; S with trading value at H3/H5 and turnover at H20; and K with turnover at H5/H10. All involve participation at 3–20 days. Adding signal-day trading controls reduces absolute candidate coefficients to a median of 68.2% of their original values; this is not a mediated share. Three of 24 attention-to-return paths pass the max-|t| test against the complete 144-path maximum: T at H3 (positive, P=0.045954) and S at H1/H3 (negative, P=0.007992/0.022977). None satisfies the full bidirectional criterion. The primary maximum-t correction and the separately reported Holm144 correction use different reference procedures and are not interchangeable.

Stock-identity permutations operate within signal-date × industry × size-quintile × liquidity-quintile strata. Within each stratum, stocks are ordered by a fixed seeded hash and cyclically mapped to the next stock; singleton strata do not contribute a permuted contrast. The historical-misalignment control replaces attention in the attention-to-trading path with its same-stock value from 60 trading days earlier. In the trading-to-attention path, the trading observation is moved a further 60 trading days back from its H-session lag before the attention observation date. The joint criterion requires the 90% coefficient intervals for both controls to fall within the predefined ±0.02 equivalence band in both directions, alongside the directional and asymmetry tests.

The order-size module specifies 96 comparisons across three attention measures, two order-size outcomes, endpoint and path-mean responses, and eight horizons. It requires the sums of buying and selling amount buckets to reconcile with total turnover with relative errors no greater than $10^{-6}$. Among 5,495,232 otherwise finite stock-days, 12,903 (0.235%) exceed this tolerance, above the allowed 0.1%. Discrepancies are concentrated in July–August 2026 (12,753 of 228,778 stock-days, 5.574%). The full window therefore does not meet the study’s prespecified reconciliation standard, and its formal estimates are not reported.

This finding does not establish whether the discrepancies arise from source errors or differences in the definitions of the bucket and total-amount fields.

The causal-machine-learning flow controls come from a separate vendor-classified source. Those order-size categories remain proxies: their investor identities and completeness as a partition of turnover are unverified. The endpoint, trading-feedback and comovement controls do not use these amount buckets.

## Numerical detail for the attention-to-comovement diagnostic

On 123,530 complete stock-days across 202 dates, equal-date mean correlations with real concept peers and the pseudo-network benchmark are 0.19861 and 0.04782 in the earlier window and 0.19796 and 0.04983 in the future window. For a ten-percentage-point increase in the cross-sectional percentile ranks of T innovations, five-calendar-day S changes and weekday-standardized K (Section S12), future real-minus-pseudo slopes are 0.000276, 0.000764 and −0.000431. Future-minus-past slopes are 0.000595, −0.000387 and 0.000965, with diagnostic Holm9 P values of 0.8906, 1 and 0.3533. The earlier-window K slope is −0.001396 (pointwise diagnostic 95% interval −0.002370 to −0.000452; Holm9 P=0.0270). It is the only diagnostic comparison to cross 0.05, and it does not meet the matching requirement. All 202 dates fail the required paired size, degree and liquidity balance criteria. Formal inference is therefore withheld for the entire network comparison.

## Additional source and mechanism diagnostics

### Raw-source and historical-vintage checks

Before the correction of four source closing prices, a deterministic raw-source audit covered 20 complete selected observations from each calendar year from 2016 to 2026, plus all 66 selected observations with missing endpoints, giving 286 stock–date checks. Adjusted open prices were reconstructed directly from the raw daily open, close, and previous-close records. All source files were available in the fixed snapshot, the 220 complete returns agreed exactly with the cached returns, and all 66 missingness classifications agreed. The same 286 stock–date checks reproduced the cached concept-membership sets from the raw records without disagreement. Across 2,372 calendar mappings, no membership snapshot carried a recorded date later than the preceding market day.

These checks establish agreement among the available price, membership and calendar records, but consistency of recorded dates is not equivalent to point-in-time provenance. First-publication dates, earlier versions, historical backfilling and survivor coverage require the corresponding historical source archive. The targeted corrections in Section S1 update the available source version without supplying that archive.

### Why the pseudo-network matching gate failed

The gate audit reproduced the failure classification on all 202 primary split-reference dates. Block-based exchangeability ranged from 0.9876 to 0.9941, exceeding the 0.60 requirement throughout. In contrast, the mean absolute standardized distances between original and permuted nodes exceeded 0.10 for every date and every gated attribute. The median distances were 0.4365 for size, 0.3895 for degree, and 0.7732 for liquidity; their respective ranges were 0.4261–0.4470, 0.3401–0.4040, and 0.6493–0.8841. These quantities measure average absolute paired-node distances, not standardized differences between aggregate covariate means. Permutation within industry and broad size and degree terciles preserves coarse composition but does not ensure close individual matches; liquidity was evaluated by the gate but was not a permutation stratum. The failed gate therefore cannot be interpreted as evidence for the absence of an attention mechanism.

A covariate-only extension applied the existing strict-caliper diagnostic to all 202 saved graph dates, from 25 May 2022 to 23 July 2026. The extension was specified after reviewing the existing gate failures and before calculating these diagnostics. It retained the original industry, size-tercile and degree-tercile strata and required distinct candidate partners to lie within 0.10 population standard deviations on each of size, degree and liquidity. In the current 5,236-stock universe, the fraction with any eligible partner ranged from 6.86% to 12.82%, with a median of 9.36%; no date reached 60%. These fractions are upper bounds on candidate coverage under the stronger per-pair rule, not evidence that a complete permutation exists.

A deterministic disjoint-swap construction covered 6.61%–11.76%, with a median of 8.77%, providing a feasible lower bound rather than an optimal assignment. The original coarse-stratum exchangeable fractions were reproduced on every date. The diagnostic used covariates only and calculated no replacement return outcomes.

Because the original gate constrains mean distances rather than every pair, these results do not prove that all assignments satisfying the original gate are infeasible. No replacement network passed the original validation procedure, and no threshold was relaxed.

## Retrospective account simulations

The implementation grid is run in a separate stock-selection framework (select-stock-xhzheng 2.5.0). All 48 runs were repeated using the frozen corrected inputs documented in Section S1, with their original scientific configurations unchanged. The daily net values, selection records and configurations are retained in the replication archive together with the framework code; the archived outputs are byte-identical to the framework outputs. The score follows the research construction with different details: concept daily returns are equal-weighted over all member stocks, including the focal stock; F is ranked across concepts and W across stock–concept pairs; memberships come from the preceding market day; and ties take the minimum rank, so a list can exceed its nominal size on tied dates. The research lists instead use leave-one-out concept returns and rank both quantities across membership records. These backtests therefore evaluate an implementation variant of the idea rather than the frozen research lists.

Each backtest starts with CNY 30,000,000 and excludes Beijing Stock Exchange listings. Daily rankings use information through the preceding close. After selection, the archived framework also removes target stocks using a one-price upper-limit flag defined by the execution day's minimum price being at or above its upper limit. Holding periods of three or five days are implemented as three or five equally capitalized sub-portfolios rebalanced on staggered days. Buy orders blocked by the upper price limit are retried at subsequent simulated price points that day and cancelled if still unfilled at the close; positions blocked by the lower limit are retained, with sale attempts repeated at subsequent simulated price points. Missing daily closes are carried forward for valuation; this convention does not imply that a carried-forward price is executable. Commission is fixed at 1.2 basis points and stamp duty at 0.1% on sales, above the statutory rate of 0.05% after 28 August 2023. Slippage and market impact are not modelled, and capacity is not assessed.

Both timing overlays first rank candidates using information through the preceding close, then form market-breadth signals and simulate execution at their specified times. The one-level and two-level rules set the target exposure of the sub-portfolio scheduled for rebalancing to zero when a specified proportion of candidates within a specified top-ranked fraction have prices below a specified price. Their parameters are framework and trading settings and were not searched in this study. Because the earlier parameter-selection process was not recorded, the timing results should be interpreted as exploratory retrospective evidence conditional on the target filter described above.

A single-gate sensitivity removed the full-session limit-up exclusion across all 48 configurations, holding ranking and weighting rules, fees and dynamic price-limit handling fixed. Relative to the archived paths, the change in annualized return ranged from −0.138 to +0.340 percentage points (median +0.003); 24 changes were positive, 19 negative and five were unchanged at the reported precision. Maximum-drawdown changes ranged from −0.309 to +0.007 percentage points (median 0.000), with drawdowns expressed as negative values. The remaining next-session state filters and quote-availability assumptions were retained, so this comparison isolates the exclusion's effect on the reported accounts. S3 Dataset reports the 48 paired outcomes and numerical tolerances.

Statistics are recomputed from the daily net values rather than taken from framework summaries. Across the 48 runs, untimed compound annual returns range 12.7%–23.8%, with maximum drawdowns of −59.6%–−37.2%. Breadth-timed runs return 19.9%–26.6% a year (drawdowns −36.6%–−15.9%) and two-level-timed runs 21.9%–33.5% (−28.2%–−14.6%). Untimed runs keep approximately 96%–97% of assets invested on average; annualized two-way turnover is 71–132 times average assets, and annual fees are 4.4%–8.2% of average assets. Timed runs issue zero target-exposure signals on approximately 40%–45% of signal dates, applying to scheduled sub-portfolios rather than the entire account. Commission-implied annualized two-way turnover is approximately 86%–114% of the median for runs with the same holding period and timing. It is calculated as total commission divided by the commission rate, mean assets and actual elapsed years; the peer ratio is a trading-intensity diagnostic, not a ratio of raw cumulative traded values.

The Top5 and Top10 list sizes lie within the research grid (Fig. 2c), and the staggered sub-portfolios mirror its overlapping daily cohorts. The account grid covers two window pairs and these two list sizes. It contains no one-to-one implementation of the Top20 research lists, same-framework W-only or simple-aggregation benchmarks, or the other four window pairs. Attribution of account performance to the score is therefore confined to the comparisons actually implemented.

[FIGURE_S1]

[FIGURE_S2]

[FIGURE_S3]

[FIGURE_S4]

[FIGURE_S5]

[FIGURE_S6]

[FIGURE_S7]

[FIGURE_S8]

[FIGURE_S9]

[FIGURE_S10]

[FIGURE_S11]

[FIGURE_S12]

The complete drawdown histories of the three fixed configurations in Fig. 8a are shown in Supplementary Fig. S13.

[FIGURE_S13]

### Supplementary Table S1 | Sample description by year

| Year | Signal dates | Pool stocks per date | Distinct stocks | Concepts per date | Median concept size | Memberships per stock | Median float cap (CNY bn) |
| ------------ | --------: | ---------: | ---------: | ---------: | ---------: | ----------: | ---------: |
| 2016 (partial) | 7 | 1,740 | 1,783 | 148 | 21 | 2.0 (2) | 7.31 |
| 2017 | 244 | 1,896 | 2,312 | 153 | 22 | 2.1 (2) | 6.57 |
| 2018 | 243 | 2,300 | 2,593 | 166 | 25 | 2.5 (2) | 4.48 |
| 2019 | 244 | 2,731 | 3,022 | 183 | 29.5 | 2.9 (2) | 4.40 |
| 2020 | 243 | 3,160 | 3,504 | 206 | 36 | 3.5 (3) | 4.65 |
| 2021 | 243 | 3,732 | 4,021 | 231 | 41 | 3.9 (3) | 4.54 |
| 2022 | 242 | 4,301 | 4,614 | 295 | 45 | 4.8 (4) | 4.35 |
| 2023 | 242 | 4,761 | 4,925 | 340 | 50 | 6.2 (5) | 4.39 |
| 2024 | 242 | 4,994 | 5,075 | 370 | 59.5 | 7.9 (6) | 3.88 |
| 2025 | 243 | 5,085 | 5,167 | 387 | 73 | 10.1 (8) | 5.25 |
| 2026 (to 31 August) | 160 | 5,140 | 5,212 | 385 | 83.5 | 11.3 (9) | 6.02 |
| All | 2,353 | 3,755 | 5,228 | 266 | — | 6.1 (4) | 4.73 |

Pool statistics use the frozen common research pool; pool stocks per date are equal-date means. Concepts per date count distinct labels in daily membership snapshots; median concept size is the median across snapshot dates of the within-date median number of member stocks per concept. Memberships per stock are scored lagged memberships of pool stocks, shown as mean (median) across stock-days. Float capitalization is the pool median in CNY billions. The first-level industry classification has 28 industries through 2020 and 31 from 2021. In the partial 2016 row, pool statistics cover seven trading dates from 22 to 30 December; concept-snapshot statistics cover 38 recorded snapshot dates from 24 November to 31 December. The 2026 row ends on 31 August. The snapshots contain 404 distinct concept labels in total.

### Supplementary Table S2 | Mature signal dates by horizon

| H | Historical rules | Daily adaptations | Rolling models | Last mature signal |
| --- | --- | --- | --- | --- |
| 1 | 2,351 | 2,331 | 1,613 | 2026-08-27 |
| 3 | 2,349 | 2,329 | 1,611 | 2026-08-25 |
| 5 | 2,347 | 2,327 | 1,609 | 2026-08-21 |
| 10 | 2,342 | 2,322 | 1,604 | 2026-08-14 |
| 20 | 2,332 | 2,312 | 1,594 | 2026-07-31 |
| 60 | 2,292 | 2,272 | 1,554 | 2026-06-04 |
| 120 | 2,232 | 2,212 | 1,494 | 2026-03-06 |
| 250 | 2,102 | 2,082 | 1,364 | 2025-08-18 |

Counts are mature paired signal dates, not independent stock observations. The three series begin on 22 December 2016, 20 January 2017 and 2 January 2020. Daily adaptations use the γ common pool. The observation cutoff is 31 August 2026, and every exit falls on or before that date. Selection overlap in Fig. 4c does not require outcome maturity and uses all 2,353 frozen dates; Fig. 4a,b,d use 2,349 mature dates.

### Supplementary Table S3 | Multiplicity families and temporal support

| Family | Composition | Total | Supported |
| --- | --- | --- | --- |
| Historical rules | 2 labels × 2 formation-window conventions × 3 N × 8 H × 2 missingness × 5 controls | 960 | 720 |
| Same-information aggregation | 2 labels × 3 N × 8 H × 2 missingness × 2 controls | 192 | 144 |
| Daily adaptations | 2 labels × 3 N × 8 H × 2 missingness × 4 controls | 384 | 288 |
| Model increment | 80 primary comparisons + 16 industry-conditioned IC | 96 | 72 |
| Six-pair parameter sensitivity (complete) | 6 F/W × 3 H × 6 N × 5 controls | 540 | 540 |
| Six-pair parameter sensitivity (missing = 0) | Same grid; separately corrected sensitivity | 540 | 540 |
| Industry and size controls | 3 N × 3 group benchmarks at H3 | 9 | 9 |
| Full-training-date H3 sensitivity | 2 models × 6 metrics × 2 block lengths | 24 | 24 |
| Strength block given W (E1) | 3 endpoints × 2 block lengths; four stress cells not estimable | 6 | 2 |
| W conditional on market risk (S11) | 3 paired contrasts × 2 block lengths | 6 | 6 |
| Paired membership and return timing | 5 contrasts × 2 missingness conventions × 2 block lengths | 20 | 20 |
| Fixed-period H3 sensitivity | 2 periods × 2 benchmarks; separately per block length | 4 | 4 at L20/L40; 2 at L60 |
| Paired H5 ridge list-width diagnostic | 2 missingness conventions × 3 block lengths | 6 | 6 |
| Comovement bridge | 3 attention fields × 3 temporal contrasts | 9 | 0 formal |
| Attention endpoints | 3 measures × 8 H; block bootstrap | 24 | 21 |
| List visibility (headline) | 2 list windows × 4 tasks | 8 | 8 |
| Attention–trading maximum-t | 3 measures × 3 trading outcomes × 8 H × 2 directions | 144 | 144 |
| Reference-set reranking | 3 fields × 8 H, per block specification | 24 | 15 |
| Attention rank-change differences | 4 transformations × 8 H × 3 field pairs | 96 | 60 |
| Tree-DML temporal audit | 6 families (own/peer × α/β/γ), each 3 exposures × 8 H | 144 | 84 |
| AIPW temporal audit | 3 families (one per design), each 4 states × 8 H | 96 | 42 |
| S-coverage window: Tree-DML | Same structure, separate families | 144 | 126 |
| S-coverage window: AIPW | Same structure, separate families | 96 | 36 |
| N1 paired specification differences (S13.1) | 3 adjacent contrasts × 3 own fields × 3 HAC lengths | 27 | 27 diagnostic |
| N1 residual-projection coefficients (S13.1) | 4 specifications × 3 own fields × 3 HAC lengths | 36 | 36 diagnostic |
| Longer-block parameter checks | 6 F/W × 5 controls at H3/Top20, per block length | 30 | 30 |
| Observation-row timing (S4.1) | 5 contrasts × 8 H × 2 N, per missingness convention | 80 | 80 |
| Order-size composition | 3 measures × 2 outcomes × 2 estimands × 8 H | 96 | 0: turnover reconciliation not satisfied |

Supported counts do not denote significant discoveries; two of the eight list-visibility comparisons meet the implemented joint headline criteria. Holm adjustment is applied separately to each declared family and does not control error over the full research history. Intervals are pointwise unless stated otherwise.

Marker semantics are specified per figure: stars in Fig. 2b encode Holm540 significance; filled points in the earlier horizon plots encode temporal support, not corrected significance. Monthly mappings receive no significance tests. The bridge fails its matching requirement; all nine tests are diagnostic only. Unsupported tests remain in their correction families with P = 1. Longer-block parameter-sensitivity checks use a separate 30-test H3/Top20 family at each block length. Family sizes recur across modules; in the text each family is named with its module. The joint feedback criterion and the order-size reporting requirements are rules, not Holm families.

The temporal-audit rows refer to the retained past-calibrated implementation diagnostics; full-period-preprocessing diagnostics are listed separately in Table S11. Failure to reconcile the amount fields is not an estimate of a zero effect.

### Supplementary Table S4 | Current F5/W10 H3 Top20 timing cells

| Cell | Recorded membership date | Score prices through | Return clock | Top20 return (%) | Pool return (%) | Top20 − pool (pp) | Observed selected / intended |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| α | ≤ t | t close | close(t) → close(t+3) | 0.426989 | 0.109995 | 0.316994 | 46,935 / 46,980 |
| β | ≤ t | t close | open(t+1) → open(t+4) | 0.390197 | 0.113624 | 0.276573 | 46,912 / 46,980 |
| γ | ≤ preceding market day | t close | open(t+1) → open(t+4) | 0.396856 | 0.113624 | 0.283232 | 46,914 / 46,980 |
| δ (internal reference) | ≤ preceding market day | t close | close(t) → close(t+3) | 0.431245 | 0.109995 | 0.321250 | 46,940 / 46,980 |

Complete-observation, equal-date means on 2,349 common dates (22 December 2016–25 August 2026). Top20 and pool returns are percentages; their differences are percentage points (pp). The close references are not demonstrations of same-close execution. Cell means are descriptive; formal inference is attached to the paired contrasts in Supplementary Table S5.

### Supplementary Table S5 | Paired timing contrasts and their shared Holm20 correction

Estimates and interval bounds are complete-H3 percentage points. C means complete observations; Z means missing outcomes set to zero at fixed denominators. All rows use 2,349 dates and 20,000 replicates. Intervals are pointwise rather than simultaneous; one Holm20 family includes every row.

| Contrast | Missingness | Block L | Estimate (pp) | Pointwise 95% interval (pp) | Holm20 P |
| --- | --- | ---: | ---: | --- | ---: |
| β − γ | C | 20 | −0.006659 | [−0.023406, 0.010009] | 1.000000 |
| α − δ | C | 20 | −0.004256 | [−0.020830, 0.012197] | 1.000000 |
| α − β | C | 20 | 0.040421 | [0.018988, 0.061803] | 0.005000 |
| δ − γ | C | 20 | 0.038018 | [0.016461, 0.059703] | 0.007650 |
| interaction | C | 20 | 0.002403 | [−0.003185, 0.008025] | 1.000000 |
| β − γ | Z | 20 | −0.006790 | [−0.023389, 0.009829] | 1.000000 |
| α − δ | Z | 20 | −0.004367 | [−0.020953, 0.012044] | 1.000000 |
| α − β | Z | 20 | 0.040988 | [0.019635, 0.062247] | 0.005000 |
| δ − γ | Z | 20 | 0.038565 | [0.017011, 0.060177] | 0.007200 |
| interaction | Z | 20 | 0.002423 | [−0.003098, 0.008049] | 1.000000 |
| β − γ | C | 60 | −0.006659 | [−0.023528, 0.009632] | 1.000000 |
| α − δ | C | 60 | −0.004256 | [−0.021383, 0.012449] | 1.000000 |
| α − β | C | 60 | 0.040421 | [0.018189, 0.062627] | 0.007650 |
| δ − γ | C | 60 | 0.038018 | [0.015457, 0.060430] | 0.016249 |
| interaction | C | 60 | 0.002403 | [−0.003714, 0.008576] | 1.000000 |
| β − γ | Z | 60 | −0.006790 | [−0.023624, 0.009420] | 1.000000 |
| α − δ | Z | 60 | −0.004367 | [−0.021527, 0.012328] | 1.000000 |
| α − β | Z | 60 | 0.040988 | [0.018816, 0.063014] | 0.007650 |
| δ − γ | Z | 60 | 0.038565 | [0.015977, 0.061029] | 0.015399 |
| interaction | Z | 60 | 0.002423 | [−0.003696, 0.008568] | 1.000000 |

### Supplementary Table S6 | Annual H3 Top20 selection spreads

| Year | Pool dates | FW − pool (pp) | Joint dates | FW − joint (pp) |
| --- | --- | --- | --- | --- |
| 2016 (partial) | 7 | 0.0666 | 7 | 0.1397 |
| 2017 | 244 | 0.2311 | 242 | 0.0454 |
| 2018 | 243 | 0.3386 | 242 | 0.2224 |
| 2019 | 244 | 0.2681 | 244 | 0.2119 |
| 2020 | 243 | 0.4852 | 243 | 0.2774 |
| 2021 | 243 | 0.3599 | 243 | 0.2019 |
| 2022 | 242 | −0.0045 | 242 | 0.0711 |
| 2023 | 242 | 0.5757 | 242 | 0.2932 |
| 2024 | 242 | −0.0704 | 242 | −0.0317 |
| 2025 | 243 | 0.1645 | 243 | 0.1501 |
| 2026 (partial) | 156 | 0.6040 | 156 | 0.2783 |

All estimates are descriptive means of complete three-day gross-return differences. Years 2016 and 2026 cover only the observed part of the year; the 2026 row ends with the 25 August 2026 signal, whose exit is on 31 August 2026. Joint comparisons match industry and daily capitalization decile and retain dates with all required group means. No annual significance search was performed.

### Supplementary Table S7 | Direct contrasts in the main specification

| Comparison | Mean difference | Pointwise 95% interval | Holm P |
| --- | --- | --- | --- |
| FW − F-only | 0.2288 | [0.1213, 0.3361] | 0.02772 (960) |
| FW − W-only | 0.1565 | [0.0331, 0.2834] | 1.0000 (960) |
| FW − A | 0.0006 | [−0.0104, 0.0127] | 1.0000 (192) |
| FW − P | 0.0060 | [−0.0028, 0.0161] | 1.0000 (192) |
| FW − CM5 | 0.2642 | [0.1290, 0.3997] | 0.01620 (384) |
| FW − CM20 | 0.3306 | [0.1543, 0.5080] | 0.04056 (384) |
| FW − BC10 | 1.2800 | [1.0551, 1.5084] | 0.00384 (384) |
| FW − BC60 | 1.1392 | [0.8788, 1.4032] | 0.00384 (384) |

All differences are complete-H3 gross return percentage points, averaged equally over signal dates, not daily profits. γ, market-calendar formation windows, the next-open return clock, F5/W10 and Top20 are fixed. The first four rows use 2,349 mature dates (22 December 2016–25 August 2026); the last four use a stricter five-rule common pool and 2,329 dates (20 January 2017–25 August 2026). Lists are frozen and missing outcomes do not cause replacement. Intervals use 100,000 paired circular block resamples, block length 20 days. Parentheses identify the separate correction families, not a common 960+192+384 correction. CM is raw concept-return averaging; BC prioritizes regression slopes and then stock momentum. These daily adaptations are not the original published strategies.

### Supplementary Table S8 | Full-training-date H3 paired increments

Block length 20.

| Model | Outcome | Estimate | Pointwise 95% interval | Holm24 P |
| --- | --- | --- | --- | --- |
| Ridge | Raw IC | 0.001668 | [−0.005248, 0.008495] | 1.0000 |
| Ridge | Industry IC | 0.002869 | [−0.000632, 0.006458] | 1.0000 |
| Ridge | Top20 observed | 0.099154 | [−0.003358, 0.203532] | 1.0000 |
| Ridge | Top20 zero | 0.096657 | [−0.003990, 0.198127] | 1.0000 |
| Ridge | Top100 observed | 0.049920 | [−0.011478, 0.112653] | 1.0000 |
| Ridge | Top100 zero | 0.050248 | [−0.011050, 0.113152] | 1.0000 |
| Boosting | Raw IC | 0.003311 | [−0.001637, 0.008211] | 1.0000 |
| Boosting | Industry IC | 0.005414 | [0.002725, 0.008111] | 0.0024 |
| Boosting | Top20 observed | 0.010073 | [−0.106148, 0.117153] | 1.0000 |
| Boosting | Top20 zero | 0.009685 | [−0.106431, 0.116002] | 1.0000 |
| Boosting | Top100 observed | 0.028080 | [−0.045440, 0.098302] | 1.0000 |
| Boosting | Top100 zero | 0.028447 | [−0.045095, 0.098503] | 1.0000 |

Block length 60.

| Model | Outcome | Estimate | Pointwise 95% interval | Holm24 P |
| --- | --- | --- | --- | --- |
| Ridge | Raw IC | 0.001668 | [−0.004934, 0.008466] | 1.0000 |
| Ridge | Industry IC | 0.002869 | [−0.000675, 0.006600] | 1.0000 |
| Ridge | Top20 observed | 0.099154 | [−0.013499, 0.214830] | 1.0000 |
| Ridge | Top20 zero | 0.096657 | [−0.013519, 0.209075] | 1.0000 |
| Ridge | Top100 observed | 0.049920 | [−0.014507, 0.115713] | 1.0000 |
| Ridge | Top100 zero | 0.050248 | [−0.013976, 0.115785] | 1.0000 |
| Boosting | Raw IC | 0.003311 | [−0.002039, 0.008763] | 1.0000 |
| Boosting | Industry IC | 0.005414 | [0.002388, 0.008413] | 0.0230 |
| Boosting | Top20 observed | 0.010073 | [−0.103309, 0.111629] | 1.0000 |
| Boosting | Top20 zero | 0.009685 | [−0.102621, 0.110814] | 1.0000 |
| Boosting | Top100 observed | 0.028080 | [−0.053933, 0.105585] | 1.0000 |
| Boosting | Top100 zero | 0.028447 | [−0.053340, 0.105746] | 1.0000 |

All rows compare enhanced minus ordinary versions of the same model on 1,611 dates (2 January 2020–25 August 2026). Return units are complete-three-day percentage points; IC units are correlations. Zero denotes a fixed-denominator missing-outcome convention. Holm24 includes both block lengths and all twelve contrasts. The two passing rows measure the same industry-conditioned boosting contrast.

### Supplementary Table S9 | F and FW added jointly to W and ordinary predictors

| Endpoint | Block | Mean difference | Pointwise 95% interval | Raw P | Holm6 P |
| --- | ---: | ---: | --- | ---: | ---: |
| All dates | 20 | −0.0225 | [−0.1131, 0.0660] | 0.6211 | 1.0000 |
| All dates | 60 | −0.0225 | [−0.1165, 0.0700] | 0.6358 | 1.0000 |
| Stress | 20 | NE | — | — | — |
| Stress | 60 | NE | — | — | — |
| Stress − nonstress | 20 | NE | — | — | — |
| Stress − nonstress | 60 | NE | — | — | — |

Differences are augmented minus baseline Top20 mean gross returns in percentage points per full H3 period, averaged equally over formation dates. Primary means use observed outcomes on fixed lists. NE denotes not estimable because the stress-support gate failed; correction placeholders of one are not estimated P values. Pointwise intervals are unadjusted, and L20/L60 are dependence sensitivities rather than independent replications. This six-test family does not replace the earlier component or prediction families.

### Supplementary Table S10 | Synchronization increments with and without observed market risk

| Contrast | Block | Mean difference | Pointwise 95% interval | Raw P | Holm6 P |
| --- | ---: | ---: | --- | ---: | ---: |
| W increment: B−A | 20 | 0.0742 | [−0.0034, 0.1547] | 0.0659 | 0.3373 |
| Risk-conditioned W: D−C | 20 | 0.0032 | [−0.0790, 0.0871] | 0.9405 | 1.0000 |
| Direct change: (D−C)−(B−A) | 20 | −0.0711 | [−0.1566, 0.0082] | 0.0903 | 0.3373 |
| W increment: B−A | 60 | 0.0742 | [−0.0007, 0.1518] | 0.0562 | 0.3373 |
| Risk-conditioned W: D−C | 60 | 0.0032 | [−0.0756, 0.0871] | 0.9398 | 1.0000 |
| Direct change: (D−C)−(B−A) | 60 | −0.0711 | [−0.1524, 0.0049] | 0.0755 | 0.3373 |

Differences are percentage points per complete H3 period, averaged equally over the same 1,611 formation dates. All four models include the same missing-risk summary. Pointwise intervals and family-adjusted P values answer different inferential questions. This predictive comparison is conditional on the specified feature sets and single-market proxy; it is not a causal risk decomposition.

### Supplementary Table S11 | Support and correction in the temporal audit

| Workflow and method | Planned | Estimable | Deferred | Holm | Secondary BH |
| :------------------------ | ------: | ------: | ------: | -----: | ------: |
| Global preprocessing diagnostic: Tree-DML | 144 | 84 | 60 | 0 | 6 |
| Past calibration, fixed 80 rounds: Tree-DML | 144 | 84 | 60 | 0 | 0 |
| Global preprocessing diagnostic: AIPW | 96 | 42 | 54 | 1 | 2 |
| Training-fold encoding, fixed 80 rounds: AIPW | 96 | 42 | 54 | 0 | 0 |
| S-coverage window: Tree-DML | 144 | 126 | 18 | 0 | 0 |
| S-coverage window: AIPW | 96 | 36 | 60 | 0 | 0 |

DML uses six separate 24-test families: own and peer exposures under each of α, β and γ. AIPW uses three separate 32-test families, one per design, each with four states and eight horizons. Counts are not based on single 144- or 96-test corrections. Deferred estimates remain blank and enter correction with placeholder P = 1. The Holm and secondary BH columns count diagnostic adjusted P or q < 0.05. Global preprocessing is diagnostic; the past-information and S-coverage protocols provide temporally calibrated conditional comparisons. Role-specific nuisance fitting supports in the retained DML implementation do not establish the standard orthogonal-score inference guarantee (Section S13). Signals end on 27 August 2026 and outcomes must mature by 31 August 2026. H250 has no mature final-fold test outcomes. The S-coverage branch calibrates from 30 March 2022 to before 30 September 2022 and evaluates four folds from 30 September 2022 on common finite T/S/K exposures. Its DML estimates extend through H120; AIPW at H120 fails the first-fold training-size gate within the high and low groups, and S/joint-state comparisons fail overlap. All branches share observations and are not independent replications.

### Supplementary Table S12 | Own-popularity residual projections on a common γ/H3 sample

| Specification | T | S | K |
| --- | ---: | ---: | ---: |
| M0 · Original role-specific supports | +0.087976 | −0.088081 | −0.042007 |
| M1 · Aligned support and label purging | +0.092221 | −0.089768 | −0.043204 |
| M2 · Conventional controls at the signal date | +0.095007 | −0.074457 | −0.046100 |
| M3 · Same controls before attention | +0.063678 | −0.089530 | −0.071321 |

Coefficients use the same 217,831 evaluation stock-days and 185 dates. Each column is a separate exposure projection, in complete-H3 return percentage points per its original historical standard deviation. M1 changes both auxiliary support and outcome-label purging; M2 removes four concept controls; M3 changes the date of the same remaining 22 characteristics and industry. Coefficient values do not change with the diagnostic HAC lag. The complete 36 coefficient tests and 27 paired-contrast tests include lags 1, 8 and 12 in their respective Holm families; none passes. Different exposure scales preclude interpreting cross-column magnitudes as relative attention effects.

### Supplementary Table S13 | Implementation backtest grid

| Holding | F/W | Similarity | N | Timing | CAGR (%) | Max DD (%) | Sharpe | Fees (%/yr) |
| -------- | ------- | ---------- | -----: | ----------- | ---------: | ---------: | --------: | ---------: |
| 3 days | 5/10 | Pearson | 10 | None | 15.0 | −42.1 | 0.63 | 8.0 |
| 3 days | 5/10 | Pearson | 10 | Breadth | 21.9 | −19.6 | 1.15 | 4.8 |
| 3 days | 5/10 | Pearson | 10 | Two-level | 29.8 | −18.5 | 1.44 | 5.1 |
| 3 days | 5/10 | Pearson | 5 | None | 14.6 | −48.8 | 0.60 | 8.2 |
| 3 days | 5/10 | Pearson | 5 | Breadth | 23.2 | −25.0 | 1.13 | 4.9 |
| 3 days | 5/10 | Pearson | 5 | Two-level | 30.0 | −20.9 | 1.35 | 5.2 |
| 3 days | 5/10 | CCC | 10 | None | 21.7 | −42.3 | 0.87 | 7.8 |
| 3 days | 5/10 | CCC | 10 | Breadth | 24.9 | −18.8 | 1.37 | 4.7 |
| 3 days | 5/10 | CCC | 10 | Two-level | 29.8 | −16.6 | 1.54 | 5.1 |
| 3 days | 5/10 | CCC | 5 | None | 23.8 | −46.5 | 0.89 | 8.1 |
| 3 days | 5/10 | CCC | 5 | Breadth | 26.4 | −20.7 | 1.35 | 4.8 |
| 3 days | 5/10 | CCC | 5 | Two-level | 30.9 | −17.1 | 1.49 | 5.2 |
| 3 days | 10/20 | Pearson | 10 | None | 19.6 | −42.3 | 0.74 | 6.3 |
| 3 days | 10/20 | Pearson | 10 | Breadth | 26.6 | −23.6 | 1.28 | 4.2 |
| 3 days | 10/20 | Pearson | 10 | Two-level | 33.5 | −19.2 | 1.52 | 4.6 |
| 3 days | 10/20 | Pearson | 5 | None | 12.7 | −59.6 | 0.53 | 6.5 |
| 3 days | 10/20 | Pearson | 5 | Breadth | 22.1 | −36.6 | 1.03 | 4.3 |
| 3 days | 10/20 | Pearson | 5 | Two-level | 29.1 | −26.1 | 1.25 | 4.6 |
| 3 days | 10/20 | CCC | 10 | None | 21.8 | −42.0 | 0.83 | 6.1 |
| 3 days | 10/20 | CCC | 10 | Breadth | 26.0 | −21.5 | 1.35 | 4.2 |
| 3 days | 10/20 | CCC | 10 | Two-level | 31.6 | −17.3 | 1.53 | 4.5 |
| 3 days | 10/20 | CCC | 5 | None | 19.0 | −43.9 | 0.72 | 6.4 |
| 3 days | 10/20 | CCC | 5 | Breadth | 26.3 | −26.0 | 1.26 | 4.3 |
| 3 days | 10/20 | CCC | 5 | Two-level | 29.2 | −19.8 | 1.32 | 4.6 |
| 5 days | 5/10 | Pearson | 10 | None | 16.1 | −37.8 | 0.67 | 5.4 |
| 5 days | 5/10 | Pearson | 10 | Breadth | 19.9 | −15.9 | 1.12 | 3.1 |
| 5 days | 5/10 | Pearson | 10 | Two-level | 22.9 | −16.9 | 1.22 | 3.3 |
| 5 days | 5/10 | Pearson | 5 | None | 14.4 | −43.7 | 0.60 | 5.5 |
| 5 days | 5/10 | Pearson | 5 | Breadth | 20.3 | −20.5 | 1.08 | 3.1 |
| 5 days | 5/10 | Pearson | 5 | Two-level | 21.9 | −23.6 | 1.11 | 3.3 |
| 5 days | 5/10 | CCC | 10 | None | 21.4 | −42.4 | 0.87 | 5.4 |
| 5 days | 5/10 | CCC | 10 | Breadth | 22.3 | −19.1 | 1.32 | 3.0 |
| 5 days | 5/10 | CCC | 10 | Two-level | 24.3 | −14.6 | 1.36 | 3.3 |
| 5 days | 5/10 | CCC | 5 | None | 22.8 | −43.0 | 0.87 | 5.4 |
| 5 days | 5/10 | CCC | 5 | Breadth | 24.0 | −18.2 | 1.33 | 3.1 |
| 5 days | 5/10 | CCC | 5 | Two-level | 25.1 | −15.0 | 1.32 | 3.3 |
| 5 days | 10/20 | Pearson | 10 | None | 20.3 | −39.7 | 0.77 | 4.5 |
| 5 days | 10/20 | Pearson | 10 | Breadth | 23.7 | −20.9 | 1.23 | 2.7 |
| 5 days | 10/20 | Pearson | 10 | Two-level | 25.8 | −17.7 | 1.28 | 3.0 |
| 5 days | 10/20 | Pearson | 5 | None | 16.6 | −54.1 | 0.64 | 4.6 |
| 5 days | 10/20 | Pearson | 5 | Breadth | 20.7 | −36.3 | 1.03 | 2.8 |
| 5 days | 10/20 | Pearson | 5 | Two-level | 22.1 | −28.2 | 1.05 | 3.0 |
| 5 days | 10/20 | CCC | 10 | None | 21.5 | −37.2 | 0.84 | 4.4 |
| 5 days | 10/20 | CCC | 10 | Breadth | 24.1 | −17.3 | 1.34 | 2.7 |
| 5 days | 10/20 | CCC | 10 | Two-level | 24.7 | −15.2 | 1.30 | 3.0 |
| 5 days | 10/20 | CCC | 5 | None | 16.8 | −41.6 | 0.67 | 4.6 |
| 5 days | 10/20 | CCC | 5 | Breadth | 22.4 | −20.9 | 1.18 | 2.7 |
| 5 days | 10/20 | CCC | 5 | Two-level | 22.2 | −17.6 | 1.12 | 3.0 |

Each row is an archived retrospective simulation with the execution-day full-session target filter (Section S17). Runs extend from 1 December 2016 (8 December for F10/W20) to 31 August 2026 with CNY 30,000,000, three or five staggered sub-portfolios, simulated execution at 09:55, a 1.2-basis-point commission, 0.1% stamp duty on sales and no slippage. N is the list size. CAGR uses the actual start and end dates, and maximum drawdowns (Max DD) are recomputed from daily net values. Sharpe ratios use a zero risk-free rate. Fees are commissions and stamp duty divided by mean assets, per year; mean exposure, commission-implied turnover and final net values of every run are in the source data. Breadth and two-level denote the one-level and two-level breadth timing rules (Supplementary Section S17). No inference is attached.

### Supplementary Table S14 | Calendar-year returns of the reference configurations

| Year | Top10 none | Top10 breadth | Top10 two-level | Top5 none | Top5 breadth | Top5 two-level | CSI 300 | CSI 500 | CSI 1000 |
| ------------ | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| 2016 (December) | −6.1 | −3.3 | −5.4 | −3.7 | −3.2 | −3.8 | −7.2 | −5.4 | −6.2 |
| 2017 | −17.4 | 8.5 | 9.8 | −24.8 | 5.9 | 11.2 | 21.8 | −0.2 | −17.4 |
| 2018 | −12.1 | −0.6 | 5.9 | −15.9 | −3.7 | 6.1 | −25.3 | −33.3 | −36.9 |
| 2019 | 31.7 | 51.0 | 29.1 | 29.1 | 52.4 | 30.4 | 36.1 | 26.4 | 25.7 |
| 2020 | 66.0 | 51.7 | 64.5 | 92.2 | 67.0 | 74.3 | 27.2 | 20.9 | 19.4 |
| 2021 | 40.9 | 16.5 | 40.5 | 28.1 | 12.0 | 31.5 | −5.2 | 15.6 | 20.5 |
| 2022 | −29.9 | −6.7 | −1.0 | −42.3 | −16.6 | −11.6 | −21.6 | −20.3 | −21.6 |
| 2023 | 82.3 | 52.6 | 55.1 | 109.7 | 70.8 | 76.3 | −11.4 | −7.4 | −6.3 |
| 2024 | −9.4 | 30.4 | 55.1 | −5.6 | 37.6 | 57.4 | 14.7 | 5.5 | 1.2 |
| 2025 | 39.8 | 27.4 | 29.4 | 46.2 | 37.4 | 33.0 | 17.7 | 30.4 | 27.5 |
| 2026 (to 31 August) | 15.4 | 5.0 | 25.7 | 17.3 | 0.8 | 16.5 | −0.1 | 6.5 | 2.3 |

The archived accounts include the execution-day full-session target filter described in Section S17. Returns are expressed as percentages and calculated from daily net values after fees for the three-day holding-period, F5/W10, Pearson configurations. The 2016 row covers December and the 2026 row ends on 31 August. Index returns are price returns without dividends. Annual values are descriptive.
