# Support gates

The H3 comparison requires finite future T/S/K values, at least 30 stocks and defined baseline/fixed Spearman correlations for all three measures. A constant cross-section yields an undefined correlation, not zero.

The code records counts for maturity, insufficient stocks, undefined correlations and included dates at every horizon. Recalculation uses the same cached raw values. The main current/past decomposition, rank lists and sign statistics do not depend on future fields or these correlation gates.
