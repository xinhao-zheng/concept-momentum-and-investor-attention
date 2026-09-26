"""
Stock Quant Strategy Framework
---------------------------------------------------
Author:    Xinhao Zheng
Contact:   Veritas428 (WeChat)
Copyright: (c) 2026 Xinhao Zheng. Licensed under the MIT License.
---------------------------------------------------
"""
import pandas as pd

fin_cols = []
extra_data = {'z_概念数据': ['z_概念归属']}


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    Compute the factor and return it as a single-column DataFrame.

    :param df: Daily K-line of one stock, ascending by trade date; columns declared in fin_cols / extra_data
        are merged in by the host.
    :param param: Factor parameter; see below.
    :param kwargs: col_name — the output column name.
    :return: pd.DataFrame with the single column col_name, on the index and length of df. The function does
        not modify df.

    Concept Membership Factor
    ---------------------------------------------------
    Meaning: Concept membership labels aligned by the external interface.
    Principle: The second factor_list item selects concept mode; the strategy constructs groups from the full
         market. extra_data declares the concept data dependency for host loading and client update detection.
    Formula: z_概念归属 (passthrough)
    param: Only '概念'; no default.
    Sorting: Not applicable (args=0 is a placeholder; this column does not enter the companion strategy's ranking)
    Boundary: Empty labels remain '', missing records remain pd.NA; old membership is not carried forward.
         Missing columns raise KeyError; no warm-up rows are added. The external interface aligns history to
         prior-market-trading-day labels. The client uses the market-wide final quote date and availability
         cutoff to select the latest valid membership, updating only the current decision. Raw CSVs lack snapshot
         publication times; record dates alone cannot establish strict point-in-time availability.
    Selection Case: ('z_ConceptMembership_en', False, '概念', 0)
    """
    col_name = kwargs['col_name']
    if not isinstance(param, str) or param != '概念':
        raise ValueError(f"param must be '概念', got {param!r}")

    return pd.DataFrame({col_name: df['z_概念归属']}, index=df.index)
