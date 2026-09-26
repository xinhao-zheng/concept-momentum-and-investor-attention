"""
Stock Quant Strategy Framework
---------------------------------------------------
Author:    Xinhao Zheng
Contact:   Veritas428 (WeChat)
Copyright: (c) 2026 Xinhao Zheng. Licensed under the MIT License.
---------------------------------------------------
"""
import pandas as pd

# param → Shenwan industry column in pro quotes
_INDUSTRY_COLUMNS = {
    '一级行业': '新版申万一级行业名称',
    '二级行业': '新版申万二级行业名称',
    '三级行业': '新版申万三级行业名称',
}

fin_cols = []
ov_cols = list(_INDUSTRY_COLUMNS.values())
extra_data = {}


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    Compute the factor and return it as a single-column DataFrame.

    :param df: Daily K-line of one stock, ascending by trade date; columns declared in fin_cols / extra_data
        are merged in by the host.
    :param param: Factor parameter; see below.
    :param kwargs: col_name — the output column name.
    :return: pd.DataFrame with the single column col_name, on the index and length of df. The function does
        not modify df.

    Industry Classification Factor
    ---------------------------------------------------
    Meaning: The Shenwan industry name at the selected level.
    Principle: The second factor_list item selects the industry level; the strategy constructs groups from the
         full market. ov_cols declares the pro-quote columns required for all three levels.
    Formula: 新版申万一级行业名称 / 新版申万二级行业名称 / 新版申万三级行业名称 (passthrough selected by param)
    param: '一级行业', '二级行业' or '三级行业'; no default.
    Sorting: Not applicable (args=0 is a placeholder; this column does not enter the companion strategy's ranking)
    Boundary: Requires stock-trading-data-pro quotes. Missing labels remain missing; missing columns raise
         KeyError. No warm-up rows are added. Gaps are not filled and current labels do not replace history.
         Raw fields lack publication times or ingestion versions; strict point-in-time availability cannot be
         established from these fields.
    Selection Case: ('z_IndustryClassification_en', False, '一级行业', 0)
    """
    col_name = kwargs['col_name']
    if not isinstance(param, str) or param not in _INDUSTRY_COLUMNS:
        raise ValueError(f"param must be '一级行业', '二级行业' or '三级行业', got {param!r}")

    return pd.DataFrame({col_name: df[_INDUSTRY_COLUMNS[param]]}, index=df.index)
