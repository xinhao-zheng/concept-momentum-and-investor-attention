"""
Stock Quant Strategy Framework
---------------------------------------------------
Author:    Xinhao Zheng
Contact:   Veritas428 (WeChat)
Copyright: (c) 2026 Xinhao Zheng. Licensed under the MIT License.
---------------------------------------------------
"""
import pandas as pd

# Default windows and similarity for the companion strategy
_DEFAULT_PARAM = (5, 10, 'pearson')
_SIMILARITIES = {'pearson', 'ccc'}

fin_cols = []
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

    Daily Return Factor
    ---------------------------------------------------
    Meaning: Adjusted daily returns; param supplies windows and similarity for the Attention Momentum Strategy.
    Principle: The ratio of adjacent adjusted prices measures daily returns. A large value ⇔ a larger adjusted
         daily return. As the first factor_list item, it supplies parameters to the strategy, which constructs
         group returns from the full market so candidate filters do not alter them.
    Formula: 收盘价_复权[t] / 收盘价_复权[t-1] - 1
    param: (F, W, similarity), default (5, 10, 'pearson'); None or '' uses the default.
         F is the group momentum window, F >= 1; W is the similarity window, W >= 2.
         Both accept integers or integer strings. similarity is 'pearson' or 'ccc'; settings do not affect returns.
    Sorting: Not applicable (this column does not enter the companion strategy's ranking)
    Boundary: The first return is NaN. Nonpositive, nonnumeric or infinite adjusted prices make returns NaN on
         that row and the next; price gaps are not filled. Infinite returns are NaN. If 是否交易 exists, retain
         returns only where it equals 1. Missing 收盘价_复权 raises an error. F and W add no warm-up rows;
         nonempty minutes raises an error, and timing configures intraday timing.
    Selection Case: ('z_DailyReturn_en', False, (5, 10, 'pearson'), 1)
    """
    col_name = kwargs['col_name']
    if param is None or (isinstance(param, str) and param == ''):
        param = _DEFAULT_PARAM
    if not isinstance(param, (list, tuple)) or len(param) != 3:
        raise ValueError(f'param must be (F, W, similarity), got {param!r}')
    raw_f, raw_w, similarity = param
    if any(isinstance(value, bool) or not isinstance(value, (int, str)) for value in (raw_f, raw_w)):
        raise ValueError(f'param F and W must be integers or integer strings, got {param!r}')
    try:
        f_window, w_window = int(raw_f), int(raw_w)
    except ValueError as error:
        raise ValueError(f'param F and W must be integers or integer strings, got {param!r}') from error
    if f_window < 1 or w_window < 2:
        raise ValueError(f'param must satisfy F >= 1 and W >= 2, got {param!r}')
    if not isinstance(similarity, str) or similarity not in _SIMILARITIES:
        raise ValueError(f"param similarity must be 'pearson' or 'ccc', got {param!r}")
    if kwargs.get('minutes'):
        raise ValueError('This factor requires daily data; timing configures intraday timing')
    if '收盘价_复权' not in df:
        raise ValueError('Missing 收盘价_复权; unadjusted prices cannot replace it')

    price = pd.to_numeric(df['收盘价_复权'], errors='coerce')
    price = price.where(price.gt(0) & price.lt(float('inf')))
    daily_return = price.pct_change(fill_method=None)
    daily_return = daily_return.where(daily_return.abs().lt(float('inf')))
    if '是否交易' in df:
        daily_return = daily_return.where(df['是否交易'].eq(1))

    return pd.DataFrame({col_name: daily_return}, index=df.index)
