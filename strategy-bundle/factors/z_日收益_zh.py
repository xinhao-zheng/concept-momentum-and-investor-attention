"""
Stock Quant Strategy Framework
---------------------------------------------------
Author:    Xinhao Zheng
Contact:   Veritas428 (WeChat)
Copyright: (c) 2026 Xinhao Zheng. Licensed under the MIT License.
---------------------------------------------------
"""
import pandas as pd

# 配套策略的默认窗口与相似度算法
_DEFAULT_PARAM = (5, 10, 'pearson')
_SIMILARITIES = {'pearson', 'ccc'}

fin_cols = []
extra_data = {}


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算因子，以单列 DataFrame 返回。

    :param df: 单只股票的日频 K 线，按交易日期升序；fin_cols / extra_data 声明的列已由框架合并入 df。
    :param param: 因子参数，含义见下。
    :param kwargs: col_name —— 输出列名。
    :return: pd.DataFrame，仅含 col_name 一列，与 df 同索引、同长度。本函数不修改 df。

    日收益因子
    ---------------------------------------------------
    含义：复权日收益；param 为注意力动量策略提供窗口与相似度设置。
    原理：相邻复权价格之比衡量当日收益，因子值大 ⇔ 当日复权涨幅大。作为 factor_list 第一项，策略读取其参数，
         另从全市场构造板块收益，避免候选过滤改变板块收益。
    公式：收盘价_复权[t] / 收盘价_复权[t-1] - 1
    param: (F, W, similarity)，默认 (5, 10, 'pearson')；None 或 '' 使用默认值。
         F 为板块动量窗口，F >= 1；W 为相似度窗口，W >= 2；均接受整数或整数字符串。
         similarity 为 'pearson' 或 'ccc'；这些参数不改变本因子的日收益序列。
    排序：不适用（本列不参与配套策略排名）
    边界：首行收益为 NaN；复权价格非正、非数值或无穷时，当行及下一行收益为 NaN，不填充价格缺口。
         无穷收益为 NaN；存在是否交易列时，仅保留值为 1 的收益。缺少收盘价_复权时报错。
         F、W 不增加预热行；非空 minutes 报错，分钟择时由 timing 配置。
    选股因子案例：('z_日收益_zh', False, (5, 10, 'pearson'), 1)
    """
    col_name = kwargs['col_name']
    if param is None or (isinstance(param, str) and param == ''):
        param = _DEFAULT_PARAM
    if not isinstance(param, (list, tuple)) or len(param) != 3:
        raise ValueError(f'param 须为 (F, W, similarity)，收到 {param!r}')
    raw_f, raw_w, similarity = param
    if any(isinstance(value, bool) or not isinstance(value, (int, str)) for value in (raw_f, raw_w)):
        raise ValueError(f'param 的 F、W 须为整数或整数字符串，收到 {param!r}')
    try:
        f_window, w_window = int(raw_f), int(raw_w)
    except ValueError as error:
        raise ValueError(f'param 的 F、W 须为整数或整数字符串，收到 {param!r}') from error
    if f_window < 1 or w_window < 2:
        raise ValueError(f'param 须满足 F >= 1 且 W >= 2，收到 {param!r}')
    if not isinstance(similarity, str) or similarity not in _SIMILARITIES:
        raise ValueError(f"param 的 similarity 仅接受 'pearson' 或 'ccc'，收到 {param!r}")
    if kwargs.get('minutes'):
        raise ValueError('本因子要求日频数据；分钟择时由 timing 配置')
    if '收盘价_复权' not in df:
        raise ValueError('缺少收盘价_复权，不能以未复权价格替代')

    price = pd.to_numeric(df['收盘价_复权'], errors='coerce')
    price = price.where(price.gt(0) & price.lt(float('inf')))
    daily_return = price.pct_change(fill_method=None)
    daily_return = daily_return.where(daily_return.abs().lt(float('inf')))
    if '是否交易' in df:
        daily_return = daily_return.where(df['是否交易'].eq(1))

    return pd.DataFrame({col_name: daily_return}, index=df.index)
