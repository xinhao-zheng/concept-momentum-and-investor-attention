"""
Stock Quant Strategy Framework
---------------------------------------------------
Author:    Xinhao Zheng
Contact:   Veritas428 (WeChat)
Copyright: (c) 2026 Xinhao Zheng. Licensed under the MIT License.
---------------------------------------------------
"""
import pandas as pd

# param → 全息行情的申万行业列
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
    计算因子，以单列 DataFrame 返回。

    :param df: 单只股票的日频 K 线，按交易日期升序；fin_cols / extra_data 声明的列已由框架合并入 df。
    :param param: 因子参数，含义见下。
    :param kwargs: col_name —— 输出列名。
    :return: pd.DataFrame，仅含 col_name 一列，与 df 同索引、同长度。本函数不修改 df。

    行业分类因子
    ---------------------------------------------------
    含义：所选层级的申万行业名称。
    原理：作为 factor_list 第二项选择行业层级，策略另从全市场构造板块；ov_cols 声明三个层级所需的全息行情列。
    公式：新版申万一级行业名称 / 新版申万二级行业名称 / 新版申万三级行业名称（按 param 透传）
    param: '一级行业'、'二级行业' 或 '三级行业'，无默认值。
    排序：不适用（args=0 占位，本列不参与配套策略排名）
    边界：要求 stock-trading-data-pro 行情；缺失标签保持缺失，缺列时报 KeyError，无预热行。
         不填充缺口或用最新标签回填历史；原始字段不含发布时间或入库版本，无法据此核验严格时点可得性。
    选股因子案例：('z_行业分类_zh', False, '一级行业', 0)
    """
    col_name = kwargs['col_name']
    if not isinstance(param, str) or param not in _INDUSTRY_COLUMNS:
        raise ValueError(f"param 仅接受 '一级行业'、'二级行业' 或 '三级行业'，收到 {param!r}")

    return pd.DataFrame({col_name: df[_INDUSTRY_COLUMNS[param]]}, index=df.index)
