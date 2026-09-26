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
    计算因子，以单列 DataFrame 返回。

    :param df: 单只股票的日频 K 线，按交易日期升序；fin_cols / extra_data 声明的列已由框架合并入 df。
    :param param: 因子参数，含义见下。
    :param kwargs: col_name —— 输出列名。
    :return: pd.DataFrame，仅含 col_name 一列，与 df 同索引、同长度。本函数不修改 df。

    概念归属因子
    ---------------------------------------------------
    含义：外部接口对齐后的概念成员标签。
    原理：作为 factor_list 第二项选择概念模式，策略另从全市场构造板块。extra_data 声明概念数据依赖，
         供框架加载及客户端识别产品更新。
    公式：z_概念归属（直接透传）
    param: 仅接受 '概念'，无默认值。
    排序：不适用（args=0 占位，本列不参与配套策略排名）
    边界：空标签为 ''，缺失记录为 pd.NA，不延用旧成员；缺列时报 KeyError，无预热行。
         外部接口按前一市场交易日对齐历史；客户端按全市场末行情日及允许时点取最新有效成员，仅更新当前决策。
         原始 CSV 不含快照发布时间，无法仅据记录日期核验严格时点可得性。
    选股因子案例：('z_概念归属_zh', False, '概念', 0)
    """
    col_name = kwargs['col_name']
    if not isinstance(param, str) or param != '概念':
        raise ValueError(f"param 仅接受 '概念'，收到 {param!r}")

    return pd.DataFrame({col_name: df['z_概念归属']}, index=df.index)
