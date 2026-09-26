"""
Stock Quant Strategy Framework
---------------------------------------------------
Author:    Xinhao Zheng
Contact:   Veritas428 (WeChat)
Copyright: (c) 2026 Xinhao Zheng. Licensed under the MIT License.
---------------------------------------------------
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config
from core.model.strategy_config import StrategyConfig
from 外部数据.z_概念数据 import align_membership, read_calendar, read_decision_context, read_membership

# 收益标准差阈值，避免浮点噪声拆散并列
_EPS = 1e-10

STG_INTRO = {
    '策略说明': """
    注意力动量策略，以板块动量与个股、板块收益的相似度构造核心分数。

    factor_list 解析规则：
        - 第一项为配套日收益因子，param=(F, W, similarity)，args 为核心排名权重；
        - 第二项为配套概念归属或行业分类因子，param 选择 概念 / 一级行业 / 二级行业 / 三级行业，args=0；
        - 其余项为辅助排名因子，args 为权重，0 表示停用。
    F 为板块累计收益窗口，板块日收益取全市场成员等权均值；W 为个股与板块收益的相似度窗口。
    F、W 为整数，F ≥ 1、W ≥ 2，默认 (5, 10, 'pearson')。
    similarity 可选 'pearson'（相关系数）或 'ccc'（一致性相关系数）。

    核心分数 = mean(所属板块 F 分位 × 个股与该板块 W 分位)。F 在板块间排名，W 在股票与板块成员对间排名。
    复合因子 = 核心分数排名 × 核心权重 + Σ(辅助因子排名 × 权重)，越小越优；排名方向取各因子的 ascending 位。

    建议的 filter_list：
        ('成交额Mean', 5, 'val:>=2000_0000', True)    近 5 日平均成交额不少于 2000 万元；可选，示例默认关闭

    已知限制：板块收益包含股票自身，不随候选过滤改变。历史概念取前一市场交易日成员；客户端当前截面在
    允许时点内取最新成员，缺失、空标签或过旧记录不续接。行业取行情当日分类。数据无历史发布时间与版本，
    不能证明严格时点可得性或重建历史周末补齐决策。排名取 method='min'，实际选股数可能超过 select_num，
    与框架 select_by_factor 的规则一致。日内择时由 timing 配置，执行时点须早于买入。

    用例-1：概念 Pearson，3 日调仓；用例-2：二级行业 CCC，叠加成交额排名。
    """,
    '使用案例-1':
        {
            'name': 'z_注意力动量策略_zh',
            'hold_period': '3D',
            'offset_list': [0, 1, 2],
            'select_num': 5,
            'cap_weight': 1,
            'rebalance_time': 'open',
            'factor_list': [('z_日收益_zh', False, (5, 10, 'pearson'), 1),
                            ('z_概念归属_zh', False, '概念', 0),
                            ],
            'filter_list': [],
        },
    '使用案例-2':
        {
            'name': 'z_注意力动量策略_zh',
            'hold_period': '3D',
            'offset_list': [0, 1, 2],
            'select_num': 5,
            'cap_weight': 1,
            'rebalance_time': 'open',
            'factor_list': [('z_日收益_zh', False, (5, 20, 'ccc'), 1),
                            ('z_行业分类_zh', False, '二级行业', 0),
                            ('成交额Mean', False, 20, 0.5),
                            ],
            'filter_list': [],
        },
}


def _load_market(end, mode):
    """读取截至 end 的全市场历史；候选过滤与上市天数不改变板块收益。"""
    root = Path(config.data_center_path)
    folder = root / 'stock-trading-data-pro'
    calendar = read_calendar() if mode == '概念' else None
    latest_date, asof = read_decision_context() if mode == '概念' else (None, None)
    if latest_date is not None:
        end = min(pd.Timestamp(end), latest_date)
    industry = None if mode == '概念' else f'新版申万{mode}名称'
    columns = ['交易日期', '收盘价', '前收盘价'] + ([industry] if industry else [])

    def read_stock(path):
        data = pd.read_csv(path, encoding='gb18030', skiprows=1, usecols=columns).sort_values('交易日期')
        dates = pd.to_datetime(data['交易日期'])
        close, previous = data['收盘价'], data['前收盘价']
        returns = (close / previous - 1).where(close.gt(0) & previous.gt(0))
        returns = returns.replace([np.inf, -np.inf], np.nan)
        if returns.notna().any():
            returns.loc[returns.first_valid_index()] = np.nan
        if industry:
            labels = data[industry]
        else:
            membership = read_membership(root / 'stock-popular-concept-detail' / path.name)
            labels = align_membership(dates, membership, calendar, latest_date, asof)
        panel = pd.DataFrame({'交易日期': dates, '股票代码': path.stem, '收益': returns, '板块': labels})
        return panel.loc[dates.le(end) & returns.notna() & labels.notna() & labels.ne('')]

    paths = sorted(folder.glob('*.csv'))
    if not paths:
        raise FileNotFoundError(f'缺少专业版日线数据：{folder}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        panel = pd.concat(pool.map(read_stock, paths), ignore_index=True)

    return panel


def _rolling_similarity(pairs, window, method):
    """
    按股票与板块的有效成员观测滚动；退出后重新加入沿用历史，缺失收益不计入窗口。
    W ≤ 5 时取满窗，否则至少 max(5, W // 2) 个观测。
    CCC = 2 × cov(x, y) / (var(x) + var(y) + (mean(x) − mean(y))²)，方差与协方差取 ddof=1。
    标准差 ≤ _EPS 视为常量；Pearson 无定义，CCC 协方差为 0，分母 ≤ _EPS² 时无定义。
    """
    values = np.full(len(pairs), np.nan)
    codes = pairs['股票代码'].cat.codes.to_numpy()
    groups = pairs['板块'].cat.codes.to_numpy()
    edges = np.r_[0, np.flatnonzero((codes[1:] != codes[:-1]) | (groups[1:] != groups[:-1])) + 1, len(pairs)]
    returns = pairs['收益'].to_numpy(dtype='float64')
    group_returns = pairs['板块收益'].to_numpy(dtype='float64')
    minimum = window if window <= 5 else max(5, window // 2)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        length = stop - start
        if length < minimum:
            continue
        width = window
        data = (returns[start:stop], group_returns[start:stop])
        padded = np.pad(data, ((0, 0), (width - 1, 0)), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=1)[:, minimum - 1:]
        counts = np.minimum(np.arange(minimum, length + 1), width) - 1
        output = values[start + minimum - 1:stop]
        # 分批限制大窗口的临时内存；窗口只含当前及此前观测
        batch_size = max(1, 1_000_000 // width)
        for left in range(0, windows.shape[1], batch_size):
            right = min(left + batch_size, windows.shape[1])
            current = windows[:, left:right]
            # 先减窗口末值再中心化，避免近常量收益的相消误差
            offsets = current[:, :, -1]
            centered = current - offsets[:, :, None]
            means = np.nanmean(centered, axis=2)
            centered = centered - means[:, :, None]
            stock_var, group_var = np.nansum(centered ** 2, axis=2) / counts[left:right]
            covariance = np.nansum(centered[0] * centered[1], axis=1) / counts[left:right]
            variable = (stock_var > _EPS ** 2) & (group_var > _EPS ** 2)
            if method == 'pearson':
                denominator = np.where(variable, np.sqrt(stock_var * group_var), np.nan)
                similarity = covariance / denominator
            else:
                stock_var = np.where(stock_var > _EPS ** 2, stock_var, 0)
                group_var = np.where(group_var > _EPS ** 2, group_var, 0)
                mean_diff = (offsets[0] - offsets[1]) + (means[0] - means[1])
                denominator = stock_var + group_var + mean_diff ** 2
                covariance = np.where(variable, covariance, 0)
                similarity = 2 * covariance / np.where(denominator > _EPS ** 2, denominator, np.nan)
            output[left:right] = np.clip(similarity, -1, 1)
    values[~np.isfinite(values)] = np.nan

    return values.astype('float32')


def _group_scores(panel, f_window, w_window, similarity):
    """
    F 取满窗累计收益，W 取滚动相似度；二者转 float32 后以 method='min' 取当日分位。
    F 排板块，W 排股票与板块成员对；股票分数为有效成员对的分位乘积均值，不以未来收益筛样。
    """
    panel = panel.loc[panel['收益'].notna(), ['交易日期', '股票代码', '收益', '板块']].copy()
    labels = panel['板块'].fillna('').str.replace(r'[,;/|，；。]', '、', regex=True)
    labels = labels.str.replace(r'\s+', '', regex=True).str.split('、')
    panel['板块'] = labels.map(lambda names: list(dict.fromkeys(name for name in names if name)))
    pairs = panel.explode('板块').dropna(subset=['板块'])
    if pairs.empty:
        return pd.Series(dtype='float64', index=pd.MultiIndex.from_arrays([[], []], names=['交易日期', '股票代码']))
    pairs['股票代码'] = pairs['股票代码'].astype('category')
    pairs['板块'] = pairs['板块'].astype('category')
    groups = (
        pairs.groupby(['交易日期', '板块'], observed=True, sort=False)['收益'].mean()
        .rename('板块收益').reset_index().sort_values(['板块', '交易日期'])
    )
    if f_window == 1:
        momentum = groups['板块收益']
    else:
        groups['对数收益'] = np.log1p(groups['板块收益'].clip(lower=-0.999))
        # 窗口独立求和，避免历史累计误差拆散并列
        momentum = groups.groupby('板块', observed=True, sort=False)['对数收益'].transform(
            lambda values: np.expm1(values.rolling(f_window, min_periods=f_window).apply(np.sum, raw=True))
        )
    groups['动量'] = momentum.astype('float32')
    groups['F分位'] = groups.groupby('交易日期')['动量'].rank(pct=True, method='min')
    pairs = pairs.merge(groups[['交易日期', '板块', '板块收益', 'F分位']], on=['交易日期', '板块'])
    pairs = pairs.sort_values(['股票代码', '板块', '交易日期']).reset_index(drop=True)
    pairs['相似度'] = _rolling_similarity(pairs, w_window, similarity)
    # W 分位的分母包含 F 尚未满窗的成员对
    pairs['W分位'] = pairs.groupby('交易日期')['相似度'].rank(pct=True, method='min')
    pairs['乘积'] = pairs['F分位'] * pairs['W分位']

    return pairs.groupby(['交易日期', '股票代码'], observed=True)['乘积'].mean()


def calc_select_factor(df, strategy: StrategyConfig) -> pd.DataFrame:
    """
    计算复合选股因子，返回带 strategy.factor_name 列的 df。

    :param df: filter_before_select 之后的日频面板（多股票 × 多交易日），列为框架基础列与 strategy.factor_columns。
    :param strategy: 策略配置；读取 factor_list 与 factor_name，不改写 select_num。
    :return: pd.DataFrame，新增 strategy.factor_name 列，越小越优；保留 df 的索引与长度，不修改输入。

    factor_list 解析：前两项分别声明日收益窗口与板块模式；核心分数由全市场计算，再在候选面板内排名。
    核心项与其余辅助项的 args 为数值权重，板块项的 args 须为 0。
    """
    if len(strategy.factor_list) < 2:
        raise ValueError('factor_list 须包含日收益与板块归属两项')
    core, group, *aux_factors = strategy.factor_list
    if core.name not in {'z_日收益_zh', 'z_DailyReturn_en'}:
        raise ValueError('factor_list 第一项须为配套日收益因子')
    concept = group.name in {'z_概念归属_zh', 'z_ConceptMembership_en'} and group.param == '概念'
    industry = group.name in {'z_行业分类_zh', 'z_IndustryClassification_en'} and group.param in {
        '一级行业', '二级行业', '三级行业',
    }
    if not (concept or industry) or group.weight != 0:
        raise ValueError('factor_list 第二项须为概念归属或行业分类，param 选择板块，权重须为 0')
    param = core.param if core.param not in (None, '') else (5, 10, 'pearson')
    f_window, w_window, similarity = param
    f_window, w_window = int(f_window), int(w_window)
    if f_window < 1 or w_window < 2 or similarity not in {'pearson', 'ccc'}:
        raise ValueError(f'param 须为 (F, W, similarity)，F ≥ 1、W ≥ 2，算法为 pearson 或 ccc，收到 {param!r}')
    if df.empty:
        return df.assign(**{strategy.factor_name: pd.Series(dtype='float64', index=df.index)})

    composite = pd.Series(0.0, index=df.index)
    if core.weight:
        scores = _group_scores(_load_market(df['交易日期'].max(), group.param), f_window, w_window, similarity)
        values = df[['交易日期', '股票代码']].join(scores.rename('板块分数'), on=['交易日期', '股票代码'])
        composite = values.groupby('交易日期')['板块分数'].rank(ascending=core.is_sort_asc, method='min') * core.weight
    by_date = df.groupby('交易日期')
    for af in aux_factors:
        if af.weight:
            composite = composite + by_date[af.col_name].rank(ascending=af.is_sort_asc, method='min') * af.weight

    return df.assign(**{strategy.factor_name: composite})
