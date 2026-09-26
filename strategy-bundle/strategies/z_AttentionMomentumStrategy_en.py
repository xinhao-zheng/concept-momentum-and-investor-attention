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

# Return standard-deviation floor to preserve numerical ties
_EPS = 1e-10

STG_INTRO = {
    'Strategy Description': """
    Attention Momentum Strategy, with a core score formed from group momentum and stock/group return similarity.

    factor_list rules:
        - the first factor is the companion daily return factor, param=(F, W, similarity), args is the core rank weight;
        - the second is the companion concept or industry factor, args=0;
          param selects 概念 / 一级行业 / 二级行业 / 三级行业;
        - every other factor is an auxiliary ranking factor whose args is its weight; 0 disables the term.
    F is the compounded group return window; daily group returns are equal-weight means of members across the market.
    W is the stock/group return similarity window. F and W are integers with F ≥ 1 and W ≥ 2;
    the default is (5, 10, 'pearson').
    similarity accepts 'pearson' (correlation) or 'ccc' (concordance correlation).

    Core score = mean(group F percentile × stock/group W percentile) over memberships.
    F ranks groups; W ranks stock/group pairs.
    Composite = Core Rank × Core Weight + Σ(Auxiliary Rank × Weight), smaller is better;
    each rank follows its own ascending flag.

    Recommended filter_list:
        ('成交额Mean', 5, 'val:>=2000_0000', True)    5-day mean turnover ≥ CNY 20 million; optional, off in examples

    Known limits: group returns include the stock itself and do not depend on candidate filters. Historical concepts use
    previous-market-day membership; the current client cross-section uses the latest membership within the allowed time
    range, without carrying forward missing, empty or stale labels. Industries use quote-date labels. Sources lack
    publication histories and versions; strict point-in-time availability cannot be established. Historical decisions
    based on weekend label updates cannot be reconstructed. Ranks use method='min', so selections may exceed select_num,
    as in the host's select_by_factor.
    Intraday timing is configured through timing and must execute before the buy.

    Use Case-1: concept Pearson, 3-day rebalancing; Use Case-2: level-2 industry CCC with turnover ranks.
    """,
    'Use Case-1':
        {
            'name': 'z_AttentionMomentumStrategy_en',
            'hold_period': '3D',
            'offset_list': [0, 1, 2],
            'select_num': 5,
            'cap_weight': 1,
            'rebalance_time': 'open',
            'factor_list': [('z_DailyReturn_en', False, (5, 10, 'pearson'), 1),
                            ('z_ConceptMembership_en', False, '概念', 0),
                            ],
            'filter_list': [],
        },
    'Use Case-2':
        {
            'name': 'z_AttentionMomentumStrategy_en',
            'hold_period': '3D',
            'offset_list': [0, 1, 2],
            'select_num': 5,
            'cap_weight': 1,
            'rebalance_time': 'open',
            'factor_list': [('z_DailyReturn_en', False, (5, 20, 'ccc'), 1),
                            ('z_IndustryClassification_en', False, '二级行业', 0),
                            ('成交额Mean', False, 20, 0.5),
                            ],
            'filter_list': [],
        },
}


def _load_market(end, mode):
    """Read full market history through end; candidate filters and listing age do not redefine group returns."""
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
        raise FileNotFoundError(f'Missing pro daily quotes: {folder}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        panel = pd.concat(pool.map(read_stock, paths), ignore_index=True)

    return panel


def _rolling_similarity(pairs, window, method):
    """
    Roll over valid stock/group observations; re-entry retains history and missing returns do not count.
    W ≤ 5 requires a full window; otherwise require at least max(5, W // 2) observations.
    CCC = 2 × cov(x, y) / (var(x) + var(y) + (mean(x) − mean(y))²), with ddof=1 for variances and covariance.
    Standard deviation ≤ _EPS is constant: Pearson is undefined, CCC covariance is zero;
    a denominator ≤ _EPS² is undefined.
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
        # Bound temporary memory for large windows; each window uses only current and prior observations
        batch_size = max(1, 1_000_000 // width)
        for left in range(0, windows.shape[1], batch_size):
            right = min(left + batch_size, windows.shape[1])
            current = windows[:, left:right]
            # Remove the window-end offset before centering to avoid cancellation in near-constant returns
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
    F uses full-window compounded returns and W rolling similarity; cast to float32 before daily method='min' ranks.
    F ranks groups and W ranks stock/group pairs. Average valid percentile products per stock; no future-return filter.
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
        # Sum each window independently so earlier rounding errors cannot split ties
        momentum = groups.groupby('板块', observed=True, sort=False)['对数收益'].transform(
            lambda values: np.expm1(values.rolling(f_window, min_periods=f_window).apply(np.sum, raw=True))
        )
    groups['动量'] = momentum.astype('float32')
    groups['F分位'] = groups.groupby('交易日期')['动量'].rank(pct=True, method='min')
    pairs = pairs.merge(groups[['交易日期', '板块', '板块收益', 'F分位']], on=['交易日期', '板块'])
    pairs = pairs.sort_values(['股票代码', '板块', '交易日期']).reset_index(drop=True)
    pairs['相似度'] = _rolling_similarity(pairs, w_window, similarity)
    # The W percentile denominator includes pairs whose F window is not ready
    pairs['W分位'] = pairs.groupby('交易日期')['相似度'].rank(pct=True, method='min')
    pairs['乘积'] = pairs['F分位'] * pairs['W分位']

    return pairs.groupby(['交易日期', '股票代码'], observed=True)['乘积'].mean()


def calc_select_factor(df, strategy: StrategyConfig) -> pd.DataFrame:
    """
    Compute the composite selection factor and return df with strategy.factor_name.

    :param df: Daily panel after filter_before_select, containing host columns and strategy.factor_columns.
    :param strategy: Reads factor_list and factor_name; does not change select_num.
    :return: pd.DataFrame with strategy.factor_name, lower is better; preserves index and length without mutating df.

    factor_list rules: the first two factors declare the return windows and group mode; core scores are computed from
    the full market and then ranked within the candidate panel. Core and auxiliary args are numeric weights;
    the group factor's args must be 0.
    """
    if len(strategy.factor_list) < 2:
        raise ValueError('factor_list must contain daily returns and group membership')
    core, group, *aux_factors = strategy.factor_list
    if core.name not in {'z_日收益_zh', 'z_DailyReturn_en'}:
        raise ValueError('The first factor_list item must be the companion daily return factor')
    concept = group.name in {'z_概念归属_zh', 'z_ConceptMembership_en'} and group.param == '概念'
    industry = group.name in {'z_行业分类_zh', 'z_IndustryClassification_en'} and group.param in {
        '一级行业', '二级行业', '三级行业',
    }
    if not (concept or industry) or group.weight != 0:
        raise ValueError('The second factor_list item must declare a valid concept or industry param, with weight 0')
    param = core.param if core.param not in (None, '') else (5, 10, 'pearson')
    f_window, w_window, similarity = param
    f_window, w_window = int(f_window), int(w_window)
    if f_window < 1 or w_window < 2 or similarity not in {'pearson', 'ccc'}:
        raise ValueError(f'param must be (F, W, similarity), F ≥ 1, W ≥ 2, using pearson or ccc, got {param!r}')
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
