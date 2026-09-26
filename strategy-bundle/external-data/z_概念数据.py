"""
Stock Quant Strategy Framework
---------------------------------------------------
Author:    Xinhao Zheng
Contact:   Veritas428 (WeChat)
Copyright: (c) 2026 Xinhao Zheng. Licensed under the MIT License.
---------------------------------------------------
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config

# 框架读取目录，同时供客户端识别概念产品依赖
ext_data_path = Path(config.data_center_path) / 'stock-popular-concept-detail'


def is_client_selection():
    """识别客户端选股入口。"""
    return (
        bool(os.getenv('FUEL_CLIENT_CONFIG_PATH'))
        and bool(os.getenv('FUEL_PRO_TRADING_PATH'))
        and tuple(sys.argv[-2:]) == ('select', 'trading')
    )


@lru_cache(maxsize=1)
def read_decision_context():
    """返回客户端全市场末行情日与运行时点；进程内复用，避免停牌股票采用过期决策日。"""
    asof = pd.Timestamp.now()
    if not is_client_selection():
        return None, asof
    path = Path(config.data_center_path) / 'stock-main-index-data' / 'sh000001.csv'
    index = pd.read_csv(path, usecols=['candle_end_time'])
    dates = pd.to_datetime(index['candle_end_time'])
    cutoff = asof.normalize() - pd.Timedelta(days=int(asof.hour < 15))

    return dates.loc[dates.le(cutoff)].max(), asof


@lru_cache(maxsize=1)
def read_calendar():
    """读取市场交易日历；停牌不改变交易日间隔。"""
    path = Path(config.data_center_path) / 'period_offset.csv'
    dates = pd.read_csv(path, encoding='gb18030', skiprows=1, usecols=['交易日期'])
    calendar = pd.DatetimeIndex(pd.to_datetime(dates['交易日期']))
    if calendar.hasnans or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError('交易日历须为无缺失、无重复的升序日期')

    return calendar


def read_membership(path):
    """读取按交易日期升序的成员序列；空标签保留空字符串，文件缺失时返回空序列。"""
    path = Path(path)
    if not path.is_file():
        return pd.Series(dtype='string', index=pd.DatetimeIndex([], name='交易日期'), name='板块')
    data = pd.read_csv(
        path, encoding='gb18030', skiprows=1, usecols=['交易日期', '所属概念'], keep_default_na=False,
    )
    data['交易日期'] = pd.to_datetime(data['交易日期'])
    if data['交易日期'].isna().any():
        raise ValueError(f'概念日期缺失：{path}')

    return data.set_index('交易日期', verify_integrity=True)['所属概念'].sort_index().astype('string').rename('板块')


def align_membership(dates, membership, calendar, latest_date=None, asof=None):
    """
    按交易日期对齐概念成员，保留 dates 的索引与长度。

    历史取前一市场交易日成员；latest_date 指定客户端全市场末行情日 T，仅更新 T 的当前截面。
    当前截面可用范围为前一市场交易日至下一市场交易日之前，且不晚于 asof 所在日期。
    同日或周末补齐取最新记录；缺失不续接，最新空标签不回退，后续交易日成员不回填 T。
    """
    previous = pd.Series(calendar, index=calendar).shift(1).reindex(pd.DatetimeIndex(dates))
    labels = pd.Series(membership.reindex(previous.array).array, index=dates.index, dtype='string')
    if latest_date is not None:
        cutoff = pd.Timestamp(asof).normalize()
        last_previous = pd.Series(calendar, index=calendar).shift(1).get(latest_date)
        next_date = pd.Series(calendar, index=calendar).shift(-1).get(latest_date)
        available = membership.loc[
            (membership.index >= last_previous) & (membership.index < next_date) & (membership.index <= cutoff)
        ]
        if not available.empty:
            labels = labels.mask(dates.eq(latest_date), available.iloc[-1])

    return labels


def read_ext_data(file_path, candle_df: pd.DataFrame, save_cols: list) -> pd.DataFrame:
    """
    按决策时点对齐概念成员，返回含框架订阅列的日频数据。

    :param file_path: 概念数据目录，由框架传入 ext_data_path。
    :param candle_df: 单只股票的日频 K 线，含交易日期与股票代码。
    :param save_cols: 因子 extra_data 声明的订阅列，各列写入同一成员序列。
    :return: pd.DataFrame，新增订阅列，保留 candle_df 的索引与长度。本函数不修改 candle_df。
    """
    if candle_df.empty:
        return candle_df.assign(**{name: pd.Series(dtype='string') for name in save_cols})
    dates = candle_df['交易日期']
    membership = read_membership(Path(file_path) / f"{candle_df['股票代码'].iloc[0]}.csv")
    latest_date, asof = read_decision_context()
    labels = align_membership(dates, membership, read_calendar(), latest_date, asof)

    return candle_df.assign(**{name: labels for name in save_cols})
