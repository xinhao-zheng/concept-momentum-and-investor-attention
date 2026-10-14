# Data access | 数据获取

The repository supplies author-collected concept memberships, sample indices, experiment results and daily account aggregates. External market observations are obtained separately and mapped to the field contracts below. Observations end on 31 August 2026; rolling calculations also need the preceding initialization history.
仓库提供作者采集的概念成员、样本索引、实验结果和逐日账户汇总。外部市场观测需独立取得并映射到以下字段约定。观察截止于 2026 年 8 月 31 日，滚动计算还需此前的预热历史。

## Inputs | 输入

| Input | Required fields and conventions | Access route |
|---|---|---|
| Daily stocks | Security and date; open, high, low, close, previous close, volume, turnover, capitalization and the adjustment convention | Historical data through an exchange or an entitled market-data service; see [SSE](https://www.sseinfo.com/services/sseai/aidata/) and [SZSE](https://investor.szse.cn/English/services/dataServices/index.html) services. |
| Market indices | `candle_end_time` and price-index `close`; CSI 300 is the calendar and market-return reference | Index publisher or historical market-data service. Keep the series identifiers used by the selected experiment. |
| Industry and index membership | Dated membership; `新版申万一级行业名称`, `新版申万二级行业名称`, `新版申万三级行业名称`; `Y` denotes positive index membership | [SWS Research](https://www.swsresearch.com/institute_sw/allIndex/downloadCenter/industryType) or a historical data service. Obtain the dated labels, not a current-only classification. |
| Attention | Wind `总热度(万)` (T), `自选热度(万)` (S), `关注粘性(万)` (K), security, observation date and concept string | [Wind Client API](https://www.wind.com.cn/portal/zh/ClientApi/index.html). Confirm entitlement to the exact fields, units and dates. |
| Cross-platform lists | Timestamped one-hour and 24-hour SH/SZ lists: `type,code,name,value,increment,percent,current,chg,source_timestamp` | Historical snapshots of the [Xueqiu hot-stock lists](https://xueqiu.com/hot/stock), obtained through a route that preserves the original timestamps and score meaning. |

Product pages identify access channels; they do not establish the availability of the study's complete historical series. Record the acquired product, version, dates and adjustment convention with local inputs. Credentials and workstation paths belong in local configuration files.
产品页面用于说明获取渠道，不代表当前账户或网页提供研究所需的全部历史序列。取得数据后，在本地记录产品、版本、日期和复权口径；凭证及工作站路径写入本地配置。

## Field contracts | 字段约定

- [Concept membership](concept-data/README.md): three published fields, recorded dates and empty-membership handling.
- [Daily covariates](reproduction/code/analysis/input_build_contract.json): the 26 required daily fields, industry labels and index-membership flags. The [constructor](reproduction/code/analysis/input_build_README.md) defines rolling history and observed-label forward filling.
- [Attention inputs](reproduction/code/analysis/attention_inputs_README.md): logarithms, differences, weekday transformations and sample support. The seven raw columns are security code, security name, observation date, T, S, K and concepts; preserve their order, units and missing values.
- [Cross-platform analyses](reproduction/code/analysis/attention/README.md): historical `沪深_1h.csv` and `沪深_24h.csv` files. The parser selects a complete 100-name snapshot closest to 15:30 within 90 minutes. A current-page scrape cannot replace these historical snapshots.

The CSI 300 calendar uses sorted dates from `stock-main-index-data/sh000300.csv`, retaining the first duplicate date. Market return is `close.pct_change(fill_method=None)`. The [CSI 300 methodology](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300_Index_Methodology_cn.pdf) identifies the price index. Preserve market trading dates rather than substituting calendar days or each stock's own observation dates.
沪深 300 文件提供市场日历和市场收益；保留市场交易日，不以自然日或单只股票的观测日期替代。

## Corrections and sample support | 修正与样本范围

The [daily correction ledger](reproduction/data/daily-correction-events.csv) specifies 55 removed observations, four closing-price corrections and one security-code continuity event. [Correction instructions](reproduction/docs/daily-input-corrections.md) rebuild the daily overlays from the acquired files. The four correct closing prices for 9 June 2025 are supplied locally; their security/date keys and validation rules are public.
公开修正表列明 55 个删除观测、4 条收盘价修正和 1 项证券代码历史衔接。按说明从取得的日行情重建覆盖层；四条正确收盘价在本地提供，具体证券、日期和核验规则已经公开。

The published date/security indices select the analysed observations after raw features are constructed. They contain no covariate or popularity values. Their relation to the reconstructed preprocessing rules is stated in the [reproduction record](reproduction/REPRODUCIBILITY.md).
公开日期与证券索引在原始特征计算后选取研究观测，不含协变量或人气数值；与预处理重构规则的关系见[复现记录](reproduction/REPRODUCIBILITY.md)。
