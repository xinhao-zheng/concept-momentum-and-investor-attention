# Data access | 数据获取

The repository supplies author-collected concept memberships, sample indices, experiment results and daily account aggregates. External market observations are obtained separately and mapped to the field contracts below. Observations end on 31 August 2026; rolling calculations also need the preceding initialization history.
仓库提供作者采集的概念成员、样本索引、实验结果和逐日账户汇总。外部市场观测需独立取得并映射到以下字段约定。观察截止于 2026 年 8 月 31 日，滚动计算还需此前的预热历史。

## Inputs | 输入

| Input | Required fields and conventions | Access route |
|---|---|---|
| Daily stocks | Security and date; open, high, low, close, previous close, volume, turnover, capitalization and the adjustment convention | Historical data through an exchange or an entitled market-data service; see [SSE](https://www.sseinfo.com/services/sseai/aidata/) and [SZSE](https://investor.szse.cn/English/services/dataServices/index.html) services. |
| Market indices | CSI 300 (`sh000300.csv`): `candle_end_time` and price-index `close`. SSE Composite (`sh000001.csv`): `candle_end_time` for M2 measurement dates | Index publisher or historical market-data service. Preserve each series identifier and its market-session dates. |
| Industry and index membership | Dated membership; `新版申万一级行业名称`, `新版申万二级行业名称`, `新版申万三级行业名称`; `Y` denotes positive index membership | [SWS Research](https://www.swsresearch.com/institute_sw/allIndex/downloadCenter/industryType) or a historical data service. Obtain the dated labels, not a current-only classification. |
| Attention | Wind `总热度(万)` (T), `自选热度(万)` (S), `关注粘性(万)` (K), security, observation date and concept string | [Wind Client API](https://www.wind.com.cn/portal/zh/ClientApi/index.html). Confirm entitlement to the exact fields, units and dates. |
| Supplier-labelled popularity lists | Timestamped one-hour and 24-hour SH/SZ lists: `type,code,name,value,increment,percent,current,chg,source_timestamp` | Historical supplier snapshots preserving timestamps and score definitions. [Xueqiu hot-stock lists](https://xueqiu.com/hot/stock) is a product reference. |

Product pages identify access channels; they do not establish the availability of the study's complete historical series. Record the acquired product, version, dates and adjustment convention with local inputs. Credentials and workstation paths belong in local configuration files.
产品页面用于说明获取渠道，不代表当前账户或网页提供研究所需的全部历史序列。取得数据后，在本地记录产品、版本、日期和复权口径；凭证及工作站路径写入本地配置。

The retained list records do not establish official platform identity or independence between upstream sources.
留存榜单记录不认证官方平台身份或上游来源之间的独立性。

## Input directories | 输入目录

The published runners read the following directories under `snapshot_root`. Set that root in the [local configuration](reproduction/config/README.md); the linked readers specify filenames, columns and transformations.
公开程序从 `snapshot_root` 下的以下目录读取输入。根目录在[本地配置](reproduction/config/README.md)中设置；各读取程序定义具体文件、字段和计算规则。

| Directory | Contents used | Reader |
|---|---|---|
| `raw_snapshot/stock-trading-data/` | Daily prices for return calculations | [Core inputs](reproduction/code/analysis/core_inputs.py) |
| `raw_snapshot/stock-trading-data-pro/` | Daily prices, capitalization, industry, index-membership and order-flow fields for covariates | [Covariate constructor](reproduction/code/analysis/input_build.py) |
| `raw_snapshot/stock-main-index-data/` | `sh000300.csv` for market dates and returns; `sh000001.csv` for M2 dates | [Core inputs](reproduction/code/analysis/core_inputs.py), [M2 calendar check](reproduction/code/analysis/attention/input_checks.py) |
| `raw_snapshot/stock-popular-concept-detail/` | Daily T/S/K measurements and recorded concept strings | [Attention constructor](reproduction/code/analysis/attention_inputs.py) |
| `attention/inputs/` | Historical `沪深_1h.csv` and `沪深_24h.csv` popularity lists | [Attention analyses](reproduction/code/analysis/attention/README.md) |

Corrected daily files, when applicable, reside in `input_recovery/patched_raw/stock-trading-data/` and `input_recovery/patched_raw/stock-trading-data-pro/`; they replace the corresponding raw files. The published membership input is `concept-data/concept-data.zip` in this repository. [Input preparation](reproduction/RAW_REBUILD.md) describes the complete layout and generated panels.
适用时，日行情修正文件分别放在上述两个 `input_recovery/patched_raw/` 子目录，覆盖对应原始文件。概念成员输入为仓库内的 `concept-data/concept-data.zip`。[输入准备](reproduction/RAW_REBUILD.md)说明完整目录结构与生成面板。

## Field contracts | 字段约定

- [Concept membership](concept-data/README.md): three published fields, recorded dates and empty-membership handling.
- [Daily covariates](reproduction/code/analysis/input_build_contract.json): the 26 required daily fields, industry labels and index-membership flags. The [constructor](reproduction/code/analysis/input_build_README.md) defines rolling history and observed-label forward filling.
- [Attention inputs](reproduction/code/analysis/attention_inputs_README.md): logarithms, differences, weekday transformations and sample support. The seven raw columns are security code, security name, observation date, T, S, K and concepts; preserve their order, units and missing values.
- [List analyses](reproduction/code/analysis/attention/README.md): historical `沪深_1h.csv` and `沪深_24h.csv` files. The parser selects a complete 100-name snapshot closest to 15:30 within 90 minutes. A current-page scrape cannot replace these historical snapshots.

The CSI 300 calendar uses sorted dates from `stock-main-index-data/sh000300.csv`, retaining the first duplicate date. Market return is `close.pct_change(fill_method=None)`. The [CSI 300 methodology](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300_Index_Methodology_cn.pdf) identifies the price index. Preserve market trading dates rather than substituting calendar days or each stock's own observation dates.
沪深 300 文件提供市场日历和市场收益；保留市场交易日，不以自然日或单只股票的观测日期替代。

M2 also reads `stock-main-index-data/sh000001.csv`. Supply a UTF-8 CSV with a first-row header and unique, ascending `candle_end_time` dates. M2 retains market sessions from 2016-01-01 through 2026-08-31 and requires the last retained date to equal the cutoff; it does not read this file's price fields.
M2 另读上证综指文件 `sh000001.csv`。文件为 UTF-8 CSV，首行为表头，`candle_end_time` 日期唯一且升序。程序保留 2016 年 1 月 1 日至 2026 年 8 月 31 日的市场交易日，并要求最后日期等于截止日；此分支不读取该文件的价格字段。

## Corrections and sample support | 修正与样本范围

The [daily correction ledger](reproduction/data/daily-correction-events.csv) specifies 55 removed observations, four closing-price corrections and one security-code continuity event. [Correction instructions](reproduction/docs/daily-input-corrections.md) rebuild the daily overlays from the acquired files. The four correct closing prices for 9 June 2025 are supplied locally; their security/date keys and validation rules are public.
公开修正表列明 55 个删除观测、4 条收盘价修正和 1 项证券代码历史衔接。按说明从取得的日行情重建覆盖层；四条正确收盘价在本地提供，具体证券、日期和核验规则已经公开。

The published date/security indices select the analysed observations after raw features are constructed. They contain no covariate or popularity values. Their relation to the reconstructed preprocessing rules is stated in the [reproduction record](reproduction/REPRODUCIBILITY.md).
公开日期与证券索引在原始特征计算后选取研究观测，不含协变量或人气数值；与预处理重构规则的关系见[复现记录](reproduction/REPRODUCIBILITY.md)。
