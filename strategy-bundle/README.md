# Strategy modules | 策略模块

Three factors, one strategy and a membership reader. Factors and the strategy have English and Chinese editions with the same logic; filenames identify the modules.
三个因子、一个策略和一个成员读取器。因子与策略各有逻辑相同的中英文版本，文件名即模块标识。

| Directory | Interface | Installation directory |
|---|---|---|
| `factors/` | `add_factor(df, param, **kwargs)` | `因子库/` |
| `strategies/` | `calc_select_factor(df, strategy)` | `策略库/` |
| `external-data/` | Concept-membership reader | `外部数据/` |

The CSV reader locates its header by required fields and accepts UTF-8 or GB18030. The published archive uses a first-row UTF-8 header. Historical alignment uses the previous market day and preserves empty or missing memberships.
CSV 读取器按所需字段定位表头，兼容 UTF-8 与 GB18030。公开档案首行为 UTF-8 表头；历史对齐使用前一市场交易日，保留空成员与缺失。

[Concept data](../concept-data/README.md) · [Research calculations](../reproduction/README.md)
