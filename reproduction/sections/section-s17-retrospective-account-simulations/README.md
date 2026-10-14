# S17: Retrospective account simulations

Forty-eight configurations cross holding period, F/W windows, correlation measure, portfolio size and timing rule. Each has a baseline path and a paired path with the next-session limit-up gate removed. Observations end on 31 August 2026.
48 组设定交叉持有期、F/W 窗口、相关性度量、股票数量与择时规则；每组提供基准路径与移除下一交易日涨停条件后的配对路径。观察截止于 2026 年 8 月 31 日。

| Data | Content |
|---|---|
| [Baseline paths](data/accounts/baseline/) | 48 daily accounts |
| [Paired paths](data/accounts/single-gate/) | 48 daily accounts |
| [Path index](data/account_paths.json) | Configuration identifiers, hashes and removed-gate counts |
| [Paired metrics](data/paired_single_gate_48.csv) | 48 reference comparison rows |

Fields are `date`, `nav`, `total_assets`, `cash`, `position_value`, `exposure`, `commission`, `stamp_tax` and `total_fees`. Values are account-level aggregates; NAV starts at one, exposure is a ratio and monetary fields retain the account currency unit.
字段为账户汇总值；净值从 1 开始，仓位为比例，金额保留账户币种单位。

```text
python reproduction/code/tables/account_metrics.py --config local.json
```

The calculation checks file hashes and paired dates, then recomputes return, drawdown, exposure, fees and turnover metrics. Outputs go to `output_root/accounts/paired-diagnostic`.
程序核对文件哈希与配对日期，再计算收益、回撤、仓位、费用和换手；结果写入配置指定的输出目录。

[Figure 8](../../figures/fig-08-account-simulations/README.md) · [Supplementary Figure S13](../../figures/fig-s13-complete-drawdown-histories/README.md) · [Supplementary information](../../../paper/supplementary-information.pdf)
