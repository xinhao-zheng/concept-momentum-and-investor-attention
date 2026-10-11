# concept-momentum-and-investor-attention

Concept-membership records for China's A-shares, the modules that turn those records into a selection score, and reproduction data and analysis tools for *Concept momentum and investor attention: Evidence from China's A-share market*. Records stop on 2026-08-31.
中国 A 股的概念成员记录、把这些记录变成选股评分的模块，以及《概念动量与投资者注意力：中国 A 股市场的证据》的复现数据与分析工具。记录止于 2026-08-31。

---

## Layout | 目录

```text
.
├── README.md
├── LICENSE
├── concept-data/          # concept-data.zip, one CSV per stock · 每只股票一个 CSV
├── strategy-bundle/
│   ├── factors/           # one measurement per module · 一个模块一项度量
│   ├── strategies/        # measurements composed into a score · 度量合成评分
│   └── external-data/     # membership reader · 成员读取
└── reproduction/          # reproduction data and analysis tools · 复现数据与分析工具
```

`concept-data/` supplies membership records. `strategy-bundle/` provides the account-simulation implementation variant, whose scoring rules differ from the paper's research ranking. `reproduction/` contains reproduction data and analysis tools.
`concept-data/` 提供概念成员记录。`strategy-bundle/` 提供账户模拟所用的实现变体，其评分规则与论文研究排序存在差异。`reproduction/` 收录复现数据与分析工具。

Factors and strategies each ship in two editions of the same logic — English `z_EnglishName_en.py` and Chinese `z_中文名_zh.py`. The host loads a module by the directory it sits in and by its filename: `factors/` into `因子库/`, `strategies/` into `策略库/`, `external-data/` into `外部数据/`. Membership files sit in `stock-popular-concept-detail/` under the host data directory. The host supplies `config` and `core.model.strategy_config`. Separate price inputs are required; Wind popularity measures and licensed price and industry inputs are not included.
因子与策略各有逻辑相同的两版——英文版 `z_EnglishName_en.py` 与中文版 `z_中文名_zh.py`。宿主按目录和文件名加载：`factors/` 放入 `因子库/`，`strategies/` 放入 `策略库/`，`external-data/` 放入 `外部数据/`。成员文件放在宿主数据目录下的 `stock-popular-concept-detail/`。宿主提供 `config` 与 `core.model.strategy_config`。价格数据须另备；本库不包含 Wind 人气指标及受许可约束的价格与行业数据。

---

## Encoding | 编码

UTF-8, no BOM, LF line endings. · UTF-8，无 BOM，LF 换行。

---

## License | 许可

MIT (see `LICENSE`). · MIT（见 `LICENSE`）。
