# concept-momentum-and-investor-attention

Concept-membership records for China's A-shares, the modules that turn those records into a selection score, and the code that rebuilds the tables and figures of *Concept momentum and investor attention*. Records stop on 2026-08-31.
中国 A 股的概念成员记录、把这些记录变成选股评分的模块，以及重建《概念动量与投资者注意力》表和图的代码。记录止于 2026-08-31。

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
└── reproduction/          # tables and figures rebuilt from the records and the score · 由记录与评分重建的表和图
```

`concept-data/` supplies the records. `strategy-bundle/` reads them and scores a stock. `reproduction/` rebuilds the manuscript's tables and figures from both.
`concept-data/` 提供记录。`strategy-bundle/` 读取记录并为股票打分。`reproduction/` 据此重建稿件的表和图。

Factors and strategies each ship in two editions of the same logic — English `z_EnglishName_en.py` and Chinese `z_中文名_zh.py`. The host loads a module by the directory it sits in and by its filename: `factors/` into `因子库/`, `strategies/` into `策略库/`, `external-data/` into `外部数据/`. Membership files sit in `stock-popular-concept-detail/` under the host data directory. The host supplies `config` and `core.model.strategy_config`.
因子与策略各有逻辑相同的两版——英文版 `z_EnglishName_en.py` 与中文版 `z_中文名_zh.py`。宿主按目录和文件名加载：`factors/` 放入 `因子库/`，`strategies/` 放入 `策略库/`，`external-data/` 放入 `外部数据/`。成员文件放在宿主数据目录下的 `stock-popular-concept-detail/`。宿主提供 `config` 与 `core.model.strategy_config`。

---

## Encoding | 编码

UTF-8, no BOM, LF line endings. · UTF-8，无 BOM，LF 换行。

---

## License | 许可

MIT (see `LICENSE`). · MIT（见 `LICENSE`）。
