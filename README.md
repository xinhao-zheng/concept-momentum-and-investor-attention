# concept-momentum-and-investor-attention

Research data, selection modules and analysis code for *Concept momentum and investor attention: Evidence from China's A-share market*. Observations end on 31 August 2026.
《概念动量与投资者注意力：中国 A 股市场的证据》的研究数据、选股模块与分析代码。观察截止于 2026 年 8 月 31 日。

---

## Layout | 目录

```text
.
├── paper/                 # article, appendix and complete manuscript · 正文、附录与全文
├── concept-data/          # recorded concept memberships · 概念成员记录
├── strategy-bundle/       # factors, strategy and data reader · 因子、策略与数据读取
└── reproduction/
    ├── results/           # experimental result index · 实验结果索引
    ├── figures/           # Fig 1–8 and S1–S13 · 主图与补充图
    ├── tables/            # design tables and S1–S14 · 设计表与补充表
    ├── sections/          # S1–S17 · 补充章节
    ├── code/              # shared analysis and plotting · 共用分析与绘图
    ├── config/            # input paths and environments · 输入路径与运行环境
    └── run.py             # prepare, analyse, render, verify · 准备、计算、绘图、核验
```

Read the [complete manuscript](paper/complete-manuscript.pdf), browse the [experimental results](reproduction/results/index.csv), or select a figure, table or section through the [reproduction guide](reproduction/README.md). Shared calculations are stored once and mapped to the objects that use them.
阅读[完整手稿](paper/complete-manuscript.pdf)、查看[实验结果](reproduction/results/index.csv)，或从[复现说明](reproduction/README.md)按图、表、章节进入。共用计算保留一份，再对应到各个图表。

---

## Use | 使用

Install Git LFS before cloning, then retrieve the data files:
克隆前安装 Git LFS，再获取数据文件：

```text
git lfs pull
python reproduction/run.py --list
python reproduction/verify_archive.py
```

The release contains experiment results, plotting and research code, nine strategy modules, and 96 daily account paths with metric calculations. The trading simulator and timing implementations are not included.
公开内容包含实验结果、绘图与研究代码、九个策略相关模块，以及 96 条逐日账户路径和指标计算程序。交易模拟器与择时实现不随包提供。

The author-collected membership archive is included. Prices, attention measurements and other external observations are obtained separately; see [data access](DATA_ACCESS.md). Configure local paths in a separate file. [Input preparation](reproduction/RAW_REBUILD.md) and [numerical checks](reproduction/REPRODUCIBILITY.md) define the calculation scope.
包内提供作者采集整理的概念成员档案。行情、人气测量及其他外部观测需独立取得，见[数据说明](DATA_ACCESS.md)。本地路径在独立配置文件中设置；[输入准备](reproduction/RAW_REBUILD.md)与[数值核验](reproduction/REPRODUCIBILITY.md)说明计算范围。

---

## Encoding | 编码

UTF-8, no BOM, LF line endings. · UTF-8，无 BOM，LF 换行。

---

## License | 许可

MIT (see [LICENSE](LICENSE)). External inputs are outside the published data package.
MIT（见 [LICENSE](LICENSE)）。外部输入不属于本项目公开数据包。
