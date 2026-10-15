# Reproduction | 复现

Browse by displayed article number: 8 main figures, 13 supplementary figures, 16 table objects and 17 supplementary sections. Figure, table and section numbers are separate sequences. The [result index](results/index.csv) covers the experimental output tables; shared code is mapped to each object through the three `index.json` files.
按稿件编号查找：8 张主图、13 张补充图、16 个表对象和 17 个补充章节。图、表、章节各自编号。[结果索引](results/index.csv)汇集实验输出表；三份 `index.json` 将共用代码对应到各对象。

```text
python reproduction/run.py --list
python reproduction/run.py --artifact fig-08 --mode render
python reproduction/run.py --artifact table-s04 --mode render
python reproduction/run.py --artifact section-s04 --mode analyse --config local.json --plan
python reproduction/run.py --artifact all --mode verify
python reproduction/verify_archive.py
```

| Mode | Operation |
|---|---|
| `prepare` | Build research inputs from raw records and published sample indices |
| `analyse` | Run research calculations, or recompute account metrics from released daily paths |
| `render` | Draw figures or format tables from released results |
| `verify` | Check stored files and object mappings |

Use comma-separated identifiers for several objects. `--plan` prints commands before preparation or analysis. Figure 1 and the design tables describe the method and have no fitted experiment. Supplementary sections point to calculations and manuscript text.
多个对象以逗号分隔；`--plan` 在准备输入或计算前列出命令。图 1 与设计表展示方法，没有另行拟合的实验。补充章节连接对应计算与论文文本。

## Environments | 环境

Scientific estimation and figure rendering use separate environments:
科学计算与绘图分别安装：

```text
conda env create -f reproduction/config/environment-mkl.yml
python -m venv .venv/plotting
.venv/plotting/Scripts/python.exe -m pip install -r reproduction/code/plotting/requirements-lock.txt

conda run -n concept-momentum-science python reproduction/run.py --artifact all --mode analyse --config local.json --plan
.venv/plotting/Scripts/python.exe reproduction/run.py --artifact fig-08 --mode render
```

On Linux or macOS, use `bin/python` in place of `Scripts/python.exe`. A portable scientific environment can install `requirements-lock.txt`; the conda file also pins the recorded Windows MKL builds. Plotting requires Arial, which is not bundled. Numerical and font differences are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
Linux 或 macOS 使用 `bin/python`。跨平台科学环境可安装 `requirements-lock.txt`；conda 文件另外固定记录中的 Windows MKL 构建。绘图需要自行安装 Arial。数值与字体差异见[复现记录](REPRODUCIBILITY.md)。

## Inputs and outputs | 输入与输出

From the repository root, copy `reproduction/config/local.example.json` to `local.json` and set the data, snapshot and output directories. Pass `--config local.json` or set `CONCEPT_QUANT_CONFIG`. [Configuration](config/README.md) defines path resolution; [RAW_REBUILD.md](RAW_REBUILD.md) gives the source layout and execution order; [DATA_ACCESS.md](../DATA_ACCESS.md) gives acquisition routes and required fields.
在仓库根目录将示例配置复制为 `local.json`，设置数据、快照及输出目录，通过参数或环境变量传入。[配置说明](config/README.md)定义相对路径的解析基准；原始文件布局与执行顺序见[输入准备](RAW_REBUILD.md)，获取途径及字段见[数据说明](../DATA_ACCESS.md)。

Select `all` to run shared dependencies in order. A single-object call assumes its stated prerequisites have been prepared. Calculations write to `output_root`, which defaults to `reproduction/work/`. Rendering reads bundled results using paths relative to the repository and writes each figure to its `analysis/rebuilt/` directory. Released manuscript files and figures remain the reference outputs.
选择 `all` 按依赖顺序计算；单对象调用需先备好其上游输入。计算写入 `output_root`，默认位置为 `reproduction/work/`。绘图按仓库相对路径读取随包结果，写入对应图目录的 `analysis/rebuilt/`。公开论文与图件保留为参照输出。

[`figures/`](figures/README.md) · [`tables/`](tables/) · [`sections/`](sections/) · [Recorded checks](REPRODUCIBILITY.md)
