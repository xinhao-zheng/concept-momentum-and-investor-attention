# Concept membership | 概念成员

Recorded concept memberships collected and organized by Xinhao Zheng. The ZIP contains one CSV per security, with three columns and no popularity measurements.
Xinhao Zheng 采集整理的概念成员记录。ZIP 内每只证券对应一个 CSV，保留三列，不含人气测量。

| Field | Format |
|---|---|
| `股票代码` | String: `sh`, `sz` or `bj` followed by six digits; retain leading zeros |
| `交易日期` | `YYYY-MM-DD`, the recorded snapshot date |
| `所属概念` | Original membership string; retain label order, delimiters and empty cells |

The key is security/date. Dates are nondecreasing within each file and may include non-trading days; a recorded date is not a certified first-publication timestamp. Empty memberships must not be filled from later records.
键为证券代码与日期。单文件日期非递减，可以包含非交易日；记录日期不等于经认证的首次公开时点。空成员不得用之后的记录回填。

| Coverage | Value |
|---|---:|
| CSV files | 5,720 |
| Rows | 15,743,951 |
| Dates | 2016-11-24–2026-08-31 |
| Nonempty membership rows | 13,639,505 |
| Empty membership rows | 2,104,446 |

Archive coverage exceeds the final research sample. File and row counts here therefore differ from the article's stock and stock-date counts. Files use UTF-8 without BOM, LF endings and a first-row header.
档案覆盖大于最终研究样本，此处文件数和行数不替代文中的股票数与股票日数。编码为 UTF-8 无 BOM、LF 换行，首行为表头。

```python
import pandas as pd

frame = pd.read_csv(
    csv_path,
    encoding="utf-8",
    header=0,
    dtype=str,
    keep_default_na=False,
    usecols=["股票代码", "交易日期", "所属概念"],
)
```

Retrieve the archive with `git lfs pull`. [Repository checksums](../reproduction/checksums.sha256) cover its contents as a file; [data access](../DATA_ACCESS.md) describes the external research inputs.
运行 `git lfs pull` 获取压缩包；[仓库校验表](../reproduction/checksums.sha256)记录文件哈希，[数据说明](../DATA_ACCESS.md)列出外部研究输入。
