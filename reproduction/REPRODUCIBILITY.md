# Numerical checks | 数值核验

Fresh calculations were compared with the manuscript's reference outputs using observations ending on 31 August 2026.
使用截至 2026 年 8 月 31 日的观测重新计算，并与论文参照输出核对。

| Layer | Executed comparison | Result |
|---|---|---|
| Prices and calendar | 10,447,754 price rows; 2,373 dates | Keys, values and missingness matched the corrected inputs. |
| Membership | 82,392,428 normalized keys | Matched the retained research input. |
| Covariates | 8,995,748 rows × 51 fields | Keys and null patterns matched; maximum Float64 difference 1.144e-14. Five Float32 differences did not intersect the checked models' inputs. |
| Daily attention | 8,974,107 rows × 51 fields | Values, missingness and final Parquet hash matched. |
| Scores and selection | 5,647,200 rank positions and all tested TopN sets | Matched. |
| Models | 5,018 principal fits and 384 attention HGB fits | Result tables matched within recorded tolerances; diagnostic path and Float32 differences were recorded separately. |
| Inference | Declared 100,000- and 1,000,000-draw bootstrap designs | Result tables matched. |
| Accounts | 48 baseline and 48 paired simulations; 227,088 daily records | Daily paths matched; 48 paired metric rows were independently recalculated. |
| Figures | 21 redraws | Numbers and text matched; 17 PDF rasterizations were pixel-identical at 150 dpi, four had sub-point font-position differences. |

The model count excludes 201 and 341 multioutput closed-form Ridge estimation events, 202 date-wise three-outcome OLS events, and environment-validation fits.
主模型计数不包含闭式岭估计、逐日 OLS 及环境检验拟合。

## Sample and numerical scope

The covariate constructor implements a reconstructed historical preprocessing convention checked against the full retained panel. Published date/security indices preserve the analysed observations; they do not recover the original historical admission program or certify first-publication times. Four corrected closing prices remain external inputs; their event keys and correction rules are public.
协变量使用经全表核对的历史预处理重构规则。公开日期与证券索引保留研究观测，不等于恢复原始历史入样程序或认证首次公开时点。四条修正收盘价需独立取得，事件键与修正规则已经公开。

An isolated installation matching the recorded MKL builds reproduced predictions, Top20/Top100 memberships and daily returns for four complete H3/2020 models. A portable OpenBLAS installation produced small Ridge prediction differences and three ordinary-Ridge Top100 stock/date membership changes; Top20 daily returns were unchanged in that check. This is one complete fold, not a full cross-platform rerun. Package versions alone do not fix the numerical library.
匹配 MKL 构建的独立环境在一个完整折次中复现了预测、名单与收益。OpenBLAS 环境出现微小岭回归差异，并影响普通岭回归 Top100 的三个股票日成员；该检查中的 Top20 日收益未变。软件包版本本身不固定底层数值库。

[Experimental results](results/index.csv) · [Account paths](sections/section-s17-retrospective-account-simulations/README.md) · [Input preparation](RAW_REBUILD.md)
