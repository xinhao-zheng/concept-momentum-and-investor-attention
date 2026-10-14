"""Integer-ratio ordering for the declared membership-rank aggregations."""

from functools import cmp_to_key
import numpy as np
import pandas as pd
import polars as pl


def add_aggregations(x):
    x = x.with_columns(
        (
            pl.col("F_only_num") * pl.col("W_count")
            + pl.col("W_only_num") * pl.col("F_count")
        ).alias("A_num"),
        (2 * pl.col("F_count") * pl.col("W_count") * pl.col("valid_cards")).alias(
            "A_den"
        ),
        (pl.col("F_only_num") * pl.col("W_only_num")).alias("P_num"),
        (pl.col("F_count") * pl.col("W_count") * pl.col("valid_cards").pow(2)).alias(
            "P_den"
        ),
    )
    return x.with_columns(
        (pl.col("A_num") / pl.col("A_den")).alias("A"),
        (pl.col("P_num") / pl.col("P_den")).alias("P"),
    )


def rank_table(x, col, method, extra):
    n = col + "_num"
    d = col + "_den"
    if n not in x.columns or d not in x.columns:
        r = (
            x.select("date", "code", pl.col(col).alias("score"))
            .sort(["date", "score", "code"], descending=[False, True, False])
            .with_columns(pl.col("code").cum_count().over("date").alias("rank"))
            .filter(pl.col("rank") <= 100)
        )
    else:
        p = x.select("date", "code", pl.col(col).alias("score"), n, d).to_pandas()
        rows = []
        for day, g in p.groupby("date", sort=True):
            g = g.reset_index(drop=True)
            nums = [int(v) for v in g[n]]
            dens = [int(v) for v in g[d]]
            codes = g.code.tolist()
            values = g.score.to_numpy()
            order = np.argsort(-values, kind="stable").tolist()

            def compare(i, j):
                a = nums[i] * dens[j]
                b = nums[j] * dens[i]
                if a != b:
                    return -1 if a > b else 1
                return -1 if codes[i] < codes[j] else 1 if codes[i] > codes[j] else 0

            lo = 0
            for k in range(1, len(order) + 1):
                if (
                    k == len(order)
                    or abs(values[order[k - 1]] - values[order[k]]) > 1e-12
                ):
                    if k - lo > 1:
                        order[lo:k] = sorted(order[lo:k], key=cmp_to_key(compare))
                    lo = k
            if any(compare(order[i], order[i + 1]) > 0 for i in range(len(order) - 1)):
                order = sorted(order, key=cmp_to_key(compare))
            assert all(
                compare(order[i], order[i + 1]) <= 0 for i in range(len(order) - 1)
            )
            q = g.iloc[order[:100]][["date", "code", "score"]].copy()
            q["rank"] = np.arange(1, len(q) + 1, dtype=np.uint32)
            rows.append(q)
        r = pl.from_pandas(pd.concat(rows, ignore_index=True)).with_columns(
            pl.col("rank").cast(pl.UInt32), pl.col("date").cast(pl.Datetime("ns"))
        )
    return r.with_columns(
        pl.lit(method).alias("method"), *[pl.lit(v).alias(k) for k, v in extra.items()]
    )


def ranks(x, cols, extra):
    return pl.concat(
        [rank_table(x, col, method, extra) for method, col in cols.items()]
    )
