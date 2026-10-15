from pathlib import Path
import os
import sys
import json
import hashlib
import datetime

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
]:
    os.environ[k] = "2"
sys.path.insert(0, str(HERE / "environment_vendor"))
sys.dont_write_bytecode = True
import polars as pl
import numpy as np
import pandas as pd

f = pd.read_parquet(RUN / "preoutcome_frame.parquet")
base = [
    "ret_1",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "log_float_mv_t",
    "log_total_mv_t",
    "log_amount_t",
    "log_amount_20_mean",
    "amount_to_float_mv_20_mean",
    "amihud_20",
    "beta_hs300_60",
    "idio_vol_hs300_60",
    "drawdown_from_60d_high",
    "market_return_t",
    "suspension_days_20",
    "zero_amount_days_20",
    "institution_net_inflow_to_amount_20",
    "large_net_inflow_to_amount_20",
    "retail_net_inflow_to_amount_20",
    "member_hs300_t",
    "member_zz500_t",
    "member_zz1000_t",
]
keys = pl.from_pandas(f[["row_id", "stock_code", "pre_signal_date"]]).rename(
    {"pre_signal_date": "signal_date"}
)
source = (
    pl.scan_parquet(RUN / "inputs/daily_covariates.parquet")
    .select(["stock_code", "signal_date", "sw1_industry", *base])
    .join(
        keys.select(["stock_code", "signal_date"]).unique().lazy(),
        on=["stock_code", "signal_date"],
        how="semi",
    )
    .collect()
)
assert source.select(["stock_code", "signal_date"]).is_duplicated().sum() == 0
actual = (
    keys.join(source, on=["stock_code", "signal_date"], how="left", validate="m:1")
    .sort("row_id")
    .to_pandas()
)
assert len(actual) == len(f)
records = []
for c in base:
    x = pd.to_numeric(actual[c]).to_numpy(float)
    y = pd.to_numeric(f["pre_" + c]).to_numpy(float)
    assert np.array_equal(x, y, equal_nan=True), c
    records.append(
        {
            "field": c,
            "rows_checked": len(x),
            "finite_values": int(np.isfinite(x).sum()),
            "missing_values": int(np.isnan(x).sum()),
            "all_values_exact": True,
        }
    )
x = actual.sw1_industry.fillna("<MISSING>").astype(str)
y = f.pre_sw1_industry.fillna("<MISSING>").astype(str)
assert x.equals(y)
r = {
    "status": "PASS_ALL_22_PRE_CONTROL_FIELDS_AND_INDUSTRY_VALUES_EXACT",
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "rows_checked": len(f),
    "selected_unique_source_records": len(source),
    "numeric_values_checked": len(f) * len(base),
    "fields": records,
    "industry_rows_checked": len(f),
    "method": "Independent exact stock/date source lookup after separate nearest-predecessor timing validation; no asof join reused.",
    "verification_script_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
}
(RUN / "independent_pre_control_values.json").write_text(
    json.dumps(r, ensure_ascii=False, indent=2), encoding="utf8"
)
print(r["status"], r["numeric_values_checked"])
