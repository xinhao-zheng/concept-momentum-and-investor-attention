# Point-in-time covariates

`input_build.py` constructs daily covariates from local raw files. Use `prepare_inputs.py` to run it with the configured data layout, or provide `--daily-prices`, `--market-index`, `--overlays` and `--output` explicitly. `input_build_contract.json` records fields and rules.

The reconstruction uses a history start of 2016-07-13 and the stated zero-amount convention. The original historical constructor was not recovered. Full key, value and missingness comparisons support the reconstructed calculation; they do not certify historical first-publication timestamps.

`apply_observation_index.py` selects the published date/security keys from the reconstructed panel. The indexed output is `output_root/pit/stock_pit_covariates_daily.universe.parquet`. Source fields and business names remain unchanged. Corrected files in the configured overlay take precedence by source folder and filename.

All inputs are filtered through 2026-08-31. Input checksums, rejected rows and output support are recorded. Use a new output directory when changing raw files or settings.
