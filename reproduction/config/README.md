# Local execution configuration

Copy `local.example.json` to a local JSON file. `data_root` locates separately obtained input data. `snapshot_root` resolves relative to it, unless an absolute path is supplied. `output_root` must be outside the input snapshot; a relative path resolves beside the configuration file. A null value selects `reproduction/work`.

The published observation cutoff is fixed at `2026-08-31`; the loader rejects a different date. `threads` controls the numerical worker limits. Pass the JSON through `--config` on a public runner or set `CONCEPT_QUANT_CONFIG`. Input constructors with explicit path flags can receive paths from the same configuration through their coordinating runner.

Keep local configuration files and generated work outside version control. No credentials belong in this file.
