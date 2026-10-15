# Local execution configuration

Copy `reproduction/config/local.example.json` to `local.json` in the repository root and set the paths for your machine.

| Field | Role | Relative-path base |
|---|---|---|
| `data_root` | External input storage | Configuration file's directory |
| `snapshot_root` | Research input tree described in [RAW_REBUILD.md](../RAW_REBUILD.md) | Resolved `data_root` |
| `output_root` | Generated inputs, results and execution records | Configuration file's directory |

Absolute paths are accepted. A null `output_root` selects `<repository>/reproduction/work`. Resolving this root adds no date or release name; E1 and market-risk runs create timestamped subdirectories within it to keep executions separate. The output directory must be outside `snapshot_root`. Input and output paths in code resolve through this configuration; the example contains no workstation-specific directory.

The published observation cutoff is fixed at `2026-08-31`; the loader rejects a different date. `threads` controls the numerical worker limits. Pass the JSON through `--config` on a public runner or set `CONCEPT_QUANT_CONFIG`. Input constructors with explicit path flags can receive paths from the same configuration through their coordinating runner.

`local.json`, `local.*.json`, local notes matching `local.*.md` and the default output directories are ignored by Git. Keep other local configurations and output locations outside version control. This configuration does not require credentials.
