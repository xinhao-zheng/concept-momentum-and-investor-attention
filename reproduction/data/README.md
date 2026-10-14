# Research observation indices

`pit-observation-index.parquet` contains the 8,995,748 date/security keys of the retained covariate panel. It contains no prices or popularity measurements. `apply_observation_index.py` joins newly constructed fields to this index and rejects missing or duplicated keys.

The key set is preserved research input. Its original row-selection program is unavailable, so it should not be interpreted as an independently reconstructed historical publication rule. Feature construction and observation selection are separate operations. The model training and evaluation samples apply their additional rules after this join.

The Parquet file is stored with Git LFS. Fetch LFS objects after cloning the repository.

`attention-observation-index.parquet` contains the 8,974,107 date/security keys used for the daily attention panel, across 5,236 securities from 1 December 2016 through 31 August 2026. It contains no popularity values, prices, model outcomes or concept strings. `code/analysis/attention_inputs.py` recomputes the attention fields from external raw observations on these declared keys.

The historical attention builder retained its existing valid-score keys through 30 June 2026 and appended keys satisfying the same gamma observation-clock score rule after that date. This released index preserves the resulting research sample; it is not evidence that the original historical row-selection process or publication timestamps have been independently reconstructed. Its hash and coverage are in `attention-observation-index.json`.
