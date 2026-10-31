# External Support / Conflict Intensity — Integration Policy and Pipeline

This repository is a **reproducible data-integration scaffold**, not an empirical result and not a bundle of the source datasets. Download the exact releases from the official sources, verify their codebooks, and place the files at the configured paths before running the pipeline.

The central rule is: **do not convert an unexplained blank, missing row, unmatched ID, or out-of-coverage year into zero**. The pipeline preserves unresolved values, creates audit reports, and stops when a core integration invariant fails.

## Quick start

1. Install Python 3.10+ and dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Use one version-aligned official set of **UCDP Dyadic**, **ESD TY**, and
   **ESD DY** files. The core `dyad_id` is release-specific. The configuration
   records the raw files currently found in this project, but the preflight
   gate will stop if their declared versions differ or if a core file is absent.
   For the original 1976–2017 design, download the v18.1 trio and update all
   three `files` paths plus all three `ucdp_versions` entries in
   `config/policy.yaml` together. A consistent naming convention is:

   ```text
   data/raw/ucdp/ucdp_dyadic_181.csv
   data/raw/ucdp/ucdp_esd_ty_181.csv
   data/raw/ucdp/ucdp_esd_dy_181.csv
   ```

   Alternatively, use a fully version-aligned current-release set and revise
   `ucdp_versions` and the paths in `config/policy.yaml` together. Do not
   substitute ESD TY for ESD DY: DY is the authority for the explicit
   support/no-support status.

3. Review `config/policy.yaml`. In particular, verify the region allow-list and exact country aliases. The default region filter is an initial study choice, not a verified geopolitical crosswalk.

4. Profile the files before integration:

   ```bash
   python src/audit_inputs.py
   ```

5. Build the core dyad-year panel:

   ```bash
   python src/pipeline.py
   ```

6. Run tests for critical integration policies:

   ```bash
   python -m unittest discover -s tests -v
   ```

Tests use tiny synthetic examples; they do not establish that a user's downloaded source files have been imported correctly.

## Current raw-data preflight

The present raw directory contains `ucdp-esd-ty-181.csv` (ESD TY v18.1),
`Dyadic_v26_1.csv` (Dyadic v26.1), and `BattleDeaths_v26_1_conf.csv`
(conflict-level BRD v26.1). It does not contain the required ESD DY file.
`python src/pipeline.py` therefore writes
`reports/data_audit/core_source_preflight.csv` and stops before it can make an
invalid cross-release join. The conflict-level BRD file is intentionally not
used; obtain the dyadic BRD file matching the selected Dyadic release for the
robustness outcome.

## Version and unit lock

The main analysis locks **UCDP Dyadic v18.1 + UCDP ESD v18.1** because ESD v18.1 uses the Dyadic v18.1 `dyad_id` system. The final panel key is `(dyad_id, outcome_year)`. The outcome is observed in year `t`; external-support features and country-year controls are drawn from the exact year `t-1`.

Do **not** merge ESD v18.1 straight into current Dyadic v26.1 based only on IDs. UCDP changed identifiers across releases. If newer data are used, apply official ID translation files and report the translation coverage and unresolved keys separately.

## Sample and geography

- Default main sample: conflict types `3` and `4` in UCDP Dyadic v18.1 (internal and internationalized-internal conflicts).
- Default country allow-list: Israel, Palestine, Jordan, Lebanon, Syria, Iraq, and Yemen. Edit the list to match your final operational definition of the Levant.
- Current regional assignment uses the normalized government-side `country_a` label; it does not claim to locate every battle geographically. Review `reports/data_audit/region_filter_audit.csv` before freezing the sample.
- Exact aliases are allowed; fuzzy string matching is not.
- The main sample includes only dyads observed in both `t-1` and `t`. A missing previous-year row is not interpreted as zero deaths, no conflict, or no support.
- Interstate and extrasystemic conflicts are excluded from the main sample. They require a separate rule for geographic relevance and country controls on both sides.

## ESD DY/TY integration and H3

- **DY** is the dyad-year authority for `ext_sup` (recorded support present/absent).
- **TY** preserves the supporter-recipient-opponent-year details and is needed to identify which primary party received support.
- `actor_id` is mapped against the same-version Dyadic side actor IDs; it is not directly equivalent to `side_a_id` or `side_b_id`.
- H3's “none / one side / both sides” is calculated by checking whether each side received any support, possibly from different supporters. `ext_bothsides` is not used for H3: it means the *same external supporter* provided support to both primary parties.
- Main exposure: confirmed support (`ext_sup=1`, `ext_alleged=0`). Sensitivity: all recorded support including alleged support.
- Support type variables are potentially simultaneous indicators; `ext_p` remains distinct from `ext_x`. `ext_u` remains a separate feature but does not count as a known type in the support-type breadth variable.
- Supporter counts use distinct supporter IDs, not TY rows. Coalition contributors may be listed separately, so the pipeline outputs raw and coalition-collapsed counts as alternative measures.

If a supported TY row lacks a supporter ID, the supporter count is underidentified and remains missing. If a supported TY row lacks a binary support or allegation flag, or an expected support-type flag is unexpectedly missing, the pipeline writes an audit CSV and stops rather than interpreting the blank as zero.

## Missingness policy

See `DATA_INTEGRATION_POLICY.md` for the full policy. In brief:

- `ext_sup=0` in ESD DY is the only condition that authorizes zero-filling source-described blank support-detail fields.
- Absent active dyad rows, absent ESD keys, missing/sentinel outcomes, and missing country-year controls are not structural zeroes.
- Missing WDI/NMC/V-Dem controls are preserved in the master panel; no primary-analysis interpolation or forward/backfill is done.
- Crosswalk entries are time-valid. A duplicate/ambiguous mapping for a source country-year is a hard error; unresolved entries are written to an audit queue.
- Model-stage imputation, if ultimately necessary, must be fit on training data only and compared with complete-case results.

## Optional source integrations

### COW NMC, WDI, V-Dem

The pipeline accepts these files in tidy country-year staging format:

- `data/raw/cow/nmc_country_year.csv`: `cow_ccode, year, cinc`
- `data/raw/world_bank/wdi_country_year.csv`: `iso3, year, population, gdp_per_capita`
- `data/raw/vdem/vdem_country_year.csv`: `vdem_country_text_id, year, v2x_polyarchy`

The pipeline reads the official COW NMC and V-Dem country-year downloads directly. It also extracts `SP.POP.TOTL` and `NY.GDP.PCAP.KD` from the official wide WDI bulk CSV at run time, producing an internal ISO3-year representation without modifying `data/raw/`. Complete `data/templates/country_crosswalk.csv` with mappings verified against official source definitions and valid dates; the pipeline will not guess or auto-fill the mappings. The blank template does not contain a verified Yemen succession mapping.

### Battle-related deaths robustness outcome

Optional input:

```text
data/raw/ucdp/ucdp_brd_dyadic_181.csv
```

Use the **dyadic BRD v18.1** file, not the conflict-level file, so the source unit matches `(dyad_id, year)`. The pipeline exports `model_dataset_battle_deaths_sensitivity.csv` using the observed best-estimate deaths and `log1p(battle_deaths_best)`. Unmatched keys and missing death counts are audited and remain missing; no zero filling occurs. If you use a later BRD version, first create and validate an official UCDP ID translation and a separately documented extension.

### COW Formal Alliances and MID

These are not direct merges into the core dyad-year panel:

- COW Formal Alliances v4.1 is state-state data, with member, dyad, and directed-dyad formats; terms can be asymmetric. Create a separate state-pair-year table and define whether the relationship is supporter–government, supporter–opponent-government, or another explicit state pair before collapsing to conflict-dyad-year.
- MID 5.0 contains dispute-/participant-/incident-level data and must not be merged row-for-row into the panel. For a state-pair-year extension, consider the documented dyadic MID data (the COW page lists Dyadic MID 4.03 for 1816–2014) or aggregate event/participant files first. Do not treat a civil conflict as itself a MID.

Coverage of alliances/MID should restrict only those robustness models, not the primary sample.

## Outputs

After a successful run, the pipeline writes:

- `data/processed/dyad_year_master.csv`: the integrated panel, with no imputation
- `data/processed/model_dataset.csv`: confirmed-support exposure specification
- `data/processed/model_dataset_all_recorded_support_sensitivity.csv`: all-recorded-support sensitivity specification
- `data/processed/model_dataset_battle_deaths_sensitivity.csv`: optional BRD sensitivity dataset, only if the source is provided and has observed counts

## Model analysis

After a successful pipeline run, execute:

```bash
python src/analyze_models.py
```

The script fits M0–M4 Logistic Regression models with dyad-clustered standard errors, then fits Random Forest and XGBoost models using the M4 feature set and a time-ordered test split beginning in 2009. M1 adds `n_support_types_confirmed`, the breadth of confirmed support types, while M2–M4 retain the existing CINC, population, GDP per capita, and V-Dem blocks. Focused supplementary specifications test H2 with each confirmed support type, H3 with one-sided versus both-sided support, and escalation with `minor_to_war` among dyad-years whose prior intensity was minor. The default model dataset uses confirmed support; run the same command with `--dataset data/processed/model_dataset_all_recorded_support_sensitivity.csv` for the alleged-support sensitivity analysis.

The tree models are predictive supplements, not causal estimators. Missing rows are excluded per model's complete-case sample; no outcome or control value is imputed by this script.

The analysis also writes publication-style PDF and PNG figures: logistic odds-ratio forest plots, separate ROC-AUC/Brier test-performance panels, descriptive support-mechanism panels, and an XGBoost SHAP beeswarm. The tree and SHAP importance CSV files remain available for numerical reporting, but the figures emphasize interpretable outcome relationships rather than treating model-specific importance indices as causal effects. The duplicate `year` column is de-duplicated during model-sample construction.
- `data/interim/`: standardized source extracts and support feature table
- `reports/data_audit/`: region decisions, sample flow, schema profile, duplicate keys, unmatched IDs, ESD DY/TY discrepancies, crosswalk issues, missingness, and run summary

No real source data is shipped with this scaffold. A successful run on synthetic data is only a software smoke test, not evidence about the research hypotheses.

## Source manifest and citations

Record version, downloaded file name, date, official URL, citation, and SHA-256 in a copy of `config/source_manifest_template.yaml` named `config/source_manifest.yaml`. Keep versioned raw files rather than overwriting them in place. Codebook and official source references are in `REFERENCES.md` and `DATA_INTEGRATION_POLICY.md`.
