# External Support / Conflict Intensity — Integration Policy and Pipeline

This repository is a **reproducible data-integration scaffold**, not an empirical result and not a bundle of the source datasets. Download the exact releases from the official sources, verify their codebooks, and place the files at the configured paths before running the pipeline.

The central rule is: **do not convert an unexplained blank, missing row, unmatched ID, or out-of-coverage year into zero**. The pipeline preserves unresolved values, creates audit reports, and stops when a core integration invariant fails.

## Quick start

1. Install Python 3.10+ and dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Use the fixed, version-aligned **UCDP v18.1** set of Dyadic, ESD TY, and
   ESD DY files. The core `dyad_id` is release-specific and the preflight gate
   rejects any source declaration that differs from `project.ucdp_version`.
   The configured paths are:

   ```text
   data/raw/ucdp/ucdp_dyadic_181.csv
   data/raw/ucdp/ucdp_esd_ty_181.csv
   data/raw/ucdp/ucdp_esd_dy_181.csv
   ```

   Raw source files are not distributed in this GitHub repository; download
   the official v18.1 files locally and place them at the configured paths.
   Do not substitute ESD TY for ESD DY: DY is the authority for the explicit
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

The configured raw paths are checked before integration. If a core file is
absent, or if any declared release differs from v18.1, `python src/pipeline.py`
writes `reports/data_audit/core_source_preflight.csv` and stops before it can
make an invalid cross-release join. Raw UCDP files are intentionally omitted
from GitHub and must be supplied locally.

## Version and unit lock

The main analysis fixes **UCDP Dyadic v18.1 + UCDP ESD v18.1** because ESD v18.1 uses the Dyadic v18.1 `dyad_id` system. The final panel retains current `conflict_id` and lagged `lag_conflict_id` alongside the `(dyad_id, outcome_year)` key. The outcome is observed in year `t`; external-support features and country-year controls are drawn from the exact year `t-1`.

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

The script fits inferential Logistic Regression specifications with dyad-clustered standard errors, then evaluates baseline and extended predictive specifications using L2 Logistic Regression, Random Forest, and XGBoost. H1 is operationalized in M1 as the breadth of confirmed support types (`n_support_types_confirmed`), because the binary support-presence indicator is nearly constant in this restricted sample. M2_CINC, M3_GDP, and M4_VDEM are alternative adjusted specifications estimated on one common complete-case sample; M2_POP_ALT uses log population as a separate population alternative. These models do not automatically combine CINC with population or GDP per capita with V-Dem. Logistic inference is separated from prediction: it reports sample size, outcome counts, dyad/conflict counts, pseudo-R², AIC, odds ratios with 95% confidence intervals, and raw p-values, without in-sample classification metrics. M0/M1 use a common M1 complete-case sample for the primary comparison; M0/M1 and all alternative adjusted specifications are also estimated on the common adjusted sample with explicit `sample_specification` labels. Spearman correlations and VIF diagnostics are exported for description and diagnosis, not automatic model selection. Focused supplementary specifications test H2 with each confirmed support type, applying Benjamini–Hochberg FDR correction only to the support-type terms; H3 with one-sided versus both-sided support after excluding `unknown` and `unresolved_side` rows; and escalation with `minor_to_war` among dyad-years whose prior intensity was minor. The default model dataset uses confirmed support; run the same command with `--dataset data/processed/model_dataset_all_recorded_support_sensitivity.csv` for the alleged-support sensitivity analysis.

The tree models are predictive supplements, not causal estimators. Missing rows are excluded per model's complete-case sample; no outcome or control value is imputed by this script.

The analysis also writes publication-style PDF and PNG figures: logistic odds-ratio forest plots, grouped-CV performance comparisons, descriptive support-mechanism panels, and an XGBoost SHAP beeswarm. The primary predictive validation is five-fold `StratifiedGroupKFold` on the M1 complete-case sample, holding out complete dyads so observations from the same dyad never appear in both train and test. Baseline (M0) and extended (M1) specifications use identical folds; Average Precision is primary, with Brier score and ROC-AUC as secondary metrics. By default, three configurable splitter seeds are evaluated; estimator seeds are separate and are averaged within each fold. Fold means are averaged within each split repetition, then split-repetition means and standard deviations are reported. Neither repeated split folds nor estimator seeds are additional observations. Use `--cv-split-seeds 42 1051 2059` to set the splitter states explicitly. Fold, estimator-seed, repetition, and observation-level prediction results are saved in `grouped_cv_fold_metrics.csv`, `grouped_cv_seed_metrics.csv`, `grouped_cv_repetition_metrics.csv`, and `grouped_cv_predictions.csv`. `grouped_cv_conflict_overlap.csv` reports train/test conflict overlap as a diagnostic: dyad-grouped CV evaluates held-out-dyad generalization, not generalization to entirely unseen conflicts. Forward-looking temporal validation is optional via `--temporal-validation` and is a robustness analysis, not a prerequisite for grouped CV. The tree and SHAP importance CSV files remain available for numerical reporting, but the figures emphasize interpretable outcome relationships rather than treating model-specific importance indices as causal effects. A non-converged logistic fit is marked in the metrics and supplementary output rather than silently interpreted. The duplicate `year` column is de-duplicated during model-sample construction.

### Analysis output interpretation

- `logistic_coefficients.csv`: inferential Logistic Regression estimates, odds ratios, standard errors, p-values, and 95% OR confidence intervals. These are association estimates, not predictive scores.
- `control_spearman_correlations.csv` and `control_vif.csv`: descriptive correlation and design-matrix collinearity diagnostics for the alternative country-control specifications. They are not automatic model-selection rules.
- `model_metrics.csv`: inferential model summaries plus grouped-CV predictive summaries. Logistic rows contain sample and fit statistics; predictive rows contain fold-aggregated AP, Brier, ROC-AUC, and specification comparisons.
- `support_hypothesis_coefficients.csv`: coefficient-level results for H2, the original three-category H3, the exploratory supported-dyad H3 comparison, and exploratory Minor-to-War models. H2 rows additionally include Benjamini–Hochberg-adjusted p-values for valid support-type terms only; non-estimable models retain diagnostic rows with inferential values set to missing.
- `support_hypothesis_metrics.csv`: sample sizes, outcome counts, convergence and estimability diagnostics, H2 attempted/valid/non-estimable counts, and H3 category/sample metadata for the supplementary hypotheses. `H3_supported_dyads` compares both-sided with one-sided confirmed support only; it is not a support-versus-no-support estimate. The H2 correction family contains only estimable support-type tests.
- `grouped_cv_metrics.csv`: legacy grouped-CV filename retained by older runs; current runs write `grouped_cv_fold_metrics.csv` and `grouped_cv_seed_metrics.csv` to distinguish fold averages from seed-level sensitivity results.
- `grouped_cv_repetition_metrics.csv`: one row per splitter seed, specification, and algorithm after averaging estimator seeds within folds and folds within the repetition.
- `grouped_cv_predictions.csv`: held-out prediction rows for every successful estimator-seed/fold/split combination.
- `grouped_cv_conflict_overlap.csv`: train/test conflict counts and shared conflict IDs for each grouped-CV fold. Shared conflicts are expected under dyad grouping and are diagnostic only.
- `tree_feature_importance.csv`: model-specific tree split importance for predictive interpretation only; it is not an inferential effect estimate.
- `xgboost_shap_values.csv`: held-out Extended XGBoost SHAP attribution values with fold, dyad, and `observation_year` identifiers. The feature column `year` remains an attribution column; SHAP values are on the model's raw predicted log-odds scale and describe prediction, not causal effects.
- `xgboost_shap_feature_values.csv`: original held-out feature values aligned row-for-row and column-for-column with `xgboost_shap_values.csv`; it is supplied as the feature-data matrix to the beeswarm plot.
- `xgboost_shap_importance.csv`: mean absolute held-out SHAP magnitude by feature, summarizing predictive contribution rather than statistical significance.
- `data/interim/`: standardized source extracts and support feature table
- `reports/data_audit/`: region decisions, sample flow, schema profile, duplicate keys, unmatched IDs, ESD DY/TY discrepancies, crosswalk issues, missingness, and run summary

No real source data is shipped with this scaffold. A successful run on synthetic data is only a software smoke test, not evidence about the research hypotheses.

## Source manifest and citations

Record version, downloaded file name, date, official URL, citation, and SHA-256 in a copy of `config/source_manifest_template.yaml` named `config/source_manifest.yaml`. Keep versioned raw files rather than overwriting them in place. Codebook and official source references are in `REFERENCES.md` and `DATA_INTEGRATION_POLICY.md`.
