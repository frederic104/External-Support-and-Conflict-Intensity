# Data Integration and Missingness Policy

## Purpose and scope

This document fixes the rules for building the regional dyad-year panel before model fitting. The pipeline is designed for the Levant, Iraq, and Yemen study, initially restricted to UCDP conflict types 3 and 4 (internal and internationalized-internal conflict) and years compatible with UCDP ESD v18.1. It is not a substitute for inspecting each downloaded file and the exact codebook shipped with that release.

Throughout, distinguish:

1. **Source fact**: a field/unit/coverage definition documented by the dataset producer.
2. **Study policy**: an explicit modeling or engineering choice made for this research.
3. **Unresolved data issue**: a case the pipeline should report and not infer away.

## 1. Version and unit policy

### Primary UCDP sources

- **UCDP Dyadic v18.1**: one observation per conflict dyad-year. The core outcome is `Int`: `1` = Minor, `2` = War. Relevant missing/sentinel values (such as `-99`) are treated as unknown, not as substantive categories.
- **UCDP ESD v18.1 DY**: dyad-year aggregate and authoritative source for `ext_sup` presence/absence in the main integration.
- **UCDP ESD v18.1 TY**: supporter-recipient-opponent-year detail, used to identify the recipient side, support type, and external supporter.
- **Key**: `(dyad_id, year)` for the dyad-year panel. ESD v18.1 was built against Dyadic v18.1. Never merge these IDs directly to a newer UCDP release without applying and validating the official historical-ID translation table.

The outcome year is `t`; predictors are sourced from exactly `t-1`. We do not use row-wise `shift(1)` to create lags. We use the explicit dyad ID and year key so a gap such as 2000 → 2002 cannot be bridged as if 2001 existed.

### Primary sample rule

The YAML default includes conflict types 3 and 4. The region is initially attributed by the government-side conflict-country label (`country_a`) against an explicit allow-list. The UCDP country/Location fields should not be read as a guarantee that every battle took place physically inside that territory. The researcher must inspect `region_filter_audit.csv` and manually revise the allow-list/aliases before freezing the sample.

The primary sample consists of dyads observed in both `t-1` and `t`. It is therefore a continuing-active-dyad sample. A missing row for `t-1` is not coded as inactivity, zero deaths, or no external support. This sample definition conditions on conflict activity and does not estimate conflict termination.

## 2. ESD support policy

### 2.1 DY is the support-status authority; TY is the detail source

- `ESD DY.ext_sup = 0`: explicitly recorded no support. This is the only condition that authorizes structural zero-filling for the support-detail fields that the ESD codebook says are blank when no support was provided.
- `ESD DY.ext_sup = 1`: support is recorded. TY must provide compatible detail for the dyad-year. A missing TY record or disagreement between DY `ext_sup` and TY support records is audited and stops the pipeline.
- Missing/non-binary `ext_sup` is never recoded to zero.
- Missing/non-binary `active` status in an ESD file is treated as a schema/data issue and stops the pipeline; `active=0` rows are excluded from the main exposure construction.
- ESD inactive dyad-years are not used as fully observed exposure histories, because the ESD codebook notes incomplete coding for inactive years.

### 2.2 Confirmed support and alleged support

Main exposure: a TY record is treated as confirmed only when support is recorded (`ext_sup = 1`) and `ext_alleged = 0`.

Sensitivity exposure: include all recorded support (`ext_sup = 1`) regardless of whether it is alleged, using the source coding. This is an analytic sensitivity check, not a claim that alleged support is verified.

For supported TY rows, missing/non-binary `ext_alleged` or missing/non-binary expected support-type indicators are not silently converted to zero. The pipeline writes a diagnostic CSV and stops; inspect the downloaded release, the exact codebook, and schema before deciding whether a documented source rule permits recoding.

### 2.3 Support types

The `ext_x`, `ext_p`, `ext_y`, `ext_w`, `ext_m`, `ext_t`, `ext_f`, `ext_i`, `ext_l`, `ext_o`, and `ext_u` flags are treated as separate, potentially simultaneous indicators. In particular, `ext_x` (troop support) and `ext_p` (foreign troop presence below the support threshold) are not collapsed into one measure. `ext_u` is retained as its own feature but excluded from `n_support_types_*`, which counts only known types; otherwise an unknown support type would be counted as a known category.

- An observed support-type flag of 1 is retained as 1.
- An observed flag of 0 is retained as 0.
- If the dyad-year explicitly has `ext_sup = 0`, codebook-described blank detail fields can be set to 0 for support-type dummies.
- If `ext_sup = 1` but a support-type field is unexpectedly missing/non-binary, the pipeline stops for review; it does not presume that blank means “type absent.”
- A recorded unknown type (`ext_u`) is a source category and must not be confused with a missing value.

### 2.4 Recipient-side mapping and H3

ESD TY `actor_id` identifies the recipient actor within the recipient-opponent relationship; it is not directly interchangeable with Dyadic `side_a_id` or `side_b_id`. The pipeline compares the TY `actor_id` with each same-version Dyadic side's actor-ID membership.

The study's H3 variable (“support absent / one side supported / both sides supported”) is derived by determining whether each primary side receives any support. `ESD ext_bothsides` does something different: it identifies whether the **same supporter** supports both primary parties. It is retained as source information if needed, but is not used as the H3 configuration variable.

If a support recipient cannot be assigned uniquely to side A or side B, the pipeline sets the configuration to unresolved and writes `esd_recipient_side_unmatched.csv`. It must not infer the recipient side from the external supporter's country.

### 2.5 Supporter counts and coalition records

Supporters are counted from distinct external supporter IDs, not TY rows. Multiple support types by the same supporter do not increase that supporter's count. If a supported TY row lacks `ext_id`, the corresponding supporter count becomes missing/underidentified rather than a partial count presented as complete.

Because ESD TY may list contributors of a coalition separately, the pipeline produces:

- a raw unique external-supporter-ID count; and
- an exploratory count that collapses identical coded coalition names where the source marks a coalition.

These are alternative constructions. The coalition-collapsed count is not presumed superior; inspect the audit and report the definition used in each model.

## 3. Structural zeros, unknown values, and absent rows

These situations are not interchangeable:

| Case | Policy | Reason |
|---|---|---|
| Source has an explicit valid 0 | Keep 0 | Observed absence for that variable under the codebook |
| Source cell is blank but ESD DY explicitly has `ext_sup=0`, and field is a codebook-described no-support detail | Set that detail to 0 | Structural blank is explained by recorded no support |
| Source cell is blank while ESD DY has `ext_sup=1` | Keep unresolved / fail closed for required type fields | Absence of information is not absence of support/type |
| Required ESD DY dyad-year key is absent for an active lag observation | Stop and report unmatched key | Does not establish zero support |
| Dyadic row for `t-1` is absent | Exclude from continuing-active sample and count in sample-flow audit | Does not establish zero deaths or no conflict |
| Dyadic `Int` is missing/sentinel/invalid | Exclude that outcome row from supervised outcome sample and report it | Do not infer intensity |
| Country-year control is missing | Keep NA in master; report missingness | Conflict can affect national statistics and reporting |
| Country code / crosswalk row is unresolved | Do not merge that source; report manual mapping queue | No fuzzy match or automatic country succession assumption |
| BRD record/count is absent | Keep battle-death outcome missing in that sensitivity data | Do not infer zero from an absent record |

The project does not automatically infer whether an absent record is a true zero, missing measurement, out-of-coverage year, a changed ID, or a non-applicable case. The source-specific key and coverage audits must establish which interpretation is defensible.

## 4. Region and actor identity policy

The current region allow-list is an initial operationalization, not a final geographical truth. Review the audit of raw `country_a`, `Location`, conflict type, and inclusion status before any substantive analysis. The code performs exact normalized string matching with an explicit alias table; it does not use fuzzy matching.

Do not assume that all historical political entities with similar names are a single uninterrupted country-year unit. Yemen is a specific example: North Yemen, South Yemen, and unified Yemen require time-valid and source-consistent mappings where those entities appear. The provided country crosswalk is an empty template, not a verified crosswalk.

For COW NMC, WDI, and V-Dem, first normalize to one row per source country ID-year and join through the time-aware crosswalk. The default conflict types 3 and 4 are represented by the government-side country for country-level controls. Interstate cases require separately defined side-A and side-B country controls and are not automatically compatible with this one-sided control setup.

## 5. Country-year controls

- **COW NMC/CINC**: treat as national material capability. Do not interpret CINC as rebel capability or as the actual volume of support delivered. No value is created for non-state actors.
- **World Bank WDI**: use the requested indicators (initially population and real GDP per capita) with exact source-year joins. The primary analysis does not zero-fill, forward-fill, backward-fill, or interpolate country-years. If a model requires imputation, fit it on the training fold only and compare against a complete-case analysis.
- **V-Dem**: initially include at most a small, theory-motivated set of governance indicators, not the entire codebook. Preserve the exact release/version and its country-text ID; map it through the crosswalk. Treat missing values and entity changes transparently.
- **Crosswalk**: a match must be valid for the year (`valid_from <= year <= valid_to`). More than one valid row for the same UCDP country code-year is a hard error. An unresolved mapping is not fixed by selecting the closest-looking country name.

## 6. Battle-related deaths robustness outcome

Use the **dyadic** UCDP Battle-Related Deaths file matching version 18.1 when available. The official historical download page distinguishes dyadic BRD from conflict-level BRD; use dyadic BRD because the analysis key is dyad-year. The code exports an optional sensitivity panel with the best-estimate death count and `log1p(battle_deaths_best)`.

BRD v18.1 is used to avoid silently mixing UCDP ID systems. If a newer BRD release is selected, use official ID translation tables and create a separate, documented extension. Records or counts absent from the BRD file remain missing; they are not automatically set to zero. The BRD outcome is only used for rows with observed counts in its output sensitivity file.

## 7. COW alliances and MIDs policy

These are not direct add-on columns for the primary conflict-dyad panel.

- **Formal Alliances v4.1**: state-state alliance terms, with member/dyad/directed-dyad formats and potentially asymmetric member terms. Build a separate state-pair-year table first. A conflict involving a non-state actor cannot be matched directly to an interstate alliance dyad. If used, define whether the relationship is supporter–government, supporter–opponent-government, or another explicit state pair, apply year and direction rules, then collapse to the conflict dyad-year without row multiplication.
- **MIDs**: MID 5.0 provides dispute-, participant-, and incident-level materials; do not direct-merge those rows to conflict-dyad-year. If a state-pair-year robustness measure is needed, prefer a documented dyadic MID file (the COW page lists Dyadic MID 4.03 for 1816–2014), or explicitly aggregate participant/event data before joining. Define whether the measured relation is between the external supporter and a conflict-party state; an ongoing civil conflict is not itself automatically a MID.
- Do not make alliances or MIDs a required condition for the main sample; their coverage and entity requirements differ. Any such extension must output its own state-pair mapping and row-count audit.

## 8. Tests and fail-closed behavior

The pipeline must stop rather than guess when any of the following occurs:

1. A required source file/schema/key is absent.
2. A source expected to be unique on its merge key has duplicates.
3. ESD DY has invalid `ext_sup`.
4. An active supported TY row has unknown `ext_sup`, unknown allegation status, or unexpected missing/non-binary type fields.
5. ESD DY support presence disagrees with ESD TY records.
6. A supported observation's side assignment is unresolved (it is audited and the configuration remains unresolved; this is not necessarily a global stop, but must not be silently assigned).
7. A time-valid crosswalk yields multiple mappings for one UCDP code-year.
8. A core active lag dyad-year fails to match ESD DY.

A run is considered ready for substantive analysis only after reviewing: source schema profile, region filter decisions, sample flow, ESD DY/TY mismatch report, recipient-side unmatched report, crosswalk unmatched/ambiguous reports, variable-level missingness, and the source manifest with versions/dates/hashes.

## 9. Official sources used for the policy

- UCDP ESD v18.1 codebook: https://ucdp.uu.se/downloads/extsup/ESD/ucdp-esd-181.pdf
- UCDP Dyadic v18.1 codebook: https://ucdp.uu.se/downloads/dyadic/ucdp-dyadic-181.pdf
- UCDP historical data and ID translation tables: https://ucdp.uu.se/downloads/olddw.html
- UCDP ESD article: Meier, Karlén, Pettersson, & Croicu (2023), DOI: https://doi.org/10.1177/00223433221079864
- COW NMC: https://correlatesofwar.org/data-sets/national-material-capabilities/
- World Bank WDI sources and methods: https://datatopics.worldbank.org/world-development-indicators/sources-and-methods.html
- V-Dem data and codebooks: https://www.v-dem.net/data/the-v-dem-dataset/
- COW Formal Alliances v4.1: https://correlatesofwar.org/data-sets/formal-alliances/
- COW MIDs and Dyadic MID: https://correlatesofwar.org/data-sets/mids/
