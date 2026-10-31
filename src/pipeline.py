"""Reproducible integration pipeline for UCDP ESD v18.1 and UCDP Dyadic v18.1.

The pipeline intentionally fails closed on missing core keys. It does not download data,
perform fuzzy joins, infer zero support from a missing record, or impute the outcome.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SENTINELS = {-99, -999, -9999}

# Keys are canonical names; aliases are compared after removing punctuation and lowercasing.
ALIASES: dict[str, set[str]] = {
    # UCDP Dyadic v18.1
    "dyad_id": {"dyadid", "dyad_id"},
    "year": {"year"},
    "intensity": {"int", "intensity", "intensity_level", "intensitylevel"},
    "conflict_type": {"type", "typeofconflict", "conflict_type"},
    "country_a": {"countrya", "country_a", "sidea", "side_a"},
    "country_b": {"countryb", "country_b", "sideb", "side_b"},
    "location": {"location"},
    "start_date": {"startdate", "start_date", "start_date_of_conflict"},
    "gwno_a": {"gwnoa", "gwno_a"},
    "gwno_b": {"gwnob", "gwno_b"},
    "side_a_id": {"sideaid", "side_a_id"},
    "side_b_id": {"sidebid", "side_b_id"},
    "conflict_id": {"conflictid", "conflict_id"},
    # ESD TY / DY
    "actor_id": {"actorid", "actor_id"},
    "oppo_id": {"oppoid", "oppo_id"},
    "ext_id": {"extid", "ext_id"},
    "ext_sup": {"extsup", "ext_sup"},
    "ext_alleged": {"extalleged", "ext_alleged"},
    "active": {"active"},
    "ext_nonstate": {"extnonstate", "ext_nonstate"},
    "ext_coalition": {"extcoalition", "ext_coalition"},
    "ext_coalition_name": {"extcoalitionname", "ext_coalition_name"},
    "ext_elements": {"extelements", "ext_elements"},
    "ext_bothsides": {"extbothsides", "ext_bothsides"},
    "ext_count": {"extcount", "ext_count"},
    "ext_x": {"extx", "ext_x"},
    "ext_p": {"extp", "ext_p"},
    "ext_y": {"exty", "ext_y"},
    "ext_w": {"extw", "ext_w"},
    "ext_m": {"extm", "ext_m"},
    "ext_t": {"extt", "ext_t"},
    "ext_f": {"extf", "ext_f"},
    "ext_i": {"exti", "ext_i"},
    "ext_l": {"extl", "ext_l"},
    "ext_o": {"exto", "ext_o"},
    "ext_u": {"extu", "ext_u"},
    "ext_sum": {"extsum", "ext_sum"},
    # COW NMC normalized inputs
    "cow_ccode": {"cow_ccode", "ccode", "ccode1", "state", "cowcode", "stateid"},
    "cinc": {"cinc"},
    # WDI normalized inputs
    "iso3": {"iso3", "countrycode", "country_code", "iso3code"},
    "population": {"population", "sp_pop_totl"},
    "gdp_per_capita": {"gdp_per_capita", "gdppc", "ny_gdp_pcap_kd", "ny_gdp_pcap_cd"},
    # V-Dem normalized inputs
    "vdem_country_text_id": {"vdem_country_text_id", "country_text_id", "countrytextid"},
    "v2x_polyarchy": {"v2x_polyarchy"},
    # UCDP Battle-Related Deaths Dataset (dyadic format; exact aliases only).
    "battle_deaths_best": {"bd_best", "brd_best", "best_est", "best_estimate", "bestestimate", "battle_deaths_best", "best_deaths", "bd_best_estimate", "deaths_best", "best_estimate_brd"},
    # Time-aware crosswalk
    "gwno": {"gwno", "gwno_code", "gwno_a"},
    "valid_from": {"valid_from", "validfrom"},
    "valid_to": {"valid_to", "validto"},
    "ucdp_country_name": {"ucdp_country_name", "country_name", "countryname"},
    "notes": {"notes", "note"},
}


def _key(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _clean_id(v: Any) -> Any:
    if pd.isna(v):
        return pd.NA
    s = str(v).strip()
    if s in {"", "nan", "None", "<NA>", "-99", "-999", "-9999"}:
        return pd.NA
    # CSV parsers sometimes turn integer IDs into strings like "123.0".
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def _clean_number(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.mask(x.isin(SENTINELS), np.nan)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Download the official dataset, place it at the configured path, and update config/policy.yaml."
        )
    # UCDP/COW files usually use UTF-8 or UTF-8 with BOM. Keep names as text at ingest.
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def standardize_columns(df: pd.DataFrame, wanted: set[str], source_name: str,
                        required: set[str] | None = None) -> pd.DataFrame:
    """Rename recognized raw column names to canonical names without fuzzy guessing."""
    required = required or set()
    by_key = {_key(c): c for c in df.columns}
    rename: dict[str, str] = {}
    missing: list[str] = []
    for canonical in wanted:
        aliases = ALIASES.get(canonical, {canonical})
        hit = None
        # Exact normalized aliases only.
        for alias in aliases | {canonical}:
            if _key(alias) in by_key:
                hit = by_key[_key(alias)]
                break
        if hit is None:
            if canonical in required:
                missing.append(canonical)
        else:
            rename[hit] = canonical
    if missing:
        raise ValueError(
            f"{source_name}: required canonical columns not found: {missing}. "
            f"Observed columns: {list(df.columns)}. Check the codebook/schema or edit ALIASES in src/pipeline.py."
        )
    out = df.rename(columns=rename).copy()
    # A source should not create ambiguous duplicate canonical columns.
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()].tolist()
        raise ValueError(f"{source_name}: duplicate columns after standardization: {dupes}")
    return out


def clean_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dyad_id" in df:
        df["dyad_id"] = df["dyad_id"].map(_clean_id).astype("string")
    if "year" in df:
        df["year"] = _clean_number(df["year"]).astype("Int64")
    return df


def canonical_country(value: Any, aliases: dict[str, str]) -> str | None:
    if pd.isna(value):
        return None
    s = re.sub(r"\s+", " ", str(value).strip()).lower()
    if not s or s in {"-99", "-999", "-9999", "nan", "none"}:
        return None
    if s.startswith("government of "):
        s = s.removeprefix("government of ").strip()
    return aliases.get(s, s)


def parse_id_set(value: Any) -> set[str]:
    """Parse UCDP actor-ID fields conservatively (single IDs or semicolon/comma lists)."""
    if pd.isna(value):
        return set()
    s = str(value).strip()
    if s in {"", "-99", "-999", "-9999", "nan", "None"}:
        return set()
    # Do not split hyphens: IDs are expected to be numeric actor IDs.
    parts = re.split(r"[;,|]", s)
    return {x for x in (_clean_id(p) for p in parts) if x is not pd.NA and not pd.isna(x)}


def numeric_ids(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = out[c].map(_clean_id).astype("string")
    return out


def assert_unique(df: pd.DataFrame, keys: list[str], label: str, audit_dir: Path) -> None:
    dup = df[df.duplicated(keys, keep=False)].sort_values(keys)
    if not dup.empty:
        audit_dir.mkdir(parents=True, exist_ok=True)
        dup.to_csv(audit_dir / f"duplicates_{label}.csv", index=False)
        raise ValueError(f"{label} is not unique on {keys}; details written to {audit_dir / f'duplicates_{label}.csv'}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping.")
    return cfg


def validate_core_source_set(cfg: dict[str, Any], audit_dir: Path) -> None:
    """Audit core-source availability and block unsafe UCDP ID joins.

    ESD's dyad IDs are release-specific.  A matching column name is therefore
    not evidence that an ESD row can be joined to a differently-versioned
    Dyadic or BRD file.  This check runs before any source is standardized so
    that an incomplete or mixed raw directory gets a useful audit artifact
    rather than a downstream schema or merge error.
    """
    files = cfg.get("files", {})
    versions = {key: str(value).strip() for key, value in cfg.get("ucdp_versions", {}).items()
                if value is not None and str(value).strip()}
    required = ["dyadic", "esd_ty", "esd_dy"]
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for role in required:
        configured = files.get(role)
        path = ROOT / configured if configured else None
        exists = bool(path and path.exists())
        records.append({
            "role": role,
            "configured_path": str(path) if path else None,
            "exists": exists,
            "declared_version": versions.get(role),
        })
        if not exists:
            missing.append(role)

    audit_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(records)
    report.to_csv(audit_dir / "core_source_preflight.csv", index=False)
    (audit_dir / "core_source_preflight.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")

    if missing:
        locations = ", ".join(
            f"{role} ({files.get(role, 'no path configured')})" for role in missing)
        raise FileNotFoundError(
            "Core UCDP source set is incomplete: " + locations + ". "
            "Do not derive ESD DY from ESD TY; obtain the official ESD DY file for the same release. "
            "See reports/data_audit/core_source_preflight.csv."
        )

    core_versions = {versions.get(role) for role in required}
    if None in core_versions or len(core_versions) != 1:
        translation = cfg.get("ucdp_versions", {}).get("id_translation")
        translation_path = ROOT / translation if translation else None
        shown = ", ".join(f"{role}={versions.get(role, 'undeclared')}" for role in required)
        translation_note = (
            " A translation table is configured, but this pipeline does not yet implement or validate "
            "that crosswalk, so it is not used."
            if translation_path and translation_path.exists() else
            ""
        )
        raise ValueError(
            "UCDP core files do not share one declared ID release (" + shown + "). "
            "Do not merge dyad_id values across releases. Replace the files with a same-version set, "
            "or implement and validate an official ID translation before enabling a cross-version run." +
            translation_note + " See reports/data_audit/core_source_preflight.csv."
        )


def dyadic_schema(raw: pd.DataFrame) -> pd.DataFrame:
    wanted = {"dyad_id", "year", "intensity", "conflict_type", "country_a", "country_b",
              "location", "start_date", "gwno_a", "gwno_b", "side_a_id", "side_b_id", "conflict_id"}
    required = {"dyad_id", "year", "intensity", "conflict_type", "country_a", "side_a_id", "side_b_id"}
    out = standardize_columns(raw, wanted, "UCDP Dyadic v18.1", required)
    out = clean_key_columns(out)
    for c in ["intensity", "conflict_type", "gwno_a", "gwno_b", "conflict_id"]:
        if c in out:
            out[c] = _clean_number(out[c])
    out["side_a_id"] = out["side_a_id"].map(_clean_id).astype("string")
    out["side_b_id"] = out["side_b_id"].map(_clean_id).astype("string")
    out["country_a"] = out["country_a"].astype("string").replace({"<NA>": pd.NA, "-99": pd.NA, "-999": pd.NA})
    if "country_b" not in out:
        out["country_b"] = pd.NA
    if "location" not in out:
        out["location"] = pd.NA
    if "gwno_a" not in out:
        out["gwno_a"] = np.nan
    # Start date may be stored as YYYY or a full date; parse conservatively.
    if "start_date" not in out:
        out["start_date"] = pd.NA
    def _start_year(v: Any) -> float:
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if re.fullmatch(r"\d{4}", s):
            return float(s)
        dt = pd.to_datetime(s, errors="coerce")
        return float(dt.year) if not pd.isna(dt) else np.nan
    out["startdate_year"] = out["start_date"].map(_start_year)
    return out


def get_dyadic_region(dy: pd.DataFrame, cfg: dict[str, Any], audit_dir: Path) -> pd.DataFrame:
    p = cfg["project"]
    aliases = {str(k).strip().lower(): str(v).strip().lower() for k, v in p.get("country_aliases", {}).items()}
    include = {canonical_country(x, aliases) for x in p["region_country_labels"]}
    # The main sample is internal and internationalized-internal conflict only.
    types = [int(x) for x in p["main_conflict_types"]]
    out = dy.copy()
    out["country_a_norm"] = out["country_a"].map(lambda x: canonical_country(x, aliases))
    if p.get("use_location_fallback_for_region", False):
        out["location_norm"] = out["location"].map(lambda x: canonical_country(x, aliases))
        out["region_key"] = out["country_a_norm"].where(out["country_a_norm"].notna(), out["location_norm"])
    else:
        out["region_key"] = out["country_a_norm"]
    out["region_in_scope"] = out["region_key"].isin(include)
    out["type_in_scope"] = out["conflict_type"].isin(types)
    audit = (out.groupby(["country_a", "location", "conflict_type", "region_key", "region_in_scope", "type_in_scope"],
                         dropna=False).size().reset_index(name="rows")
             .sort_values("rows", ascending=False))
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_dir / "region_filter_audit.csv", index=False)
    filtered = out.loc[out["region_in_scope"] & out["type_in_scope"]].copy()
    # Episode age is computed from contiguous active dyad-year runs, not from the first-ever start date.
    filtered = filtered.sort_values(["dyad_id", "year"]).copy()
    year_gap = filtered.groupby("dyad_id")["year"].diff()
    new_episode = year_gap.ne(1).fillna(True)
    filtered["active_episode_number"] = new_episode.groupby(filtered["dyad_id"]).cumsum().astype(int)
    filtered["episode_age_years"] = filtered.groupby(["dyad_id", "active_episode_number"]).cumcount() + 1
    if filtered.empty:
        raise ValueError("Region/type filter produced zero rows. Inspect reports/data_audit/region_filter_audit.csv and revise the explicit region rule.")
    assert_unique(filtered, ["dyad_id", "year"], "dyadic_region", audit_dir)
    return filtered


def build_consecutive_panel(dy: pd.DataFrame, cfg: dict[str, Any], audit_dir: Path) -> pd.DataFrame:
    p = cfg["project"]
    keys = ["dyad_id", "year", "intensity", "conflict_type", "country_a", "country_b", "location",
            "gwno_a", "side_a_id", "side_b_id", "episode_age_years", "active_episode_number"]
    keys = [c for c in keys if c in dy.columns]
    current = dy[keys].copy()
    current = current[(current["year"] >= int(p["outcome_year_start"])) &
                      (current["year"] <= int(p["outcome_year_end"]))].copy()
    prev = dy[keys].copy()
    prev = prev.rename(columns={c: f"lag_{c}" for c in keys if c not in {"dyad_id", "year"}})
    prev = prev.rename(columns={"year": "lag_source_year"})
    prev["target_year"] = prev["lag_source_year"] + 1
    # Exact year-key left join makes missing t-1 rows visible in the audit.
    paired = current.merge(prev, left_on=["dyad_id", "year"], right_on=["dyad_id", "target_year"],
                           how="left", validate="one_to_one", indicator="_lag_merge")
    no_lag = paired["_lag_merge"] != "both"
    audit_dir.mkdir(parents=True, exist_ok=True)
    n_target = len(paired)
    n_no_lag = int(no_lag.sum())
    base = paired[paired["_lag_merge"] == "both"].copy()
    base = base.drop(columns=["target_year", "_lag_merge"])
    base["lag_year"] = base["lag_source_year"].astype("Int64")
    base = base.drop(columns=["lag_source_year"])

    outcome_known = base["intensity"].isin([1, 2])
    lag_known = base["lag_intensity"].isin([1, 2])
    audit_counts = pd.DataFrame([
        {"stage": "regional_type_filtered_outcome_year_rows", "rows": n_target},
        {"stage": "no_exact_t_minus_1_dyad_row_excluded", "rows": n_no_lag},
        {"stage": "consecutive_active_dyad_pairs_before_intensity_check", "rows": len(base)},
        {"stage": "outcome_intensity_unknown_or_invalid", "rows": int((~outcome_known).sum())},
        {"stage": "lag_intensity_unknown_or_invalid", "rows": int((~lag_known).sum())},
    ])
    audit_counts.to_csv(audit_dir / "sample_flow.csv", index=False)
    base = base[outcome_known & lag_known].copy()
    base["war_t"] = (base["intensity"] == 2).astype("int8")
    base["war_lag1"] = (base["lag_intensity"] == 2).astype("int8")
    base["minor_to_war"] = ((base["lag_intensity"] == 1) & (base["intensity"] == 2)).astype("int8")
    if base.empty:
        raise ValueError("No consecutive active dyad-year pairs survived the sample rules.")
    return base


def esd_schema(raw: pd.DataFrame, label: str, require_detail: bool) -> pd.DataFrame:
    wanted = {"dyad_id", "year", "ext_sup", "active", "actor_id", "oppo_id", "ext_id", "ext_alleged",
              "ext_nonstate", "ext_coalition", "ext_coalition_name", "ext_elements", "ext_bothsides",
              "ext_count", "ext_sum", "ext_x", "ext_p", "ext_y", "ext_w", "ext_m", "ext_t", "ext_f",
              "ext_i", "ext_l", "ext_o", "ext_u"}
    if require_detail:
        required = {"dyad_id", "year", "ext_sup", "active", "actor_id", "ext_id", "ext_alleged"}
    else:
        required = {"dyad_id", "year", "ext_sup"}
    out = standardize_columns(raw, wanted, label, required)
    out = clean_key_columns(out)
    for c in ["active", "ext_sup", "ext_alleged", "ext_nonstate", "ext_coalition", "ext_elements",
              "ext_bothsides", "ext_count", "ext_sum", "ext_x", "ext_p", "ext_y", "ext_w", "ext_m",
              "ext_t", "ext_f", "ext_i", "ext_l", "ext_o", "ext_u"]:
        if c in out:
            out[c] = _clean_number(out[c])
    for c in ["actor_id", "oppo_id", "ext_id"]:
        if c in out:
            out[c] = out[c].map(_clean_id).astype("string")
    if "ext_coalition_name" not in out:
        out["ext_coalition_name"] = pd.NA
    return out


def _flag_max(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float((vals == 1).any())


def map_recipient_side(actor_id: Any, side_a_id: Any, side_b_id: Any) -> str:
    actor = _clean_id(actor_id)
    if pd.isna(actor):
        return "unmapped"
    a_ids, b_ids = parse_id_set(side_a_id), parse_id_set(side_b_id)
    in_a, in_b = actor in a_ids, actor in b_ids
    if in_a and not in_b:
        return "a"
    if in_b and not in_a:
        return "b"
    if in_a and in_b:
        return "ambiguous"
    return "unmapped"


def build_esd_features(esd_ty: pd.DataFrame, esd_dy: pd.DataFrame, dy_all: pd.DataFrame,
                       audit_dir: Path, type_variables: list[str],
                       type_count_variables: list[str] | None = None) -> pd.DataFrame:
    # ext_u is retained as a feature but does not count as a known support type in breadth measures.
    if type_count_variables is None:
        type_count_variables = [v for v in type_variables if v != "ext_u"]
    unknown_count_variables = sorted(set(type_count_variables) - set(type_variables))
    if unknown_count_variables:
        raise ValueError(f"Support-type count variables are not included in the feature list: {unknown_count_variables}")
    # ESD DY must be one record per dyad-year and supplies the explicit dyad-level support/no-support status.
    assert_unique(esd_dy, ["dyad_id", "year"], "esd_dy", audit_dir)
    # Inactive ESD years are not fully covered; the main design uses active dyad-years only.
    if "active" in esd_dy:
        invalid_active = esd_dy[esd_dy["active"].isna() | ~esd_dy["active"].isin([0, 1])].copy()
        if not invalid_active.empty:
            invalid_active.to_csv(audit_dir / "esd_dy_invalid_active_flag.csv", index=False)
            raise ValueError("ESD DY has missing/non-binary active flags; inspect esd_dy_invalid_active_flag.csv")
        esd_dy = esd_dy[esd_dy["active"] == 1].copy()
    if esd_dy["ext_sup"].isna().any() or not esd_dy["ext_sup"].dropna().isin([0, 1]).all():
        esd_dy.loc[esd_dy["ext_sup"].isna() | ~esd_dy["ext_sup"].isin([0, 1])].to_csv(
            audit_dir / "esd_dy_invalid_support_flag.csv", index=False)
        raise ValueError("ESD DY ext_sup has missing or non-binary values; do not recode these to zero.")
    dy_support = esd_dy[["dyad_id", "year", "ext_sup"]].rename(columns={"ext_sup": "any_support_all"}).copy()
    dy_support["any_support_all"] = dy_support["any_support_all"].astype("int8")

    # The TY file uses the recipient as actor_id; map that actor back to side A/B using the same-version Dyadic IDs.
    ty = esd_ty.copy()
    if "active" in ty:
        invalid_active = ty[ty["active"].isna() | ~ty["active"].isin([0, 1])].copy()
        if not invalid_active.empty:
            invalid_active.to_csv(audit_dir / "esd_ty_invalid_active_flag.csv", index=False)
            raise ValueError("ESD TY has missing/non-binary active flags; inspect esd_ty_invalid_active_flag.csv")
        ty = ty[ty["active"] == 1].copy()
    ty = ty[ty["year"].notna() & ty["dyad_id"].notna()].copy()

    # Fail closed on malformed ESD rows. The codebook's structural blanks for no-support
    # detail fields are resolved from DY ext_sup=0 below; a supported TY row with unknown
    # support/allegation/type coding must not silently become a zero.
    bad_ty_support = ty[ty["ext_sup"].isna() | ~ty["ext_sup"].isin([0, 1])].copy()
    if not bad_ty_support.empty:
        bad_ty_support.to_csv(audit_dir / "esd_ty_invalid_ext_sup.csv", index=False)
        raise ValueError("ESD TY active rows have missing/non-binary ext_sup; inspect esd_ty_invalid_ext_sup.csv")
    supported_ty = ty[ty["ext_sup"].eq(1)].copy()
    bad_alleged = supported_ty[supported_ty["ext_alleged"].isna() | ~supported_ty["ext_alleged"].isin([0, 1])].copy()
    if not bad_alleged.empty:
        bad_alleged.to_csv(audit_dir / "esd_ty_invalid_ext_alleged.csv", index=False)
        raise ValueError("Supported ESD TY rows have missing/non-binary ext_alleged; do not classify as confirmed or alleged")
    for v in type_variables:
        bad_type = supported_ty[supported_ty[v].isna() | ~supported_ty[v].isin([0, 1])].copy()
        if not bad_type.empty:
            bad_type.to_csv(audit_dir / f"esd_ty_invalid_{v}.csv", index=False)
            raise ValueError(f"Supported ESD TY rows have missing/non-binary {v}; see esd_ty_invalid_{v}.csv. Check downloaded schema/codebook before proceeding.")
    missing_supporter = supported_ty[supported_ty["ext_id"].isna()].copy()
    if not missing_supporter.empty:
        missing_supporter.to_csv(audit_dir / "esd_ty_missing_supporter_id.csv", index=False)

    side_map = dy_all[["dyad_id", "year", "side_a_id", "side_b_id"]].drop_duplicates(["dyad_id", "year"])
    ty = ty.merge(side_map, on=["dyad_id", "year"], how="left", validate="many_to_one", indicator="_side_map_merge")
    ty["recipient_side"] = ty.apply(lambda r: map_recipient_side(r.get("actor_id"), r.get("side_a_id"), r.get("side_b_id")), axis=1)
    ty["row_support_all"] = ty["ext_sup"].eq(1)
    ty["row_support_confirmed"] = ty["ext_sup"].eq(1) & ty["ext_alleged"].eq(0)
    ty["row_support_alleged"] = ty["ext_sup"].eq(1) & ty["ext_alleged"].eq(1)
    ty["row_support_classification_missing"] = ty["ext_sup"].eq(1) & ty["ext_alleged"].isna()
    for v in type_variables:
        if v not in ty:
            ty[v] = np.nan
        ty[f"{v}_confirmed_row"] = ty[v].eq(1) & ty["row_support_confirmed"]
        ty[f"{v}_all_row"] = ty[v].eq(1) & ty["row_support_all"]

    # Per-supporter rows include alleged support and coalition member rows. Produce both raw and coalition-collapsed counts.
    confirmed = ty[ty["row_support_confirmed"]].copy()
    all_support = ty[ty["row_support_all"]].copy()
    for d in [confirmed, all_support]:
        d["supporter_key_raw"] = d["ext_id"].astype("string")
        coalition_on = d.get("ext_coalition", pd.Series(np.nan, index=d.index)).eq(1)
        coalition_name = d.get("ext_coalition_name", pd.Series(pd.NA, index=d.index)).astype("string")
        use_coal = coalition_on & coalition_name.notna() & coalition_name.str.strip().ne("")
        d["supporter_key_coalition_collapsed"] = d["supporter_key_raw"]
        d.loc[use_coal, "supporter_key_coalition_collapsed"] = "coalition::" + coalition_name.loc[use_coal].str.lower().str.strip()

    def count_unique(frame: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["dyad_id", "year", name])
        return (frame[frame[col].notna() & frame[col].astype("string").ne("")]
                .groupby(["dyad_id", "year"])[col].nunique().rename(name).reset_index())

    # TY group summaries; no-support TY rows can be absent, so the DY file remains authoritative for zeros.
    grouped = ty.groupby(["dyad_id", "year"], dropna=False)
    summaries = grouped.agg(
        ty_detail_rows=("ext_sup", "size"),
        any_support_ty=("row_support_all", "max"),
        any_support_confirmed=("row_support_confirmed", "max"),
        any_support_alleged=("row_support_alleged", "max"),
        support_classification_missing=("row_support_classification_missing", "max"),
        supporter_id_missing_all=("ext_id", lambda s: bool((s.isna() & ty.loc[s.index, "row_support_all"]).any())),
        supporter_id_missing_confirmed=("ext_id", lambda s: bool((s.isna() & ty.loc[s.index, "row_support_confirmed"]).any())),
        support_side_unmapped_all=("recipient_side", lambda s: bool((((s == "unmapped") | (s == "ambiguous")) & ty.loc[s.index, "row_support_all"]).any())),
        support_side_unmapped_confirmed=("recipient_side", lambda s: bool((((s == "unmapped") | (s == "ambiguous")) & ty.loc[s.index, "row_support_confirmed"]).any())),
        support_side_a_all=("recipient_side", lambda s: bool(((s == "a") & ty.loc[s.index, "row_support_all"]).any())),
        support_side_b_all=("recipient_side", lambda s: bool(((s == "b") & ty.loc[s.index, "row_support_all"]).any())),
        support_side_a_confirmed=("recipient_side", lambda s: bool(((s == "a") & ty.loc[s.index, "row_support_confirmed"]).any())),
        support_side_b_confirmed=("recipient_side", lambda s: bool(((s == "b") & ty.loc[s.index, "row_support_confirmed"]).any())),
    ).reset_index()
    for v in type_variables:
        type_summary = ty.groupby(["dyad_id", "year"], dropna=False)[[f"{v}_confirmed_row", f"{v}_all_row"]].max().reset_index()
        type_summary = type_summary.rename(columns={f"{v}_confirmed_row": f"{v}_confirmed", f"{v}_all_row": f"{v}_all"})
        summaries = summaries.merge(type_summary, on=["dyad_id", "year"], how="left", validate="one_to_one")
    # Counts must be calculated from distinct IDs, not rows (one supporter can provide multiple types).
    summary_parts = [summaries,
                     count_unique(confirmed, "supporter_key_raw", "n_supporters_confirmed_raw"),
                     count_unique(confirmed, "supporter_key_coalition_collapsed", "n_supporters_confirmed_coalition_collapsed"),
                     count_unique(all_support, "supporter_key_raw", "n_supporters_all_raw"),
                     count_unique(all_support, "supporter_key_coalition_collapsed", "n_supporters_all_coalition_collapsed")]
    features = summary_parts[0]
    for part in summary_parts[1:]:
        features = features.merge(part, on=["dyad_id", "year"], how="left", validate="one_to_one")

    # Join official DY coverage flag to TY detail; disagreements are audit failures, not silently reconciled.
    comparison = dy_support.merge(features[["dyad_id", "year", "any_support_ty"]], on=["dyad_id", "year"], how="left", validate="one_to_one")
    comparison["any_support_ty_for_check"] = pd.to_numeric(comparison["any_support_ty"], errors="coerce").fillna(0).astype(int)
    comparison["mismatch"] = comparison["any_support_all"] != comparison["any_support_ty_for_check"]
    # Only active years will be analyzed, but retain the full mismatch report for inspection.
    comparison.loc[comparison["mismatch"]].to_csv(audit_dir / "esd_dy_ty_support_mismatch.csv", index=False)
    mismatches = comparison[comparison["mismatch"]]
    if not mismatches.empty:
        raise ValueError(f"ESD DY/TY support-presence mismatch in {len(mismatches)} dyad-years; see esd_dy_ty_support_mismatch.csv")

    # Reindex with ESD DY as the complete dyad-year universe, then apply documented zero rules.
    features = dy_support.merge(features, on=["dyad_id", "year"], how="left", validate="one_to_one")
    no_support = features["any_support_all"].eq(0)
    has_support = features["any_support_all"].eq(1)
    for c in [f"{v}_{mode}" for v in type_variables for mode in ["confirmed", "all"]]:
        features[c] = pd.to_numeric(features[c], errors="coerce").astype(float)
    # If ESD DY explicitly says no support, blank ESD detail fields are structural blanks and may become zero.
    features["any_support_confirmed"] = features["any_support_confirmed"].where(~no_support, 0)
    features["any_support_alleged"] = features["any_support_alleged"].where(~no_support, 0)
    # Preserve unknown side allocation as NA when support exists but TY detail/actor mapping is unresolved.
    for col in ["support_side_a_all", "support_side_b_all", "support_side_a_confirmed", "support_side_b_confirmed"]:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype(float)
        features.loc[no_support, col] = 0.0
    no_detail = has_support & features["ty_detail_rows"].isna()
    features.loc[no_detail, ["support_side_a_all", "support_side_b_all"]] = np.nan
    features.loc[features["support_side_unmapped_all"].astype("boolean").fillna(False).astype(bool), ["support_side_a_all", "support_side_b_all"]] = np.nan
    features.loc[features["support_side_unmapped_confirmed"].astype("boolean").fillna(False).astype(bool), ["support_side_a_confirmed", "support_side_b_confirmed"]] = np.nan
    for v in type_variables:
        features[f"{v}_confirmed"] = features[f"{v}_confirmed"].where(~no_support, 0)
        features[f"{v}_all"] = features[f"{v}_all"].where(~no_support, 0)
    count_cols = ["n_supporters_confirmed_raw", "n_supporters_confirmed_coalition_collapsed",
                  "n_supporters_all_raw", "n_supporters_all_coalition_collapsed"]
    for col in count_cols:
        features[col] = features[col].where(~no_support, 0)
    # If any supported TY row lacks a supporter ID, the supporter count is underidentified.
    # Do not retain a partial count as if it were complete.
    def _as_bool_false_on_missing(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes"}
    missing_supporter_all = features.get("supporter_id_missing_all", pd.Series(False, index=features.index)).map(_as_bool_false_on_missing)
    missing_supporter_confirmed = features.get("supporter_id_missing_confirmed", pd.Series(False, index=features.index)).map(_as_bool_false_on_missing)
    features.loc[missing_supporter_all, ["n_supporters_all_raw", "n_supporters_all_coalition_collapsed"]] = np.nan
    features.loc[missing_supporter_confirmed, ["n_supporters_confirmed_raw", "n_supporters_confirmed_coalition_collapsed"]] = np.nan
    # Supported DY rows without TY detail are unknown detail, not zero support-type features.
    no_detail = has_support & features["ty_detail_rows"].isna()
    for v in type_variables:
        features.loc[no_detail, [f"{v}_confirmed", f"{v}_all"]] = np.nan
    features.loc[no_detail, count_cols] = np.nan
    features["support_configuration_all"] = features.apply(_configuration, axis=1, args=("support_side_a_all", "support_side_b_all", "support_side_unmapped_all", "any_support_all"))
    features["support_configuration_confirmed"] = features.apply(_configuration, axis=1, args=("support_side_a_confirmed", "support_side_b_confirmed", "support_side_unmapped_confirmed", "any_support_confirmed"))
    for c in ["any_support_confirmed", "any_support_alleged"] + [f"{v}_{mode}" for v in type_variables for mode in ["confirmed", "all"]]:
        features[c] = pd.to_numeric(features[c], errors="coerce").astype(float)
    features["n_support_types_confirmed"] = features[[f"{v}_confirmed" for v in type_count_variables]].sum(axis=1, min_count=len(type_count_variables))
    features["n_support_types_all"] = features[[f"{v}_all" for v in type_count_variables]].sum(axis=1, min_count=len(type_count_variables))
    features["esd_detail_present"] = features["ty_detail_rows"].notna().astype("int8")
    features["support_side_unmapped_all"] = features["support_side_unmapped_all"].astype("boolean").fillna(False).astype(bool)
    features["support_side_unmapped_confirmed"] = features["support_side_unmapped_confirmed"].astype("boolean").fillna(False).astype(bool)
    features.loc[no_detail & has_support, "support_side_unmapped_all"] = True
    features["side_mapping_unresolved_all"] = features["support_side_unmapped_all"].astype("int8")
    features["side_mapping_unresolved_confirmed"] = features["support_side_unmapped_confirmed"].astype("int8")

    # Audit all supported rows whose recipient cannot be matched to a side. Review; do not infer a side.
    unmatched = ty[(ty["row_support_all"]) & ty["recipient_side"].isin(["unmapped", "ambiguous"])].copy()
    if not unmatched.empty:
        cols = [c for c in ["dyad_id", "year", "actor_id", "oppo_id", "ext_id", "ext_alleged", "recipient_side", "side_a_id", "side_b_id"] if c in unmatched]
        unmatched[cols].to_csv(audit_dir / "esd_recipient_side_unmatched.csv", index=False)
    return features


def _configuration(row: pd.Series, side_a_col: str, side_b_col: str, unresolved_col: str, any_col: str) -> str:
    any_support = row.get(any_col, np.nan)
    if pd.isna(any_support):
        return "unknown"
    if int(any_support) == 0:
        return "none"
    # A supported row with unassigned recipient side could belong to either side.
    unresolved = row.get(unresolved_col, False)
    if pd.notna(unresolved) and bool(unresolved):
        return "unresolved_side"
    a, b = row.get(side_a_col, np.nan), row.get(side_b_col, np.nan)
    if pd.isna(a) or pd.isna(b):
        return "unresolved_side"
    if bool(a) and bool(b):
        return "both_sides"
    if bool(a) or bool(b):
        return "one_side_only"
    return "unresolved_side"


def load_crosswalk(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    raw = read_csv(path)
    wanted = {"gwno", "cow_ccode", "iso3", "vdem_country_text_id", "valid_from", "valid_to", "ucdp_country_name", "notes"}
    cw = standardize_columns(raw, wanted, "time-aware country crosswalk", required=set())
    if "gwno" not in cw:
        return None
    cw["gwno"] = cw["gwno"].map(_clean_id).astype("string")
    usable = cw["gwno"].notna() & cw["gwno"].astype("string").ne("")
    cw = cw[usable].copy()
    if cw.empty:
        return None
    for c in ["valid_from", "valid_to", "cow_ccode"]:
        if c not in cw:
            cw[c] = np.nan
        cw[c] = _clean_number(cw[c])
    missing_dates = cw[cw["valid_from"].isna() | cw["valid_to"].isna()].copy()
    if not missing_dates.empty:
        missing_dates.to_csv(ROOT / "reports/data_audit/country_crosswalk_invalid_dates.csv", index=False)
        raise ValueError("Populated country crosswalk rows require explicit valid_from and valid_to years; see country_crosswalk_invalid_dates.csv")
    invalid_ranges = cw[cw["valid_from"] > cw["valid_to"]].copy()
    if not invalid_ranges.empty:
        invalid_ranges.to_csv(ROOT / "reports/data_audit/country_crosswalk_invalid_dates.csv", index=False)
        raise ValueError("Country crosswalk has valid_from > valid_to; see country_crosswalk_invalid_dates.csv")
    cw["valid_from"] = cw["valid_from"].astype(int)
    cw["valid_to"] = cw["valid_to"].astype(int)
    for c in ["iso3", "vdem_country_text_id"]:
        if c not in cw:
            cw[c] = pd.NA
        cw[c] = cw[c].astype("string").replace({"": pd.NA, "<NA>": pd.NA})
    return cw


def tidy_country_source(path: Path, source: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    raw = read_csv(path)
    if raw.empty:
        return None
    if source == "nmc":
        wanted = {"cow_ccode", "year", "cinc"}
        required = wanted
    elif source == "wdi":
        wanted = {"iso3", "year", "population", "gdp_per_capita"}
        required = {"iso3", "year"}
    elif source == "vdem":
        wanted = {"vdem_country_text_id", "year", "v2x_polyarchy"}
        required = {"vdem_country_text_id", "year"}
    else:
        raise ValueError(source)
    # The World Bank bulk download is a wide indicator file, not a country-year
    # table. Convert only the two indicators specified in the study design here;
    # every other source continues through the ordinary schema normalizer below.
    # This preserves the original download in data/raw and avoids an undocumented
    # intermediate manual conversion step.
    if source == "wdi" and "year" not in {_key(c) for c in raw.columns}:
        by_key = {_key(c): c for c in raw.columns}
        iso_col = by_key.get("countrycode")
        indicator_col = by_key.get("indicatorcode")
        if iso_col is None or indicator_col is None:
            raise ValueError(
                "WDI bulk file must contain Country Code and Indicator Code columns. "
                "Use the official WDI CSV download or provide a tidy country-year file."
            )
        year_cols = [c for c in raw.columns if re.fullmatch(r"\d{4}", str(c).strip())]
        if not year_cols:
            raise ValueError("WDI bulk file has no four-digit year columns.")
        requested = {
            "SP.POP.TOTL": "population",
            "NY.GDP.PCAP.KD": "gdp_per_capita",
        }
        long = raw.loc[raw[indicator_col].astype("string").str.upper().isin(requested),
                       [iso_col, indicator_col, *year_cols]].copy()
        long["variable"] = long[indicator_col].astype("string").str.upper().map(requested)
        long = long.melt(id_vars=[iso_col, "variable"], value_vars=year_cols,
                         var_name="year", value_name="value")
        long = long.rename(columns={iso_col: "iso3"})
        long["iso3"] = long["iso3"].astype("string").str.strip().str.upper()
        long["year"] = _clean_number(long["year"]).astype("Int64")
        long["value"] = _clean_number(long["value"])
        duplicates = long.duplicated(["iso3", "year", "variable"], keep=False)
        if duplicates.any():
            raise ValueError("WDI bulk file has duplicate ISO3-year-indicator values; inspect the source download.")
        out = (long.pivot(index=["iso3", "year"], columns="variable", values="value")
                   .reset_index().rename_axis(columns=None))
        for c in ["population", "gdp_per_capita"]:
            if c not in out:
                out[c] = np.nan
        assert_unique(out, ["iso3", "year"], "wdi_country_year", ROOT / "reports/data_audit")
        return out

    out = standardize_columns(raw, wanted, f"{source.upper()} country-year", required)
    out["year"] = _clean_number(out["year"]).astype("Int64")
    key = {"nmc": "cow_ccode", "wdi": "iso3", "vdem": "vdem_country_text_id"}[source]
    if key not in out:
        out[key] = pd.NA
    if source == "nmc":
        out[key] = _clean_number(out[key]).astype("Int64")
        out["cinc"] = _clean_number(out["cinc"])
    elif source == "wdi":
        out[key] = out[key].astype("string").str.strip().str.upper()
        for c in ["population", "gdp_per_capita"]:
            if c not in out:
                out[c] = np.nan
            out[c] = _clean_number(out[c])
    else:
        out[key] = out[key].astype("string").str.strip()
        if "v2x_polyarchy" not in out:
            out["v2x_polyarchy"] = np.nan
        out["v2x_polyarchy"] = _clean_number(out["v2x_polyarchy"])
    assert_unique(out, [key, "year"], f"{source}_country_year", ROOT / "reports/data_audit")
    return out


def attach_country_controls(base: pd.DataFrame, cfg: dict[str, Any], audit_dir: Path) -> pd.DataFrame:
    files = cfg["files"]
    out = base.copy()
    out["control_year"] = out["lag_year"].astype("Int64")
    out["gwno_lag"] = out.get("lag_gwno_a", pd.Series(pd.NA, index=out.index)).map(_clean_id).astype("string")
    cw_path = ROOT / files["country_crosswalk"]
    cw = load_crosswalk(cw_path)
    if cw is None:
        for c in ["cinc", "population", "gdp_per_capita", "v2x_polyarchy"]:
            if c not in out:
                out[c] = np.nan
        pd.DataFrame([{"source": "country_crosswalk", "status": "not available or template not populated"}]).to_csv(
            audit_dir / "optional_controls_status.csv", index=False)
        out[[c for c in ["gwno_lag", "control_year"] if c in out]].drop_duplicates().to_csv(
            audit_dir / "country_crosswalk_unmatched.csv", index=False)
        return out

    gw_years = out[["gwno_lag", "control_year"]].drop_duplicates()
    possible_all = gw_years.merge(cw, left_on="gwno_lag", right_on="gwno", how="left")
    possible = possible_all[(possible_all["control_year"] >= possible_all["valid_from"]) &
                            (possible_all["control_year"] <= possible_all["valid_to"])].copy()
    dup = possible.duplicated(["gwno_lag", "control_year"], keep=False)
    if dup.any():
        possible.loc[dup].to_csv(audit_dir / "country_crosswalk_ambiguous.csv", index=False)
        raise ValueError("Country crosswalk has multiple valid mappings for a GW code/year; resolve this manually.")
    cross_cols = ["gwno_lag", "control_year", "cow_ccode", "iso3", "vdem_country_text_id", "ucdp_country_name", "notes"]
    cross_cols = [c for c in cross_cols if c in possible]
    mapping = possible[cross_cols].drop_duplicates(["gwno_lag", "control_year"])
    matched_keys = mapping[["gwno_lag", "control_year"]].drop_duplicates().assign(crosswalk_matched=1)
    crosswalk_audit = gw_years.merge(matched_keys, on=["gwno_lag", "control_year"], how="left")
    crosswalk_audit["crosswalk_matched"] = crosswalk_audit["crosswalk_matched"].fillna(0).astype(int)
    crosswalk_audit.to_csv(audit_dir / "country_crosswalk_match_audit.csv", index=False)
    crosswalk_unmatched = crosswalk_audit[crosswalk_audit["crosswalk_matched"] == 0]
    if not crosswalk_unmatched.empty:
        crosswalk_unmatched.to_csv(audit_dir / "country_crosswalk_unmatched.csv", index=False)
    out = out.merge(mapping, on=["gwno_lag", "control_year"], how="left", validate="many_to_one")
    out["cow_ccode"] = _clean_number(out.get("cow_ccode", pd.Series(np.nan, index=out.index))).astype("Int64")
    out["iso3"] = out.get("iso3", pd.Series(pd.NA, index=out.index)).astype("string").str.upper()
    out["vdem_country_text_id"] = out.get("vdem_country_text_id", pd.Series(pd.NA, index=out.index)).astype("string")

    nmc_path = ROOT / files["nmc"]
    wdi_path = ROOT / files["wdi"]
    vdem_path = ROOT / files["vdem"]
    statuses = []
    optional = cfg.get("optional_controls", {})

    if optional.get("use_nmc", True):
        nmc = tidy_country_source(nmc_path, "nmc")
    else:
        nmc = None
    if nmc is not None:
        out = out.merge(nmc[["cow_ccode", "year", "cinc"]].rename(columns={"year": "control_year"}),
                        on=["cow_ccode", "control_year"], how="left", validate="many_to_one")
        statuses.append({"source": "NMC", "status": "loaded", "rows": len(nmc)})
    else:
        out["cinc"] = np.nan
        why = "disabled by config" if not optional.get("use_nmc", True) else "file missing or empty"
        statuses.append({"source": "NMC", "status": f"skipped ({why})", "rows": 0})

    if optional.get("use_wdi", True):
        wdi = tidy_country_source(wdi_path, "wdi")
    else:
        wdi = None
    if wdi is not None:
        keep = [c for c in ["iso3", "year", "population", "gdp_per_capita"] if c in wdi]
        out = out.merge(wdi[keep].rename(columns={"year": "control_year"}), on=["iso3", "control_year"],
                        how="left", validate="many_to_one")
        statuses.append({"source": "WDI", "status": "loaded", "rows": len(wdi)})
    else:
        out["population"] = np.nan
        out["gdp_per_capita"] = np.nan
        why = "disabled by config" if not optional.get("use_wdi", True) else "file missing or empty"
        statuses.append({"source": "WDI", "status": f"skipped ({why})", "rows": 0})

    if optional.get("use_vdem", True):
        vdem = tidy_country_source(vdem_path, "vdem")
    else:
        vdem = None
    if vdem is not None:
        out = out.merge(vdem[["vdem_country_text_id", "year", "v2x_polyarchy"]].rename(columns={"year": "control_year"}),
                        on=["vdem_country_text_id", "control_year"], how="left", validate="many_to_one")
        statuses.append({"source": "V-Dem", "status": "loaded", "rows": len(vdem)})
    else:
        out["v2x_polyarchy"] = np.nan
        why = "disabled by config" if not optional.get("use_vdem", True) else "file missing or empty"
        statuses.append({"source": "V-Dem", "status": f"skipped ({why})", "rows": 0})
    pd.DataFrame(statuses).to_csv(audit_dir / "optional_controls_status.csv", index=False)
    return out


def brd_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize version-aligned UCDP Battle-Related Deaths dyadic data.

    Only the dyadic-unit file is accepted. Conflict-level files generally do not carry
    the same analysis unit and must not be merged into a dyad-year panel.
    """
    wanted = {"dyad_id", "year", "battle_deaths_best"}
    required = wanted
    out = standardize_columns(raw, wanted, "UCDP Battle-Related Deaths dyadic dataset", required)
    out = clean_key_columns(out)
    out["battle_deaths_best"] = _clean_number(out["battle_deaths_best"])
    invalid = out["battle_deaths_best"].notna() & (out["battle_deaths_best"] < 0)
    if invalid.any():
        raise ValueError("BRD best-estimate death counts contain negative values; inspect source/coding.")
    return out[["dyad_id", "year", "battle_deaths_best"]]


def attach_brd_sensitivity(panel: pd.DataFrame, cfg: dict[str, Any], audit_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Optional secondary-outcome build; never zero-fill missing death counts."""
    path_value = cfg["files"].get("brd_dyadic")
    if not path_value:
        return {"status": "not_configured", "rows": 0}
    path = ROOT / path_value
    if not path.exists():
        return {"status": "file_missing_skipped", "rows": 0}
    raw = read_csv(path)
    brd = brd_schema(raw)
    assert_unique(brd, ["dyad_id", "year"], "brd_dyadic", audit_dir)
    # Historical v18.1 BRD should use the same UCDP ID system as the primary v18.1 panel.
    # A different version must be translated separately, not silently matched.
    out = panel.merge(brd, on=["dyad_id", "year"], how="left", validate="one_to_one", indicator="_brd_merge")
    out["brd_key_matched"] = out["_brd_merge"].eq("both").astype("int8")
    out["log1p_battle_deaths_best"] = np.log1p(out["battle_deaths_best"].where(out["battle_deaths_best"] >= 0))
    unmatched = out.loc[out["_brd_merge"] != "both", ["dyad_id", "year", "_brd_merge"]].copy()
    if not unmatched.empty:
        unmatched.to_csv(audit_dir / "brd_dyad_year_unmatched.csv", index=False)
    missing = out.loc[out["battle_deaths_best"].isna(), ["dyad_id", "year", "brd_key_matched"]].copy()
    if not missing.empty:
        missing.to_csv(audit_dir / "brd_deaths_missing.csv", index=False)
    # Retain only observations with an actually observed BRD outcome for this sensitivity set.
    sens = out[out["battle_deaths_best"].notna()].drop(columns=["_brd_merge"]).copy()
    sens.to_csv(out_dir / "model_dataset_battle_deaths_sensitivity.csv", index=False)
    status = {
        "status": "loaded",
        "input_rows": int(len(brd)),
        "panel_rows": int(len(panel)),
        "key_unmatched_rows": int((out["_brd_merge"] != "both").sum()),
        "death_count_missing_rows": int(out["battle_deaths_best"].isna().sum()),
        "sensitivity_rows_with_observed_deaths": int(len(sens)),
        "year_min": int(sens["year"].min()) if len(sens) else None,
        "year_max": int(sens["year"].max()) if len(sens) else None,
        "policy": "Missing BRD records/counts remain missing; no zero filling. Use dyadic BRD v18.1 or a separately audited official ID crosswalk."
    }
    (audit_dir / "brd_sensitivity_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def save_missingness_audit(df: pd.DataFrame, audit_dir: Path) -> None:
    rows = []
    for c in df.columns:
        n = int(df[c].isna().sum())
        rows.append({"variable": c, "n_rows": len(df), "n_missing": n, "missing_rate": (n / len(df) if len(df) else np.nan),
                     "dtype": str(df[c].dtype)})
    pd.DataFrame(rows).sort_values("missing_rate", ascending=False).to_csv(audit_dir / "missingness_by_variable.csv", index=False)


def run(config_path: Path) -> None:
    cfg = load_config(config_path)
    files = cfg["files"]
    out_dir = ROOT / files["output_dir"]
    interim_dir = ROOT / files["interim_dir"]
    audit_dir = ROOT / files["audit_dir"]
    for d in [out_dir, interim_dir, audit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Fail before parsing or joining: a familiar dyad_id column is not a
    # cross-release identifier.  The raw files and declared releases are
    # recorded even when the run cannot proceed.
    validate_core_source_set(cfg, audit_dir)

    # 1) Load and standardize the core UCDP files.
    dy_raw = read_csv(ROOT / files["dyadic"])
    dy_all = dyadic_schema(dy_raw)
    dy_region = get_dyadic_region(dy_all, cfg, audit_dir)
    dy_region.to_csv(interim_dir / "dyadic_region_v18_1.csv", index=False)

    # 2) Pair outcome year t with the exact same dyad at t-1.
    base = build_consecutive_panel(dy_region, cfg, audit_dir)
    base.to_csv(interim_dir / "consecutive_active_base.csv", index=False)

    # 3) Load the ESD TY and DY versions. Both must be the v18.1 version compatible with Dyadic v18.1.
    ty_raw = read_csv(ROOT / files["esd_ty"])
    dy_esd_raw = read_csv(ROOT / files["esd_dy"])
    ty = esd_schema(ty_raw, "UCDP ESD TY v18.1", require_detail=True)
    esd_dy = esd_schema(dy_esd_raw, "UCDP ESD DY v18.1", require_detail=False)
    ty.to_csv(interim_dir / "esd_ty_standardized.csv", index=False)
    esd_dy.to_csv(interim_dir / "esd_dy_standardized.csv", index=False)
    type_vars = [v for v in cfg.get("support_type_variables", []) if v in ty.columns]
    missing_type_vars = [v for v in cfg.get("support_type_variables", []) if v not in ty.columns]
    if missing_type_vars:
        raise ValueError(f"ESD TY is missing expected support type variables: {missing_type_vars}")
    type_count_vars = [v for v in cfg.get("support_type_count_variables", [x for x in type_vars if x != "ext_u"]) if v in type_vars]
    esd_features = build_esd_features(ty, esd_dy, dy_all, audit_dir, type_vars, type_count_vars)
    esd_features.to_csv(interim_dir / "esd_dyad_year_features.csv", index=False)

    # 4) Merge support features for the prior year. Missing key match is not interpreted as no support.
    support_lag = esd_features.rename(columns={"year": "lag_year"})
    panel = base.merge(support_lag, on=["dyad_id", "lag_year"], how="left", validate="many_to_one", indicator="_esd_merge")
    unmatched_esd = panel[panel["_esd_merge"] != "both"].copy()
    if not unmatched_esd.empty:
        unmatched_esd[["dyad_id", "year", "lag_year", "_esd_merge"]].to_csv(audit_dir / "esd_key_unmatched_for_core_sample.csv", index=False)
        raise ValueError(f"{len(unmatched_esd)} active lag dyad-years did not match ESD DY v18.1. Do not fill with zero; inspect audit file/version IDs.")
    panel = panel.drop(columns=["_esd_merge"])

    # 5) Add optional country-level controls from a time-aware manual crosswalk.
    panel = attach_country_controls(panel, cfg, audit_dir)

    # 6) Audit sample, key uniqueness, and missingness. Do not impute here.
    assert_unique(panel, ["dyad_id", "year"], "model_panel", audit_dir)
    panel = panel.sort_values(["dyad_id", "year"]).reset_index(drop=True)
    panel.to_csv(out_dir / "dyad_year_master.csv", index=False)

    # Model-ready copy preserves both confirmed-only and all-recorded exposure measures.
    model = panel.copy()
    model["any_support_main"] = model["any_support_confirmed"]
    model["support_configuration_main"] = model["support_configuration_confirmed"]
    for v in type_vars:
        model[f"{v}_main"] = model[f"{v}_confirmed"]
    model.to_csv(out_dir / "model_dataset.csv", index=False)
    if cfg["project"].get("alleged_support_sensitivity", True):
        sens = panel.copy()
        sens["any_support_main"] = sens["any_support_all"]
        sens["support_configuration_main"] = sens["support_configuration_all"]
        for v in type_vars:
            sens[f"{v}_main"] = sens[f"{v}_all"]
        sens.to_csv(out_dir / "model_dataset_all_recorded_support_sensitivity.csv", index=False)

    brd_status = attach_brd_sensitivity(panel, cfg, audit_dir, out_dir)
    save_missingness_audit(panel, audit_dir)
    # Save compact descriptive counts for the report.
    stats = {
        "rows": int(len(panel)),
        "unique_dyads": int(panel["dyad_id"].nunique()),
        "outcome_year_min": int(panel["year"].min()),
        "outcome_year_max": int(panel["year"].max()),
        "war_rows": int(panel["war_t"].sum()),
        "war_share": float(panel["war_t"].mean()),
        "minor_to_war_transitions": int(panel["minor_to_war"].sum()),
        "any_confirmed_support_rows": int(panel["any_support_confirmed"].fillna(0).sum()),
        "any_recorded_support_rows": int(panel["any_support_all"].fillna(0).sum()),
        "side_mapping_unresolved_rows_all_support": int(panel["side_mapping_unresolved_all"].fillna(0).sum()),
        "side_mapping_unresolved_rows_confirmed_support": int(panel["side_mapping_unresolved_confirmed"].fillna(0).sum()),
        "battle_deaths_sensitivity": brd_status,
        "note": "No missing values were imputed by this pipeline. Country-control gaps remain missing and are audited."
    }
    (audit_dir / "run_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nSaved master panel: {out_dir / 'dyad_year_master.csv'}")
    print(f"Saved model dataset: {out_dir / 'model_dataset.csv'}")
    print(f"Audits: {audit_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/policy.yaml", help="Path relative to project root or absolute YAML config")
    args = parser.parse_args()
    p = Path(args.config)
    config_path = p if p.is_absolute() else (ROOT / p)
    try:
        run(config_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
