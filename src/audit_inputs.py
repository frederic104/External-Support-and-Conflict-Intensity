"""List source schema, year coverage, keys, and apparent missingness before building the panel."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/policy.yaml")
    args = parser.parse_args()
    cp = Path(args.config)
    cp = cp if cp.is_absolute() else ROOT / cp
    cfg = yaml.safe_load(cp.read_text(encoding="utf-8"))
    audit_dir = ROOT / cfg["files"]["audit_dir"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    records = []
    key_candidates = {
        "dyadic": ["dyadid", "year"],
        "esd_ty": ["dyadid", "year", "actorid", "oppoid", "extid"],
        "esd_dy": ["dyadid", "year"],
        "brd_dyadic": ["dyadid", "year"],
        "nmc": ["cowccode", "year"],
        "wdi": ["iso3", "year"],
        "vdem": ["vdemcountrytextid", "year"],
    }
    for key in ["dyadic", "esd_ty", "esd_dy", "brd_dyadic", "nmc", "wdi", "vdem", "country_crosswalk"]:
        path = ROOT / cfg["files"].get(key, "")
        if not path.exists():
            records.append({"source": key, "path": str(path), "status": "missing", "rows": None,
                            "columns": None, "year_min": None, "year_max": None, "duplicate_candidate_keys": None})
            continue
        try:
            df = read(path)
            normalizer = lambda value: "".join(ch for ch in str(value).lower() if ch.isalnum())
            normalized = {normalizer(c): c for c in df.columns}
            year_col = normalized.get("year")
            years = pd.to_numeric(df[year_col], errors="coerce") if year_col else pd.Series(dtype=float)
            candidate = key_candidates.get(key, [])
            hit_cols = [normalized[c] for c in candidate if c in normalized]
            key_schema_complete = bool(candidate) and len(hit_cols) == len(candidate)
            duplicate_candidate_keys = None
            if key_schema_complete and len(df):
                duplicate_candidate_keys = int(df.duplicated(hit_cols, keep=False).sum())
            records.append({"source": key, "path": str(path), "status": "readable", "rows": len(df),
                            "columns": " | ".join(map(str, df.columns)),
                            "year_min": years.min() if not years.empty else None,
                            "year_max": years.max() if not years.empty else None,
                            "candidate_key": " + ".join(hit_cols) if hit_cols else None,
                            "candidate_key_schema_complete": key_schema_complete,
                            "duplicate_candidate_keys": duplicate_candidate_keys})
        except Exception as exc:
            records.append({"source": key, "path": str(path), "status": f"read_error: {exc}", "rows": None,
                            "columns": None, "year_min": None, "year_max": None, "duplicate_candidate_keys": None})
    pd.DataFrame(records).to_csv(audit_dir / "input_schema_profile.csv", index=False)
    (audit_dir / "input_schema_profile.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(records)[["source", "status", "rows", "year_min", "year_max"]].to_string(index=False))
    print(f"\nSchema profile saved to {audit_dir / 'input_schema_profile.csv'}")


if __name__ == "__main__":
    main()
