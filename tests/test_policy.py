"""Small synthetic tests for critical integration policies. Run: python -m unittest discover -s tests"""
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import pandas as pd
from src.pipeline import (esd_schema, dyadic_schema, build_esd_features, get_dyadic_region,
                          build_consecutive_panel, brd_schema, tidy_country_source,
                          validate_core_source_set)

TYPE_VARS = ["ext_x", "ext_p", "ext_y", "ext_w", "ext_m", "ext_t", "ext_f", "ext_i", "ext_l", "ext_o", "ext_u"]


def make_ty_row(dyad, year, actor, opponent, supporter, support=1, alleged=0, weapon=0, coalition=0, coalition_name=""):
    row = {"DyadID": dyad, "Year": year, "active": 1, "actor_id": actor, "oppo_id": opponent,
           "ext_id": supporter, "ext_sup": support, "ext_alleged": alleged,
           "ext_nonstate": 0, "ext_coalition": coalition, "ext_coalition_name": coalition_name,
           "ext_elements": 0, "ext_bothsides": 0, "ext_count": 1, "ext_sum": int(weapon)}
    for v in TYPE_VARS:
        row[v] = int(v == "ext_w" and bool(weapon))
    return row


class TestIntegrationPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audit = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_both_sides_means_any_support_to_each_side(self):
        dy_raw = pd.DataFrame({
            "DyadID": ["1"], "ConflictID": [100], "Year": [2000], "Int": [1], "Type": [4],
            "CountryA": ["Lebanon"], "CountryB": [""], "Location": ["Lebanon"],
            "GWNoA": [660], "SideAID": [10], "SideBID": [20]
        })
        dy = dyadic_schema(dy_raw)
        ty_raw = pd.DataFrame([
            make_ty_row(1, 2000, 10, 20, 501, weapon=1),
            make_ty_row(1, 2000, 20, 10, 502, weapon=0),
        ])
        dy_esd_raw = pd.DataFrame({"DyadID": [1], "Year": [2000], "ext_sup": [1], "active": [1]})
        ty = esd_schema(ty_raw, "test TY", True)
        dy_esd = esd_schema(dy_esd_raw, "test DY", False)
        f = build_esd_features(ty, dy_esd, dy, self.audit, TYPE_VARS)
        row = f.loc[(f.dyad_id == "1") & (f.year == 2000)].iloc[0]
        self.assertEqual(row["support_configuration_all"], "both_sides")
        # ESD ext_bothsides is 0 for both rows in this test; it is not used as the H3 variable.
        self.assertEqual(row["any_support_all"], 1)

    def test_explicit_no_support_authorizes_zero_filling_detail_blanks(self):
        dy_raw = pd.DataFrame({
            "DyadID": ["1"], "ConflictID": [100], "Year": [2000], "Int": [1], "Type": [4],
            "CountryA": ["Lebanon"], "CountryB": [""], "Location": ["Lebanon"],
            "GWNoA": [660], "SideAID": [10], "SideBID": [20]
        })
        dy = dyadic_schema(dy_raw)
        ty = esd_schema(pd.DataFrame(columns=["DyadID", "Year", "active", "actor_id", "ext_id", "ext_sup", "ext_alleged"] + TYPE_VARS), "test TY", True)
        dy_esd = esd_schema(pd.DataFrame({"DyadID": [1], "Year": [2000], "ext_sup": [0], "active": [1]}), "test DY", False)
        f = build_esd_features(ty, dy_esd, dy, self.audit, TYPE_VARS)
        row = f.iloc[0]
        self.assertEqual(row["any_support_all"], 0)
        self.assertEqual(row["ext_w_confirmed"], 0)
        self.assertEqual(row["n_supporters_all_raw"], 0)
        self.assertEqual(row["support_configuration_all"], "none")

    def test_exact_previous_year_pairing_does_not_bridge_a_gap(self):
        dy_raw = pd.DataFrame({
            "DyadID": [1, 1], "ConflictID": [100, 100], "Year": [2000, 2002], "Int": [1, 2], "Type": [4, 4],
            "CountryA": ["Lebanon", "Lebanon"], "CountryB": ["", ""], "Location": ["Lebanon", "Lebanon"],
            "GWNoA": [660, 660], "SideAID": [10, 10], "SideBID": [20, 20]
        })
        dy = dyadic_schema(dy_raw)
        cfg = {"project": {"country_aliases": {}, "region_country_labels": ["Lebanon"],
                           "main_conflict_types": [3, 4], "use_location_fallback_for_region": False,
                           "outcome_year_start": 2001, "outcome_year_end": 2002}}
        reg = get_dyadic_region(dy, cfg, self.audit)
        with self.assertRaisesRegex(ValueError, "No consecutive active dyad-year pairs"):
            build_consecutive_panel(reg, cfg, self.audit)

    def test_supported_type_missing_fails_closed(self):
        dy_raw = pd.DataFrame({
            "DyadID": [1], "ConflictID": [100], "Year": [2000], "Int": [1], "Type": [4],
            "CountryA": ["Lebanon"], "CountryB": [""], "Location": ["Lebanon"],
            "GWNoA": [660], "SideAID": [10], "SideBID": [20]
        })
        dy = dyadic_schema(dy_raw)
        row = make_ty_row(1, 2000, 10, 20, 501, weapon=1)
        row["ext_w"] = None
        ty = esd_schema(pd.DataFrame([row]), "test TY", True)
        dy_esd = esd_schema(pd.DataFrame({"DyadID": [1], "Year": [2000], "ext_sup": [1], "active": [1]}), "test DY", False)
        with self.assertRaisesRegex(ValueError, "missing/non-binary ext_w"):
            build_esd_features(ty, dy_esd, dy, self.audit, TYPE_VARS)
        self.assertTrue((self.audit / "esd_ty_invalid_ext_w.csv").exists())

    def test_missing_supporter_id_does_not_create_partial_count(self):
        dy_raw = pd.DataFrame({
            "DyadID": [1], "ConflictID": [100], "Year": [2000], "Int": [1], "Type": [4],
            "CountryA": ["Lebanon"], "CountryB": [""], "Location": ["Lebanon"],
            "GWNoA": [660], "SideAID": [10], "SideBID": [20]
        })
        dy = dyadic_schema(dy_raw)
        ty_raw = pd.DataFrame([make_ty_row(1, 2000, 10, 20, None, weapon=1)])
        ty = esd_schema(ty_raw, "test TY", True)
        dy_esd = esd_schema(pd.DataFrame({"DyadID": [1], "Year": [2000], "ext_sup": [1], "active": [1]}), "test DY", False)
        f = build_esd_features(ty, dy_esd, dy, self.audit, TYPE_VARS)
        self.assertTrue(pd.isna(f.iloc[0]["n_supporters_all_raw"]))
        self.assertTrue((self.audit / "esd_ty_missing_supporter_id.csv").exists())

    def test_brd_missing_deaths_remain_missing(self):
        brd = brd_schema(pd.DataFrame({"DyadID": [1, 2], "Year": [1990, 1991], "bd_best": [45, None]}))
        self.assertEqual(brd.loc[0, "battle_deaths_best"], 45)
        self.assertTrue(pd.isna(brd.loc[1, "battle_deaths_best"]))

    def test_wdi_bulk_csv_is_reshaped_without_manual_staging(self):
        raw = pd.DataFrame({
            "Country Name": ["Korea, Rep.", "Korea, Rep.", "Korea, Rep."],
            "Country Code": ["KOR", "KOR", "KOR"],
            "Indicator Name": ["Population", "GDP per capita", "Unused"],
            "Indicator Code": ["SP.POP.TOTL", "NY.GDP.PCAP.KD", "NY.GDP.MKTP.CD"],
            "2016": [51014947, 30000, 1],
            "2017": [51361911, 31000, 2],
        })
        path = Path(self.tmp.name) / "WDICSV.csv"
        raw.to_csv(path, index=False)
        out = tidy_country_source(path, "wdi")
        row = out.loc[(out["iso3"] == "KOR") & (out["year"] == 2017)].iloc[0]
        self.assertEqual(row["population"], 51361911)
        self.assertEqual(row["gdp_per_capita"], 31000)

    def test_preflight_rejects_mixed_ucdp_id_releases_before_joining(self):
        for name in ["dyadic.csv", "esd_ty.csv", "esd_dy.csv"]:
            (self.audit / name).write_text("placeholder\n", encoding="utf-8")
        cfg = {
            "files": {"dyadic": "unused", "esd_ty": "unused", "esd_dy": "unused"},
            "ucdp_versions": {"dyadic": "26.1", "esd_ty": "18.1", "esd_dy": "18.1"},
        }
        # The function resolves source files from its project root.  A missing
        # source is already a safe failure; this test exercises the version
        # gate directly by using filenames that exist at the project root only
        # through a patched ROOT-independent configuration below.
        with mock.patch("src.pipeline.ROOT", self.audit):
            cfg["files"] = {"dyadic": "dyadic.csv", "esd_ty": "esd_ty.csv", "esd_dy": "esd_dy.csv"}
            with self.assertRaisesRegex(ValueError, "must use project.ucdp_version"):
                validate_core_source_set(cfg, self.audit)
        self.assertTrue((self.audit / "core_source_preflight.csv").exists())

    def test_preflight_rejects_release_not_matching_project_version(self):
        for name in ["dyadic.csv", "esd_ty.csv", "esd_dy.csv"]:
            (self.audit / name).write_text("placeholder\n", encoding="utf-8")
        cfg = {
            "project": {"ucdp_version": "18.1"},
            "files": {"dyadic": "dyadic.csv", "esd_ty": "esd_ty.csv", "esd_dy": "esd_dy.csv"},
            "ucdp_versions": {"dyadic": "26.1", "esd_ty": "26.1", "esd_dy": "26.1"},
        }
        with mock.patch("src.pipeline.ROOT", self.audit):
            with self.assertRaisesRegex(ValueError, "project.ucdp_version=18.1"):
                validate_core_source_set(cfg, self.audit)

    def test_conflict_id_is_preserved_in_current_and_lagged_panel(self):
        dy_raw = pd.DataFrame({
            "DyadID": [1, 1], "ConflictID": [100, 100], "Year": [2000, 2001],
            "Int": [1, 2], "Type": [4, 4],
            "CountryA": ["Lebanon", "Lebanon"], "CountryB": ["", ""],
            "Location": ["Lebanon", "Lebanon"], "GWNoA": [660, 660],
            "SideAID": [10, 10], "SideBID": [20, 20],
        })
        dy = dyadic_schema(dy_raw)
        cfg = {"project": {"country_aliases": {}, "region_country_labels": ["Lebanon"],
                           "main_conflict_types": [3, 4], "use_location_fallback_for_region": False,
                           "outcome_year_start": 2001, "outcome_year_end": 2001}}
        reg = get_dyadic_region(dy, cfg, self.audit)
        panel = build_consecutive_panel(reg, cfg, self.audit)
        self.assertEqual(panel.iloc[0]["conflict_id"], 100)
        self.assertEqual(panel.iloc[0]["lag_conflict_id"], 100)

    def test_missing_conflict_id_is_schema_error(self):
        raw = pd.DataFrame({
            "DyadID": [1], "Year": [2000], "Int": [1], "Type": [4],
            "CountryA": ["Lebanon"], "SideAID": [10], "SideBID": [20],
        })
        with self.assertRaisesRegex(ValueError, "required canonical columns not found.*conflict_id"):
            dyadic_schema(raw)



if __name__ == "__main__":
    unittest.main()
