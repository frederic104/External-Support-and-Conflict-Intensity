import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import analyze_models
from src.pipeline import build_esd_features, dyadic_schema, esd_schema


TYPE_VARS = [
    "ext_x", "ext_p", "ext_y", "ext_w", "ext_m", "ext_t",
    "ext_f", "ext_i", "ext_l", "ext_o", "ext_u",
]


def make_analysis_sample(rows=60):
    data = []
    for index in range(rows):
        x = index % 5
        row = {
            "war_t": int((index * 3 + x) % 7 in {0, 1}),
            "year": 1976 + index % 20,
            "dyad_id": index % 12,
            "conflict_id": 100 + index % 6,
            "lag_conflict_id": 100 + index % 6,
            "lag_intensity": float(x),
            "lag_conflict_type": float(index % 2),
            "lag_episode_age_years": float(1 + index % 8),
            "n_support_types_confirmed": float(index % 3),
            "cinc": 0.1 + (index % 9) / 100,
            "population": 1_000_000 + index * 10_000,
            "gdp_per_capita": 2_000 + index * 100,
            "v2x_polyarchy": 0.2 + (index % 7) / 10,
            "support_configuration_confirmed": "none",
        }
        row.update({feature: float(index % 2) for feature in analyze_models.SUPPORT_TYPE_FEATURES})
        data.append(row)
    return pd.DataFrame(data)


class TestAnalysisPolicy(unittest.TestCase):
    def test_load_panel_does_not_require_unused_supporter_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.csv"
            pd.DataFrame({
                "war_t": [0, 1],
                "year": [2000, 2001],
                "dyad_id": [1, 1],
                "n_support_types_confirmed": [1, 2],
                "support_configuration_confirmed": ["none", "one_side_only"],
            }).to_csv(path, index=False)
            loaded = analyze_models.load_panel(path)
            self.assertNotIn("log_supporters_confirmed", loaded)

    def test_h3_excludes_unknown_and_unresolved_side_from_none_reference(self):
        baseline = analyze_models.MODEL_FEATURES["M0"]
        rows = []
        for index, category in enumerate(
            ["none", "one_side_only", "both_sides", "unknown", "unresolved_side"]
        ):
            row = {
                "war_t": index % 2,
                "minor_to_war": index % 2,
                "dyad_id": index,
                "support_configuration_confirmed": category,
                "n_support_types_confirmed": 1,
            }
            row.update({feature: 0 for feature in baseline})
            row.update({feature: 0 for feature in analyze_models.SUPPORT_TYPE_FEATURES})
            rows.append(row)
        captured = {}

        def fake_fit(sample, features, outcome, model_name, hypothesis, extra_metric=None):
            if model_name == "H3_support_configuration":
                captured["sample"] = sample.copy()
                captured["metric"] = extra_metric
            coefficient = [{
                "model": model_name,
                "hypothesis": hypothesis,
                "term": features[-1],
                "p_value": 0.05,
            }]
            return coefficient, {"model": model_name, "hypothesis": hypothesis}

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                analyze_models, "_fit_binary_specification", side_effect=fake_fit
            ):
                analyze_models.fit_support_hypotheses(pd.DataFrame(rows), Path(tmp))

        self.assertEqual(
            set(captured["sample"]["support_configuration_confirmed"]),
            {"none", "one_side_only", "both_sides"},
        )
        self.assertEqual(captured["metric"]["h3_reference_category"], "none")
        self.assertEqual(captured["metric"]["h3_excluded_invalid_categories"], 2)

    def test_confirmed_breadth_excludes_ext_u_and_keeps_ext_x_ext_p_distinct(self):
        dy_raw = pd.DataFrame({
            "DyadID": [1], "ConflictID": [100], "Year": [2000], "Int": [1],
            "Type": [4], "CountryA": ["Lebanon"], "CountryB": [""],
            "Location": ["Lebanon"], "GWNoA": [660], "SideAID": [10],
            "SideBID": [20],
        })
        dy = dyadic_schema(dy_raw)
        ty_row = {
            "DyadID": 1, "Year": 2000, "active": 1, "actor_id": 10,
            "oppo_id": 20, "ext_id": 501, "ext_sup": 1, "ext_alleged": 0,
            "ext_nonstate": 0, "ext_coalition": 0, "ext_coalition_name": "",
            "ext_elements": 0, "ext_bothsides": 0, "ext_count": 1, "ext_sum": 0,
        }
        ty_row.update({name: int(name in {"ext_x", "ext_p", "ext_u"}) for name in TYPE_VARS})
        ty = esd_schema(pd.DataFrame([ty_row]), "test TY", True)
        dy_esd = esd_schema(pd.DataFrame({
            "DyadID": [1], "Year": [2000], "ext_sup": [1], "active": [1],
        }), "test DY", False)
        with tempfile.TemporaryDirectory() as tmp:
            features = build_esd_features(ty, dy_esd, dy, Path(tmp), TYPE_VARS)
        row = features.iloc[0]
        self.assertEqual(row["ext_x_confirmed"], 1)
        self.assertEqual(row["ext_p_confirmed"], 1)
        self.assertEqual(row["ext_u_confirmed"], 1)
        self.assertEqual(row["n_support_types_confirmed"], 2)

    def test_logistic_outputs_or_ci_and_inference_metrics_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics = analyze_models.fit_logistic(make_analysis_sample(), output_dir)
            coefficients = pd.read_csv(output_dir / "logistic_coefficients.csv")
            self.assertTrue({"or_ci_lower", "or_ci_upper", "ci_status"} <= set(coefficients))
            self.assertTrue(metrics)
            self.assertFalse(any("accuracy" in row or "roc_auc" in row for row in metrics))
            self.assertIn("war_n", metrics[0])
            self.assertIn("minor_n", metrics[0])

    def test_h2_fdr_is_only_added_to_support_type_terms(self):
        df = make_analysis_sample(30)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                analyze_models, "_fit_binary_specification",
                side_effect=lambda *args, **kwargs: (
                    [{"model": args[3], "hypothesis": args[4], "term": "const", "p_value": 0.5},
                     {"model": args[3], "hypothesis": args[4], "term": args[1][-1], "p_value": 0.01},
                     {"model": args[3], "hypothesis": args[4], "term": "year", "p_value": 0.5}],
                    {"model": args[3], "hypothesis": args[4]},
                ),
            ):
                analyze_models.fit_support_hypotheses(df, Path(tmp))
            coefficients = pd.read_csv(Path(tmp) / "support_hypothesis_coefficients.csv")
        corrected = coefficients[
            coefficients["hypothesis"].eq("support_type")
            & coefficients["term"].isin(analyze_models.SUPPORT_TYPE_FEATURES)
        ]
        self.assertTrue(corrected["p_value_fdr_bh"].notna().all())
        non_h2 = coefficients[~coefficients["hypothesis"].eq("support_type")]
        self.assertTrue(non_h2["p_value_fdr_bh"].isna().all())
        self.assertTrue(coefficients.loc[coefficients["term"].isin(["const", "year"]), "p_value_fdr_bh"].isna().all())

    def test_grouped_cv_writes_distinct_seed_and_fold_outputs(self):
        sample = make_analysis_sample(60)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with mock.patch.object(
                analyze_models, "temporal_cross_validation",
                side_effect=AssertionError("temporal validation must be optional"),
            ):
                metrics, _, _ = analyze_models.fit_predictive_models(
                    sample, output_dir, test_start_year=2090, random_state=7, cv_splits=3
                )
            seed = pd.read_csv(output_dir / "grouped_cv_seed_metrics.csv")
            folds = pd.read_csv(output_dir / "grouped_cv_fold_metrics.csv")
            self.assertEqual(set(folds["specification"]), {"baseline", "extended"})
            self.assertEqual(set(seed["repetition"]), {1, 2, 3})
            self.assertTrue({"test_n_war", "test_n_minor", "test_war_prevalence"} <= set(folds))
            self.assertEqual(len(folds), 3 * 2 * len(analyze_models.PREDICTIVE_MODELS))
            for model in analyze_models.PREDICTIVE_MODELS:
                baseline_folds = set(folds.loc[
                    (folds["specification"] == "baseline") & (folds["model"] == model), "fold"
                ])
                extended_folds = set(folds.loc[
                    (folds["specification"] == "extended") & (folds["model"] == model), "fold"
                ])
                self.assertEqual(baseline_folds, extended_folds)
            self.assertTrue(metrics)
