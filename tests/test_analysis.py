import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import statsmodels.api as sm

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

    def test_supported_dyads_h3_uses_one_side_as_reference(self):
        df = make_analysis_sample(12)
        categories = [
            "none", "one_side_only", "both_sides", "unknown",
            "unresolved_side", "one_side_only", "both_sides",
            "one_side_only", "both_sides", "both_sides", "one_side_only", "both_sides",
        ]
        df["support_configuration_confirmed"] = categories
        original = df["support_configuration_confirmed"].copy()
        calls = {}

        def fake_fit(sample, features, outcome, model_name, hypothesis, extra_metric=None):
            if model_name in {"H3_support_configuration", "H3_supported_dyads"}:
                calls[model_name] = (sample.copy(), extra_metric)
            rows = [{
                "model": model_name, "hypothesis": hypothesis,
                "term": features[-1], "p_value": 0.5,
                "inference_status": "valid",
            }]
            return rows, {
                "model": model_name, "hypothesis": hypothesis, "status": "ok",
            }

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                analyze_models, "_fit_binary_specification", side_effect=fake_fit
            ):
                analyze_models.fit_support_hypotheses(df, Path(tmp))

        supported, metric = calls["H3_supported_dyads"]
        self.assertEqual(
            set(supported["support_configuration_confirmed"]),
            {"one_side_only", "both_sides"},
        )
        self.assertEqual(len(supported), 9)
        self.assertEqual(int(supported["war_t"].sum()), int(df.loc[
            df["support_configuration_confirmed"].isin(["one_side_only", "both_sides"]),
            "war_t",
        ].sum()))
        self.assertEqual(metric["h3_reference_category"], "one_side_only")
        self.assertEqual(metric["h3_supported_n"], 9)
        self.assertEqual(metric["h3_supported_one_side_n"], 4)
        self.assertEqual(metric["h3_supported_both_sides_n"], 5)
        pd.testing.assert_series_equal(
            original, df["support_configuration_confirmed"]
        )

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

    def test_adjusted_logistic_specs_share_common_sample_and_separate_controls(self):
        df = make_analysis_sample(30)
        df.loc[0, "population"] = np.nan
        df.loc[1, "cinc"] = np.nan
        df.loc[2, "gdp_per_capita"] = np.nan
        df.loc[3, "v2x_polyarchy"] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics = analyze_models.fit_logistic(df, output_dir)
            metric_frame = pd.DataFrame(metrics)
            adjusted = metric_frame[
                metric_frame["sample_specification"].eq("adjusted_alternatives_common")
            ]
            self.assertEqual(
                set(adjusted["model"]),
                {"M0_adjusted", "M1_adjusted", "M2_CINC", "M3_GDP",
                 "M4_VDEM", "M2_POP_ALT"},
            )
            self.assertEqual(adjusted["n"].nunique(), 1)
            self.assertEqual(adjusted["war_n"].nunique(), 1)
            self.assertTrue((adjusted["n"] == 26).all())
            self.assertTrue("control_spearman_correlations.csv" in {
                path.name for path in output_dir.iterdir()
            })
            vif = pd.read_csv(output_dir / "control_vif.csv")
            self.assertEqual(set(vif["model"]), {
                "M2_CINC", "M3_GDP", "M4_VDEM", "M2_POP_ALT"
            })
            self.assertNotIn("population", analyze_models.MODEL_FEATURES["M2_CINC"])
            self.assertNotIn("cinc", analyze_models.MODEL_FEATURES["M2_POP_ALT"])
            self.assertNotIn("v2x_polyarchy", analyze_models.MODEL_FEATURES["M3_GDP"])

    def test_h1_sensitivity_centers_breadth_and_excludes_one_conflict_each_time(self):
        df = make_analysis_sample(36)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics = analyze_models.fit_h1_sensitivity(df, output_dir)
            quadratic = pd.read_csv(output_dir / "h1_quadratic_coefficients.csv")
            conflict = pd.read_csv(output_dir / "h1_conflict_summary.csv")
            loo = pd.read_csv(output_dir / "h1_leave_one_conflict_out.csv")
            predictions = pd.read_csv(
                output_dir / "h1_functional_form_predictions.csv"
            )
        self.assertAlmostEqual(metrics[0]["breadth_centered_mean"], 0.0)
        expected_sq_mean = (
            (df["n_support_types_confirmed"]
             - df["n_support_types_confirmed"].mean()) ** 2
        ).mean()
        self.assertAlmostEqual(
            metrics[0]["breadth_centered_sq_mean"], expected_sq_mean
        )
        self.assertIn("breadth_centered_sq", set(quadratic["term"]))
        self.assertEqual(len(conflict), df["conflict_id"].nunique())
        self.assertEqual(len(loo), df["conflict_id"].nunique())
        self.assertEqual(
            set(loo["excluded_conflict_id"]),
            set(df["conflict_id"].unique()),
        )
        self.assertTrue((loo["unique_conflicts"] == df["conflict_id"].nunique() - 1).all())
        self.assertEqual({row["sensitivity_role"] for row in metrics},
                         {"exploratory_functional_form", "exploratory_conflict_sensitivity"})
        self.assertTrue(predictions.empty or {
            "breadth", "predicted_probability"
        } <= set(predictions))

    def test_primary_m1_definition_remains_linear_breadth(self):
        self.assertEqual(
            analyze_models.MODEL_FEATURES["M1"][-1],
            "n_support_types_confirmed",
        )
        self.assertNotIn("breadth_centered", analyze_models.MODEL_FEATURES["M1"])
        self.assertNotIn("breadth_centered_sq", analyze_models.MODEL_FEATURES["M1"])

    def test_h1_leave_one_conflict_out_records_failed_fits(self):
        df = make_analysis_sample(18)

        def failed_fit(sample, features, outcome, model_name, hypothesis,
                       extra_metric=None):
            if model_name == "H1_leave_one_conflict_out":
                return [], {
                    "model": model_name, "hypothesis": hypothesis,
                    "status": "non_estimable", "converged": False,
                    "inference_error": "synthetic_failure",
                }
            return [], {
                "model": model_name, "hypothesis": hypothesis,
                "status": "non_estimable", "converged": False,
                "inference_error": "synthetic_failure",
            }

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                analyze_models, "_fit_binary_specification",
                side_effect=failed_fit,
            ):
                analyze_models.fit_h1_sensitivity(df, Path(tmp))
            loo = pd.read_csv(Path(tmp) / "h1_leave_one_conflict_out.csv")
        self.assertEqual(len(loo), df["conflict_id"].nunique())
        self.assertTrue((loo["status"] == "non_estimable").all())
        self.assertTrue((loo["inference_error"] == "synthetic_failure").all())

    def test_h2_fdr_is_only_added_to_support_type_terms(self):
        df = make_analysis_sample(30)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                analyze_models, "_fit_binary_specification",
                side_effect=lambda *args, **kwargs: (
                    [{"model": args[3], "hypothesis": args[4], "term": "const", "p_value": 0.5,
                      "inference_status": "valid"},
                     {"model": args[3], "hypothesis": args[4], "term": args[1][-1], "p_value": 0.01,
                      "inference_status": "valid"},
                     {"model": args[3], "hypothesis": args[4], "term": "year", "p_value": 0.5,
                      "inference_status": "valid"}],
                    {"model": args[3], "hypothesis": args[4], "status": "ok"},
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

    def test_h2_nonconverged_model_is_non_estimable_with_nan_inference(self):
        class FakeResult:
            mle_retvals = {"converged": False}
            params = pd.Series([1.0], index=["const"])
            bse = pd.Series([0.1], index=["const"])
            pvalues = pd.Series([0.1], index=["const"])

            def conf_int(self):
                return pd.DataFrame({0: [-1.0], 1: [1.0]}, index=["const"])

        fake_logit = mock.Mock()
        fake_logit.fit.return_value = FakeResult()
        with mock.patch.object(sm, "Logit", return_value=fake_logit):
            rows, metric = analyze_models._fit_binary_specification(
                make_analysis_sample(30),
                analyze_models.MODEL_FEATURES["M0"],
                "war_t", "H2_ext_x", "support_type",
            )
        self.assertEqual(metric["status"], "non_estimable")
        self.assertEqual(metric["inference_error"], "maximum_likelihood_not_converged")
        self.assertTrue(pd.isna(rows[0]["odds_ratio"]))
        self.assertTrue(pd.isna(rows[0]["p_value"]))
        self.assertEqual(rows[0]["inference_status"], "non_estimable")

    def test_h2_nonfinite_standard_errors_are_non_estimable(self):
        class FakeResult:
            mle_retvals = {"converged": True}
            params = pd.Series([1.0], index=["const"])
            bse = pd.Series([float("nan")], index=["const"])
            pvalues = pd.Series([0.1], index=["const"])

            def conf_int(self):
                return pd.DataFrame({0: [-1.0], 1: [1.0]}, index=["const"])

        fake_logit = mock.Mock()
        fake_logit.fit.return_value = FakeResult()
        with mock.patch.object(sm, "Logit", return_value=fake_logit):
            rows, metric = analyze_models._fit_binary_specification(
                make_analysis_sample(30),
                analyze_models.MODEL_FEATURES["M0"],
                "war_t", "H2_ext_x", "support_type",
            )
        self.assertEqual(metric["status"], "non_estimable")
        self.assertEqual(metric["inference_error"], "non_finite_standard_errors")
        self.assertTrue(pd.isna(rows[0]["or_ci_lower"]))
        self.assertTrue(pd.isna(rows[0]["p_value"]))

    def test_h2_fdr_excludes_non_estimable_model_and_reports_counts(self):
        df = make_analysis_sample(30)

        def fake_fit(sample, features, outcome, model_name, hypothesis, extra_metric=None):
            target = features[-1]
            failed = model_name == "H2_ext_x"
            rows = [
                {
                    "model": model_name, "hypothesis": hypothesis, "term": target,
                    "p_value": np.nan if failed else 0.01,
                    "inference_status": "non_estimable" if failed else "valid",
                    "odds_ratio": np.nan if failed else 1.5,
                }
            ]
            metric = {
                "model": model_name, "hypothesis": hypothesis,
                "status": "non_estimable" if failed else "ok",
                "converged": not failed,
            }
            return rows, metric

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                analyze_models, "_fit_binary_specification", side_effect=fake_fit
            ):
                analyze_models.fit_support_hypotheses(df, Path(tmp))
            coefficients = pd.read_csv(Path(tmp) / "support_hypothesis_coefficients.csv")
            metrics = pd.read_csv(Path(tmp) / "support_hypothesis_metrics.csv")
        h2 = coefficients[coefficients["hypothesis"].eq("support_type")]
        self.assertTrue(pd.isna(h2.loc[h2["model"].eq("H2_ext_x"), "p_value_fdr_bh"]).all())
        self.assertEqual(h2["p_value_fdr_bh"].notna().sum(), 9)
        h2_metrics = metrics[metrics["hypothesis"].eq("support_type")]
        self.assertTrue((h2_metrics["h2_attempted"] == 10).all())
        self.assertTrue((h2_metrics["h2_valid"] == 9).all())
        self.assertTrue((h2_metrics["h2_non_estimable"] == 1).all())
        self.assertTrue((h2_metrics["h2_fdr_family_size"] == 9).all())

    def test_grouped_cv_writes_distinct_seed_and_fold_outputs(self):
        sample = make_analysis_sample(60)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with mock.patch.object(
                analyze_models, "temporal_cross_validation",
                side_effect=AssertionError("temporal validation must be optional"),
            ):
                metrics, _, _ = analyze_models.fit_predictive_models(
                    sample, output_dir, test_start_year=2090, random_state=7,
                    cv_splits=3, split_seeds=[7]
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

    def test_repeated_grouped_cv_records_reproducible_split_and_conflict_diagnostics(self):
        sample = make_analysis_sample(60)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            analyze_models.fit_predictive_models(
                sample, output_dir, test_start_year=2090, random_state=7,
                cv_splits=3, split_seeds=[7, 17],
            )
            seed = pd.read_csv(output_dir / "grouped_cv_seed_metrics.csv")
            folds = pd.read_csv(output_dir / "grouped_cv_fold_metrics.csv")
            repetitions = pd.read_csv(
                output_dir / "grouped_cv_repetition_metrics.csv"
            )
            overlap = pd.read_csv(
                output_dir / "grouped_cv_conflict_overlap.csv"
            )
            predictions = pd.read_csv(output_dir / "grouped_cv_predictions.csv")
            shap_values = pd.read_csv(output_dir / "xgboost_shap_values.csv")
            feature_values = pd.read_csv(
                output_dir / "xgboost_shap_feature_values.csv"
            )
            importance = pd.read_csv(
                output_dir / "xgboost_shap_importance.csv"
            )

        self.assertEqual(set(seed["split_seed"]), {7, 17})
        self.assertEqual(set(repetitions["split_seed"]), {7, 17})
        self.assertEqual(len(overlap), 2 * 3)
        self.assertTrue((overlap["shared_conflicts"] >= 0).all())
        self.assertTrue(
            (folds["valid_estimator_seeds"] == len([7, 108, 209])).all()
        )
        self.assertEqual(
            len(folds),
            2 * 3 * 2 * len(analyze_models.PREDICTIVE_MODELS),
        )
        self.assertEqual(
            len(repetitions),
            2 * 2 * len(analyze_models.PREDICTIVE_MODELS),
        )
        self.assertTrue(
            {"split_repetition", "split_seed", "fold", "predicted_probability"}
            <= set(predictions)
        )
        self.assertEqual(shap_values.shape, feature_values.shape)
        self.assertEqual(
            list(shap_values.columns), list(feature_values.columns)
        )
        self.assertEqual(
            set(zip(shap_values["dyad_id"], shap_values["observation_year"])),
            set(zip(feature_values["dyad_id"], feature_values["observation_year"])),
        )
        self.assertEqual(len(shap_values), len(sample))
        expected_importance = dict(zip(
            analyze_models.PREDICTIVE_SPECS["extended"],
            np.abs(
                shap_values[analyze_models.PREDICTIVE_SPECS["extended"]]
                .to_numpy()
            ).mean(axis=0),
        ))
        self.assertEqual(set(importance["feature"]), set(expected_importance))
        for row in importance.itertuples():
            self.assertAlmostEqual(
                row.mean_abs_shap, expected_importance[row.feature], places=6
            )
        for split_seed in [7, 17]:
            split = folds[folds["split_seed"] == split_seed]
            for fold_id in split["fold"].unique():
                fold = split[split["fold"] == fold_id]
                self.assertEqual(
                    set(fold["specification"]), {"baseline", "extended"}
                )

    def test_grouped_cv_split_assignments_are_reproducible(self):
        sample = make_analysis_sample(60)
        outputs = []
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (first, second):
                analyze_models.grouped_cross_validation(
                    sample, Path(directory), random_state=7, n_splits=3,
                    split_seeds=[11, 23],
                )
                outputs.append(pd.read_csv(Path(directory) / "grouped_cv_conflict_overlap.csv"))
        pd.testing.assert_frame_equal(outputs[0], outputs[1])

    def test_shap_and_feature_value_matrices_preserve_keys_and_feature_order(self):
        features = analyze_models.PREDICTIVE_SPECS["extended"]
        metadata = pd.DataFrame({
            "fold": [1, 2],
            "dyad_id": [10, 11],
            "observation_year": [2000, 2001],
        })
        values = pd.DataFrame(np.arange(2 * len(features)).reshape(2, -1), columns=features)
        attributions = pd.concat([metadata, values], axis=1)
        original = pd.concat([metadata, values + 100], axis=1)
        analyze_models._validate_shap_alignment(attributions, original, features)
        pd.testing.assert_frame_equal(
            attributions[["dyad_id", "observation_year"]],
            original[["dyad_id", "observation_year"]],
        )
        self.assertEqual(
            attributions[features].shape,
            original[features].shape,
        )

    def test_shap_alignment_rejects_duplicate_observation_keys(self):
        features = analyze_models.PREDICTIVE_SPECS["extended"]
        frame = pd.DataFrame({
            "fold": [1, 2],
            "dyad_id": [10, 10],
            "observation_year": [2000, 2000],
        })
        for feature in features:
            frame[feature] = [0.0, 1.0]
        with self.assertRaises(ValueError):
            analyze_models._validate_shap_alignment(frame, frame.copy(), features)

    def test_shap_plot_receives_original_feature_values(self):
        features = analyze_models.PREDICTIVE_SPECS["extended"]
        shap_values = np.ones((2, len(features)))
        original = pd.DataFrame(np.arange(2 * len(features)).reshape(2, -1), columns=features)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(analyze_models.shap, "summary_plot") as summary_plot:
                analyze_models.save_plots(
                    Path(tmp), pd.DataFrame({"term": []}),
                    pd.DataFrame({"method": []}),
                    make_analysis_sample(4),
                    shap_values, original,
                )
        summary_plot.assert_called_once()
        args, kwargs = summary_plot.call_args
        np.testing.assert_array_equal(args[0], shap_values)
        pd.testing.assert_frame_equal(args[1], original)
        self.assertEqual(kwargs["max_display"], len(features))
