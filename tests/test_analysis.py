import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import analyze_models


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

