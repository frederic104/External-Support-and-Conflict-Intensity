"""Fit the planned inferential and predictive models on the integrated panel."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).resolve().parents[1]

MODEL_FEATURES: dict[str, list[str]] = {
    "M0": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year"],
    "M1": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "n_support_types_confirmed"],
    "M2_CINC": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
                "n_support_types_confirmed", "cinc"],
    "M3_GDP": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
               "n_support_types_confirmed", "gdp_per_capita"],
    "M4_VDEM": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
                "n_support_types_confirmed", "v2x_polyarchy"],
    "M2_POP_ALT": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
                   "n_support_types_confirmed", "log_population"],
}
ADJUSTED_SAMPLE_FEATURES = [
    "lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
    "n_support_types_confirmed", "cinc", "population", "gdp_per_capita",
    "v2x_polyarchy", "log_population",
]
SUPPORT_TYPE_FEATURES = [
    "ext_x_confirmed", "ext_p_confirmed", "ext_y_confirmed", "ext_w_confirmed",
    "ext_m_confirmed", "ext_t_confirmed", "ext_f_confirmed", "ext_i_confirmed",
    "ext_l_confirmed", "ext_o_confirmed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/model_dataset.csv")
    parser.add_argument("--output-dir", default="reports/model_analysis")
    parser.add_argument("--test-start-year", type=int, default=2009)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--cv-split-seeds", type=int, nargs="+", default=None,
        help="Random states for repeated grouped-CV split sensitivity checks.",
    )
    parser.add_argument(
        "--temporal-validation", action="store_true",
        help="Run forward-looking temporal validation as a supplementary analysis.",
    )
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model dataset not found: {path}. Run src/pipeline.py first.")
    df = pd.read_csv(path)
    required = {"war_t", "year", "dyad_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Model dataset is missing required columns: {missing}")
    required_support = {"n_support_types_confirmed", "support_configuration_confirmed"}
    missing_support = sorted(required_support - set(df.columns))
    if missing_support:
        raise ValueError(f"Model dataset is missing support-analysis columns: {missing_support}")
    return df


def complete_case(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"Model dataset is missing model columns: {missing}")
    metadata = ["war_t", "year", "dyad_id", "conflict_id", "lag_conflict_id"]
    columns = list(dict.fromkeys([*metadata, *features]))
    columns = [column for column in columns if column in df.columns]
    out = df[columns].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out["war_t"].nunique() < 2:
        raise ValueError("A model sample must contain both war_t classes.")
    return out


def save_plots(output_dir: Path, coefficients: pd.DataFrame,
               metrics: pd.DataFrame, df: pd.DataFrame,
               shap_values: np.ndarray | None = None,
               shap_features: pd.DataFrame | None = None) -> None:
    """Write publication-style vector and raster figures from saved summaries."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })

    logit = coefficients[coefficients["term"].ne("const")].copy()
    if not logit.empty:
        logit["label"] = logit["term"].str.replace("_", " ", regex=False)
        logit["lo"] = np.exp(logit["estimate"] - 1.96 * logit["std_error"])
        logit["hi"] = np.exp(logit["estimate"] + 1.96 * logit["std_error"])
        models = list(dict.fromkeys(logit["model"]))
        fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 5),
                                 sharey=True, squeeze=False)
        for ax, model in zip(axes[0], models):
            part = logit[logit["model"].eq(model)].sort_values("odds_ratio")
            y = np.arange(len(part))
            ax.errorbar(part["odds_ratio"], y,
                        xerr=[part["odds_ratio"] - part["lo"],
                              part["hi"] - part["odds_ratio"]],
                        fmt="o", color="#1f4e79", ecolor="#6b7280", capsize=3)
            ax.axvline(1, color="#9ca3af", linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_title(model)
            ax.set_xlabel("Odds ratio (95% CI)")
            ax.set_yticks(y)
            ax.set_yticklabels(part["label"])
        fig.suptitle("Logistic regression estimates", y=1.02, fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "logistic_odds_ratios.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "logistic_odds_ratios.png", bbox_inches="tight")
        plt.close(fig)

    predictive = metrics[metrics["method"].eq("predictive_group_cv")].copy()
    if predictive.empty:
        predictive = metrics[metrics["method"].eq("predictive_temporal_cv")].copy()
    if not predictive.empty:
        plot_cols = [c for c in ["roc_auc_mean", "average_precision_mean", "brier_mean"]
                     if c in predictive]
        if not plot_cols:
            plot_cols = ["test_roc_auc", "test_accuracy", "test_brier"]
        long = predictive.melt(id_vars="model", value_vars=plot_cols,
                               var_name="metric", value_name="value").dropna()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for metric, part in long.groupby("metric", sort=False):
            label = metric.replace("_mean", "").replace("test_", "").replace("_", " ")
            ax.plot(part["model"].str.replace("_", " ").str.title(), part["value"],
                    marker="o", linewidth=2, label=label)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Predictive performance across dyad-grouped CV folds")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "predictive_performance.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "predictive_performance.png", bbox_inches="tight")
        plt.close(fig)

    configuration = df["support_configuration_confirmed"].value_counts()
    rates = df.groupby("support_configuration_confirmed", observed=False)["war_t"].agg(
        war_rate="mean", n="size"
    ).reindex(["none", "one_side_only", "both_sides"]).dropna()
    if not rates.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        labels = {"none": "None", "one_side_only": "One side", "both_sides": "Both sides"}
        xlabels = [labels.get(x, x) for x in rates.index]
        axes[0].bar(xlabels, rates["war_rate"], color=["#adb5bd", "#f4a261", "#2a9d8f"])
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Share reaching war level")
        axes[0].set_title("Support configuration")
        for i, (rate, n) in enumerate(zip(rates["war_rate"], rates["n"])):
            axes[0].text(i, min(rate + 0.04, 0.97), f"{rate:.2f}\n(n={int(n)})",
                         ha="center", fontsize=9)
        grouped = df.assign(type_group=pd.cut(
            df["n_support_types_confirmed"], bins=[-1, 2, 4, np.inf],
            labels=["0–2 types", "3–4 types", "5+ types"]
        )).groupby("type_group", observed=False)["war_t"].agg(war_rate="mean", n="size").dropna()
        axes[1].bar(grouped.index.astype(str), grouped["war_rate"], color="#e76f51")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Share reaching war level")
        axes[1].set_title("Support-type breadth")
        for i, (rate, n) in enumerate(zip(grouped["war_rate"], grouped["n"])):
            axes[1].text(i, min(rate + 0.04, 0.97), f"{rate:.2f}\n(n={int(n)})",
                         ha="center", fontsize=9)
        fig.suptitle("Descriptive support mechanisms and subsequent intensity", y=1.02, fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "support_mechanisms.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "support_mechanisms.png", bbox_inches="tight")
        plt.close(fig)

    observed_path = output_dir / "h1_breadth_observed.csv"
    prediction_path = output_dir / "h1_functional_form_predictions.csv"
    loo_path = output_dir / "h1_leave_one_conflict_out.csv"
    if observed_path.exists() and prediction_path.exists():
        observed = pd.read_csv(observed_path)
        predictions = pd.read_csv(prediction_path)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(
            observed["n_support_types_confirmed"], observed["war_proportion"],
            s=35, color="#e76f51", label="Observed war proportion",
        )
        for specification, part in predictions.groupby("specification"):
            label = f"{specification.title()} adjusted probability"
            style = "-" if specification == "quadratic" else "--"
            ax.plot(
                part["breadth"], part["predicted_probability"],
                linewidth=2, linestyle=style, label=label,
            )
        ax.set_xlabel("Confirmed support breadth")
        ax.set_ylabel("Probability / proportion of war")
        ax.set_title("Observed and adjusted H1 breadth patterns")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "h1_breadth_functional_form.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "h1_breadth_functional_form.png", bbox_inches="tight")
        plt.close(fig)
    if loo_path.exists():
        loo = pd.read_csv(loo_path).dropna(
            subset=["odds_ratio", "or_ci_lower", "or_ci_upper"]
        )
        loo = loo[
            np.isfinite(loo[["odds_ratio", "or_ci_lower", "or_ci_upper"]]).all(axis=1)
            & (loo["odds_ratio"] > 0)
            & (loo["or_ci_lower"] > 0)
            & (loo["or_ci_upper"] > 0)
        ]
        if not loo.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.errorbar(
                loo["excluded_conflict_id"].astype(str), loo["odds_ratio"],
                yerr=[
                    loo["odds_ratio"] - loo["or_ci_lower"],
                    loo["or_ci_upper"] - loo["odds_ratio"],
                ],
                fmt="o", color="#1f4e79", capsize=3,
            )
            ax.axhline(1, color="#9ca3af", linestyle="--", linewidth=1)
            ax.set_yscale("log")
            ax.set_xlabel("Excluded conflict ID")
            ax.set_ylabel("Breadth odds ratio (95% CI)")
            ax.set_title("Leave-one-conflict-out H1 sensitivity")
            fig.tight_layout()
            fig.savefig(output_dir / "h1_conflict_sensitivity.pdf", bbox_inches="tight")
            fig.savefig(output_dir / "h1_conflict_sensitivity.png", bbox_inches="tight")
            plt.close(fig)

    if shap_values is not None and shap_features is not None and len(shap_features):
        shap.summary_plot(shap_values, shap_features, plot_type="dot", show=False,
                          max_display=min(10, shap_features.shape[1]))
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.title("XGBoost SHAP beeswarm\n(right = higher predicted war probability)")
        fig.tight_layout()
        fig.savefig(output_dir / "xgboost_shap_beeswarm.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "xgboost_shap_beeswarm.png", bbox_inches="tight")
        plt.close(fig)


def binary_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    has_both_classes = y.nunique() == 2
    result = {
        "n": int(len(y)),
        "n_war": int(y.sum()),
        "n_minor": int((y == 0).sum()),
        "war_prevalence": float(y.mean()),
        "accuracy": float(accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, probability)),
    }
    result["roc_auc"] = float(roc_auc_score(y, probability)) if has_both_classes else np.nan
    result["average_precision"] = (
        float(average_precision_score(y, probability)) if has_both_classes else np.nan
    )
    return result


def _safe_exp(value: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.exp(value))


def _ci_status(lower: float, upper: float) -> str:
    return "finite" if np.isfinite(lower) and np.isfinite(upper) else "unbounded"


def _inference_rows(
    model_name: str, hypothesis: str, terms: list[str], n: int,
    converged: bool, reason: str | None,
) -> list[dict[str, Any]]:
    status = "valid" if reason is None and converged else "non_estimable"
    return [{
        "model": model_name,
        "hypothesis": hypothesis,
        "term": term,
        "estimate": np.nan,
        "odds_ratio": np.nan,
        "or_ci_lower": np.nan,
        "or_ci_upper": np.nan,
        "ci_status": "not_available",
        "std_error": np.nan,
        "p_value": np.nan,
        "n": n,
        "converged": converged,
        "inference_status": status,
        "inference_error": reason or "",
    } for term in terms]


def _inference_diagnostics(
    fit: Any, captured_warnings: list[warnings.WarningMessage],
) -> tuple[bool, str]:
    converged = bool(fit.mle_retvals.get("converged", False))
    if not converged:
        return False, "maximum_likelihood_not_converged"
    arrays = {
        "parameters": np.asarray(fit.params, dtype=float),
        "standard_errors": np.asarray(fit.bse, dtype=float),
        "p_values": np.asarray(fit.pvalues, dtype=float),
    }
    invalid = [name for name, values in arrays.items() if not np.isfinite(values).all()]
    if invalid:
        return False, f"non_finite_{'_'.join(invalid)}"
    numerical = [
        warning for warning in captured_warnings
        if issubclass(warning.category, (RuntimeWarning, ConvergenceWarning))
    ]
    if numerical:
        return False, "numerical_warning:" + "; ".join(
            str(warning.message) for warning in numerical
        )
    return True, ""


def _fit_logit_sample(
    sample: pd.DataFrame, features: list[str], model_name: str,
    sample_specification: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    X = sm.add_constant(sample[features], has_constant="add")
    y = sample["war_t"].astype(int)
    try:
        fit = sm.Logit(y, X).fit(
            disp=False, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": sample["dyad_id"]}, warn_convergence=False,
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return [], {"model": model_name, "method": "logistic", "status": "failed",
                    "sample_specification": sample_specification, "n": len(sample),
                    "error": str(exc)}
    converged = bool(fit.mle_retvals.get("converged", True))
    ci = fit.conf_int()
    rows = [{
        "model": model_name, "sample_specification": sample_specification,
        "term": term, "estimate": float(estimate),
        "odds_ratio": _safe_exp(estimate),
        "or_ci_lower": _safe_exp(ci.loc[term].iloc[0]),
        "or_ci_upper": _safe_exp(ci.loc[term].iloc[1]),
        "ci_status": _ci_status(
            _safe_exp(ci.loc[term].iloc[0]), _safe_exp(ci.loc[term].iloc[1])
        ),
        "std_error": float(stderr), "p_value": float(pvalue),
        "n": len(sample), "converged": converged,
    } for term, estimate, stderr, pvalue in zip(
        fit.params.index, fit.params, fit.bse, fit.pvalues
    )]
    return rows, {
        "model": model_name, "method": "logistic",
        "status": "ok" if converged else "nonconverged",
        "sample_specification": sample_specification, "n": len(sample),
        "war_n": int(y.sum()), "minor_n": int((y == 0).sum()),
        "unique_dyads": int(sample["dyad_id"].nunique()),
        "unique_conflicts": int(sample["conflict_id"].nunique())
        if "conflict_id" in sample else np.nan,
        "pseudo_r2": float(fit.prsquared), "aic": float(fit.aic),
        "converged": converged,
        "warning": "" if converged else "maximum_likelihood_not_converged",
    }


def fit_logistic(df: pd.DataFrame, output_dir: Path) -> list[dict[str, Any]]:
    coefficients: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    m1_sample = complete_case(df, MODEL_FEATURES["M1"])
    adjusted = df.copy()
    adjusted["log_population"] = np.where(
        pd.to_numeric(adjusted["population"], errors="coerce") > 0,
        np.log(pd.to_numeric(adjusted["population"], errors="coerce")),
        np.nan,
    )
    adjusted_sample = complete_case(adjusted, ADJUSTED_SAMPLE_FEATURES)
    _write_control_diagnostics(adjusted_sample, output_dir)
    specifications = [
        ({"M0": MODEL_FEATURES["M0"], "M1": MODEL_FEATURES["M1"]},
         m1_sample, "primary_m1_common"),
        ({"M0_adjusted": MODEL_FEATURES["M0"],
          "M1_adjusted": MODEL_FEATURES["M1"],
          "M2_CINC": MODEL_FEATURES["M2_CINC"],
          "M3_GDP": MODEL_FEATURES["M3_GDP"],
          "M4_VDEM": MODEL_FEATURES["M4_VDEM"],
          "M2_POP_ALT": MODEL_FEATURES["M2_POP_ALT"]},
         adjusted_sample, "adjusted_alternatives_common"),
    ]
    for model_set, sample, specification in specifications:
        for model_name, features in model_set.items():
            rows, metric = _fit_logit_sample(sample, features, model_name, specification)
            coefficients.extend(rows)
            metrics.append(metric)
    pd.DataFrame(coefficients).to_csv(output_dir / "logistic_coefficients.csv", index=False)
    sensitivity_metrics = fit_h1_sensitivity(df, output_dir)
    metrics.extend(sensitivity_metrics)
    return metrics


def fit_h1_sensitivity(df: pd.DataFrame, output_dir: Path) -> list[dict[str, Any]]:
    """Run exploratory functional-form and leave-one-conflict-out H1 checks."""
    sample = complete_case(df, MODEL_FEATURES["M1"])
    breadth = "n_support_types_confirmed"
    sample = sample.copy()
    mean_breadth = float(sample[breadth].mean())
    sample["breadth_centered"] = sample[breadth] - mean_breadth
    sample["breadth_centered_sq"] = sample["breadth_centered"] ** 2
    quadratic_features = MODEL_FEATURES["M0"] + [
        "breadth_centered", "breadth_centered_sq"
    ]
    rows, metric = _fit_binary_specification(
        sample, quadratic_features, "war_t", "H1_quadratic",
        "h1_functional_form",
    )
    for row in rows:
        row["sample_specification"] = "primary_m1_common"
    metric.update({
        "method": "logistic_sensitivity",
        "sample_specification": "primary_m1_common",
        "breadth_mean": mean_breadth,
        "breadth_centered_mean": float(sample["breadth_centered"].mean()),
        "breadth_centered_sq_mean": float(sample["breadth_centered_sq"].mean()),
        "joint_test": _quadratic_joint_test(sample, quadratic_features),
        "sensitivity_role": "exploratory_functional_form",
    })
    pd.DataFrame(rows).to_csv(
        output_dir / "h1_quadratic_coefficients.csv", index=False
    )
    pd.DataFrame([metric]).to_csv(
        output_dir / "h1_quadratic_metrics.csv", index=False
    )

    conflict_summary = sample.groupby("conflict_id", dropna=False).agg(
        n=("war_t", "size"),
        unique_dyads=("dyad_id", "nunique"),
        war_n=("war_t", "sum"),
        mean_breadth=(breadth, "mean"),
        median_breadth=(breadth, "median"),
        max_breadth=(breadth, "max"),
    ).reset_index()
    conflict_summary.to_csv(
        output_dir / "h1_conflict_summary.csv", index=False
    )
    observed = sample.groupby(breadth).agg(
        n=("war_t", "size"), war_n=("war_t", "sum"),
        war_proportion=("war_t", "mean"),
    ).reset_index()
    observed.to_csv(output_dir / "h1_breadth_observed.csv", index=False)
    _write_h1_predictions(sample, output_dir, quadratic_features, mean_breadth)

    loo_rows: list[dict[str, Any]] = []
    for conflict_id in sample["conflict_id"].drop_duplicates():
        loo = sample[sample["conflict_id"] != conflict_id].copy()
        fit_rows, fit_metric = _fit_binary_specification(
            loo, MODEL_FEATURES["M1"], "war_t", "H1_leave_one_conflict_out",
            "h1_conflict_sensitivity",
        )
        target = next(
            (row for row in fit_rows if row["term"] == breadth),
            {"odds_ratio": np.nan, "or_ci_lower": np.nan,
             "or_ci_upper": np.nan, "p_value": np.nan,
             "converged": False, "inference_status": "non_estimable"},
        )
        loo_rows.append({
            "excluded_conflict_id": conflict_id,
            "n": len(loo),
            "war_n": int(loo["war_t"].sum()),
            "minor_n": int((loo["war_t"] == 0).sum()),
            "unique_dyads": int(loo["dyad_id"].nunique()),
            "unique_conflicts": int(loo["conflict_id"].nunique()),
            "breadth_mean": float(loo[breadth].mean()),
            "breadth_estimate": target.get("estimate", np.nan),
            "odds_ratio": target.get("odds_ratio", np.nan),
            "or_ci_lower": target.get("or_ci_lower", np.nan),
            "or_ci_upper": target.get("or_ci_upper", np.nan),
            "p_value": target.get("p_value", np.nan),
            "converged": fit_metric.get("converged", False),
            "status": fit_metric.get("status", "non_estimable"),
            "inference_error": fit_metric.get("inference_error", ""),
        })
    loo_frame = pd.DataFrame(loo_rows)
    loo_frame.to_csv(
        output_dir / "h1_leave_one_conflict_out.csv", index=False
    )
    loo_summary = {
        "model": "H1_leave_one_conflict_out",
        "method": "logistic_sensitivity",
        "status": "ok",
        "sensitivity_role": "exploratory_conflict_sensitivity",
        "attempted": len(loo_frame),
        "valid": int((loo_frame["status"] == "ok").sum()),
        "non_estimable": int((loo_frame["status"] != "ok").sum()),
    }
    return [metric, loo_summary]


def _write_h1_predictions(
    sample: pd.DataFrame, output_dir: Path, features: list[str], mean_breadth: float,
) -> None:
    breadth = "n_support_types_confirmed"
    grid = np.linspace(
        sample[breadth].min(), sample[breadth].max(), 100,
    )
    baseline_values = sample[MODEL_FEATURES["M0"]].median().to_dict()
    rows: list[dict[str, Any]] = []
    specifications = {
        "linear": MODEL_FEATURES["M0"] + [breadth],
        "quadratic": features,
    }
    for specification, specification_features in specifications.items():
        try:
            X = sm.add_constant(sample[specification_features], has_constant="add")
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fit = sm.Logit(sample["war_t"].astype(int), X).fit(
                    disp=False, maxiter=200, cov_type="cluster",
                    cov_kwds={"groups": sample["dyad_id"]},
                    warn_convergence=False,
                )
            estimable, _ = _inference_diagnostics(fit, captured)
            if not estimable:
                continue
            for value in grid:
                row = baseline_values.copy()
                row[breadth] = value
                row["breadth_centered"] = value - mean_breadth
                row["breadth_centered_sq"] = row["breadth_centered"] ** 2
                prediction = fit.predict(sm.add_constant(
                    pd.DataFrame([row])[specification_features], has_constant="add"
                ))
                rows.append({
                    "specification": specification,
                    "breadth": value,
                    "predicted_probability": float(prediction.iloc[0]),
                })
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    pd.DataFrame(rows).to_csv(
        output_dir / "h1_functional_form_predictions.csv", index=False
    )


def _quadratic_joint_test(sample: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    X = sm.add_constant(sample[features], has_constant="add")
    y = sample["war_t"].astype(int)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fit = sm.Logit(y, X).fit(
                disp=False, maxiter=200, cov_type="cluster",
                cov_kwds={"groups": sample["dyad_id"]},
                warn_convergence=False,
            )
        estimable, reason = _inference_diagnostics(fit, captured)
        if not estimable:
            return {"status": "non_estimable", "p_value": np.nan, "error": reason}
        restriction = np.zeros((2, len(fit.params)))
        restriction[0, list(fit.params.index).index("breadth_centered")] = 1
        restriction[1, list(fit.params.index).index("breadth_centered_sq")] = 1
        test = fit.wald_test(restriction, scalar=True)
        return {
            "status": "ok",
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "error": "",
        }
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return {"status": "non_estimable", "p_value": np.nan, "error": str(exc)}


def _write_control_diagnostics(sample: pd.DataFrame, output_dir: Path) -> None:
    diagnostic_features = [
        "cinc", "population", "gdp_per_capita", "v2x_polyarchy", "log_population",
    ]
    correlations = sample[diagnostic_features].corr(method="spearman")
    correlations.to_csv(output_dir / "control_spearman_correlations.csv")

    vif_rows: list[dict[str, Any]] = []
    for model_name in ("M2_CINC", "M3_GDP", "M4_VDEM", "M2_POP_ALT"):
        features = MODEL_FEATURES[model_name]
        design = sample[features].astype(float)
        try:
            values = [
                variance_inflation_factor(design.to_numpy(), index)
                for index in range(design.shape[1])
            ]
            error = ""
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            values = [np.nan] * len(features)
            error = str(exc)
        vif_rows.extend({
            "model": model_name,
            "feature": feature,
            "vif": float(value),
            "error": error,
            "n": len(sample),
            "war_n": int(sample["war_t"].sum()),
        } for feature, value in zip(features, values))
    pd.DataFrame(vif_rows).to_csv(output_dir / "control_vif.csv", index=False)


def _fit_binary_specification(
    sample: pd.DataFrame, features: list[str], outcome: str, model_name: str,
    hypothesis: str, extra_metric: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[outcome, "dyad_id", *features]
    )
    if len(sample) == 0 or sample[outcome].nunique() < 2:
        return [], {"model": model_name, "hypothesis": hypothesis, "status": "not_estimable",
                    "n": len(sample), "error": "Outcome has fewer than two classes."}
    X = sm.add_constant(sample[features], has_constant="add")
    y = sample[outcome].astype(int)
    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            fit = sm.Logit(y, X).fit(
                disp=False, maxiter=200, cov_type="cluster",
                cov_kwds={"groups": sample["dyad_id"]},
                warn_convergence=False,
            )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        terms = list(X.columns)
        return _inference_rows(
            model_name, hypothesis, terms, len(sample), False, f"fit_failed:{exc}",
        ), {"model": model_name, "hypothesis": hypothesis, "status": "non_estimable",
            "n": len(sample), "error": str(exc), "converged": False,
            "inference_error": f"fit_failed:{exc}"}
    converged = bool(fit.mle_retvals.get("converged", True))
    estimable, reason = _inference_diagnostics(fit, captured_warnings)
    ci = fit.conf_int()
    if not estimable:
        terms = list(fit.params.index)
        metric = {
            "model": model_name, "hypothesis": hypothesis,
            "status": "non_estimable", "n": len(sample),
            "pseudo_r2": np.nan, "aic": np.nan, "converged": converged,
            "warning": reason, "inference_error": reason,
        }
        if extra_metric:
            metric.update(extra_metric)
        return _inference_rows(
            model_name, hypothesis, terms, len(sample), converged, reason,
        ), metric
    rows = [{
        "model": model_name,
        "hypothesis": hypothesis,
        "term": term,
        "estimate": float(estimate),
        "odds_ratio": _safe_exp(estimate),
        "or_ci_lower": _safe_exp(ci.loc[term].iloc[0]),
        "or_ci_upper": _safe_exp(ci.loc[term].iloc[1]),
        "ci_status": _ci_status(
            _safe_exp(ci.loc[term].iloc[0]), _safe_exp(ci.loc[term].iloc[1])
        ),
        "std_error": float(stderr),
        "p_value": float(pvalue),
        "n": len(sample),
        "converged": converged,
        "inference_status": "valid",
        "inference_error": "",
    } for term, estimate, stderr, pvalue in zip(
        fit.params.index, fit.params, fit.bse, fit.pvalues
    )]
    metric = {
        "model": model_name, "hypothesis": hypothesis,
        "status": "ok" if converged else "nonconverged",
        "n": len(sample), "war_n": int(y.sum()), "minor_n": int((y == 0).sum()),
        "unique_dyads": int(sample["dyad_id"].nunique()),
        "unique_conflicts": int(sample["conflict_id"].nunique())
        if "conflict_id" in sample else np.nan,
        "pseudo_r2": float(fit.prsquared), "aic": float(fit.aic),
        "converged": converged,
        "warning": "",
        "inference_error": "",
    }
    if extra_metric:
        metric.update(extra_metric)
    return rows, metric


def fit_support_hypotheses(df: pd.DataFrame, output_dir: Path) -> None:
    """Estimate focused H2/H3 specifications without expanding the M0–M4 ladder."""
    baseline = MODEL_FEATURES["M0"]
    coefficient_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for feature in SUPPORT_TYPE_FEATURES:
        if feature not in df.columns:
            continue
        rows, metric = _fit_binary_specification(
            df, baseline + [feature], "war_t", f"H2_{feature.removesuffix('_confirmed')}",
            "support_type",
        )
        coefficient_rows.extend(rows)
        metric_rows.append(metric)

    valid_categories = ["none", "one_side_only", "both_sides"]
    configuration_counts = df["support_configuration_confirmed"].value_counts()
    excluded = (~df["support_configuration_confirmed"].isin(valid_categories)).sum()
    config = df[df["support_configuration_confirmed"].isin(valid_categories)].copy()
    config["support_one_side"] = (config["support_configuration_confirmed"] == "one_side_only").astype(int)
    config["support_both_sides"] = (config["support_configuration_confirmed"] == "both_sides").astype(int)
    descriptive_metrics = {
        "h3_none_n": int(configuration_counts.get("none", 0)),
        "h3_one_side_n": int(configuration_counts.get("one_side_only", 0)),
        "h3_both_sides_n": int(configuration_counts.get("both_sides", 0)),
        "h3_unknown_n": int(configuration_counts.get("unknown", 0)),
        "h3_unresolved_side_n": int(configuration_counts.get("unresolved_side", 0)),
    }
    rows, metric = _fit_binary_specification(
        config, baseline + ["support_one_side", "support_both_sides"],
        "war_t", "H3_support_configuration", "support_configuration",
        {**descriptive_metrics,
         "h3_excluded_invalid_categories": int(excluded),
         "h3_reference_category": "none"},
    )
    coefficient_rows.extend(rows)
    metric_rows.append(metric)
    supported = df[df["support_configuration_confirmed"].isin(
        ["one_side_only", "both_sides"]
    )].copy()
    supported["support_both_sides"] = (
        supported["support_configuration_confirmed"] == "both_sides"
    ).astype(int)
    supported_metric = {
        "h3_analysis": "exploratory_supported_dyads",
        "h3_reference_category": "one_side_only",
        "h3_supported_only": True,
        "h3_supported_n": int(len(supported)),
        "h3_supported_war_n": int(supported["war_t"].sum()),
        "h3_supported_one_side_n": int(
            (supported["support_configuration_confirmed"] == "one_side_only").sum()
        ),
        "h3_supported_both_sides_n": int(
            (supported["support_configuration_confirmed"] == "both_sides").sum()
        ),
        "h3_excluded_none_n": int((df["support_configuration_confirmed"] == "none").sum()),
        "h3_excluded_unresolved_n": int(
            (~df["support_configuration_confirmed"].isin(
                ["one_side_only", "both_sides", "none"]
            )).sum()
        ),
        "h3_interpretation": (
            "Among confirmed-support observations, both-sided versus one-sided "
            "support; not support versus no support."
        ),
    }
    rows, metric = _fit_binary_specification(
        supported, baseline + ["support_both_sides"], "war_t",
        "H3_supported_dyads", "support_configuration_supported",
        supported_metric,
    )
    coefficient_rows.extend(rows)
    metric_rows.append(metric)
    h2_rows = [row for row in coefficient_rows
               if row["hypothesis"] == "support_type"
               and row["term"] in SUPPORT_TYPE_FEATURES
               and row.get("inference_status") == "valid"
               and np.isfinite(row.get("p_value", np.nan))]
    h2_metrics = [row for row in metric_rows if row["hypothesis"] == "support_type"]
    valid_models = {
        row["model"] for row in h2_metrics if row.get("status") == "ok"
    }
    attempted_h2 = len(h2_metrics)
    valid_h2 = len(valid_models)
    non_estimable_h2 = attempted_h2 - valid_h2
    for metric in h2_metrics:
        metric.update({
            "h2_attempted": attempted_h2,
            "h2_valid": valid_h2,
            "h2_non_estimable": non_estimable_h2,
            "h2_fdr_family_size": len(h2_rows),
        })
    if h2_rows:
        _, corrected, _, _ = multipletests(
            [row["p_value"] for row in h2_rows], method="fdr_bh"
        )
        correction = {row["term"]: float(value) for row, value in zip(h2_rows, corrected)}
        for row in coefficient_rows:
            if (row["hypothesis"] == "support_type"
                    and row["term"] in correction
                    and row.get("model") in valid_models):
                row["p_value_fdr_bh"] = correction[row["term"]]
            elif row["hypothesis"] == "support_type":
                row["p_value_fdr_bh"] = np.nan

    escalation = df[df["lag_intensity"].eq(1)].copy()
    rows, metric = _fit_binary_specification(
        escalation,
        ["lag_conflict_type", "lag_episode_age_years", "year",
         "n_support_types_confirmed"],
        "minor_to_war", "escalation_type_breadth", "minor_to_war",
    )
    coefficient_rows.extend(rows)
    metric_rows.append(metric)
    pd.DataFrame(coefficient_rows).to_csv(
        output_dir / "support_hypothesis_coefficients.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "support_hypothesis_metrics.csv", index=False
    )


def temporal_cross_validation(
    sample: pd.DataFrame, features: list[str], output_dir: Path,
    random_state: int, first_test_year: int,
) -> list[dict[str, Any]]:
    """Evaluate tree models on repeated, forward-looking time folds.

    Each fold trains only on earlier years and tests on the following two years.
    Three estimator seeds provide a small sensitivity check without fabricating
    observations or treating repeated draws as independent data.
    """
    years = sorted(int(year) for year in sample["year"].dropna().unique())
    candidate_starts = [year for year in years if year >= first_test_year]
    starts = candidate_starts[::2][:4]
    folds: list[dict[str, Any]] = []
    for fold_id, start in enumerate(starts, start=1):
        train = sample[sample["year"] < start]
        test = sample[(sample["year"] >= start) & (sample["year"] < start + 2)]
        if (len(train) < 30 or len(test) < 4 or train["war_t"].nunique() < 2
                or test["war_t"].nunique() < 2):
            continue
        for model_name in ("random_forest", "xgboost"):
            for repetition, seed in enumerate(
                [random_state, random_state + 101, random_state + 202], start=1
            ):
                if model_name == "random_forest":
                    estimator = RandomForestClassifier(
                        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                        random_state=seed, n_jobs=-1,
                    )
                else:
                    estimator = XGBClassifier(
                        n_estimators=250, max_depth=2, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                        objective="binary:logistic", eval_metric="logloss",
                        random_state=seed, n_jobs=1,
                    )
                estimator.fit(train[features], train["war_t"].astype(int))
                probability = estimator.predict_proba(test[features])[:, 1]
                result = binary_metrics(test["war_t"].astype(int), probability)
                folds.append({
                    "fold": fold_id,
                    "model": model_name,
                    "repetition": repetition,
                    "test_year_min": int(test["year"].min()),
                    "test_year_max": int(test["year"].max()),
                    **{key: value for key, value in result.items() if key != "n"},
                    "train_n": len(train),
                    "test_n": len(test),
                })
    fold_frame = pd.DataFrame(folds)
    fold_frame.to_csv(output_dir / "temporal_cv_metrics.csv", index=False)
    if fold_frame.empty:
        return []
    summary = fold_frame.groupby("model").agg(
        folds=("fold", "nunique"),
        repetitions=("repetition", "count"),
        train_n_mean=("train_n", "mean"),
        test_n_mean=("test_n", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_sd=("average_precision", "std"),
        brier_mean=("brier", "mean"),
        brier_sd=("brier", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
    ).reset_index()
    summary.insert(1, "method", "predictive_temporal_cv")
    summary["status"] = "ok"
    return summary.to_dict("records")


PREDICTIVE_SPECS: dict[str, list[str]] = {
    "baseline": MODEL_FEATURES["M0"],
    "extended": MODEL_FEATURES["M1"],
}
PREDICTIVE_MODELS = ("l2_logistic", "random_forest", "xgboost")


def _predictive_estimator(model_name: str, seed: int) -> Any:
    if model_name == "l2_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, solver="liblinear",
                max_iter=1000, random_state=seed,
            ),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
    return XGBClassifier(
        n_estimators=250, max_depth=2, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        objective="binary:logistic", eval_metric="logloss",
        random_state=seed, n_jobs=1,
    )


def grouped_cross_validation(
    sample: pd.DataFrame, output_dir: Path, random_state: int, n_splits: int,
    split_seeds: list[int] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray | None, pd.DataFrame | None]:
    """Compare specifications on repeated, identical dyad-held-out folds."""
    if n_splits < 2 or sample["dyad_id"].nunique() < n_splits:
        raise ValueError("Grouped CV requires at least as many dyads as folds.")
    split_seeds = split_seeds or [
        random_state, random_state + 1009, random_state + 2017
    ]
    y = sample["war_t"].astype(int)
    groups = sample["dyad_id"]
    folds: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    shap_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    estimator_seeds = [
        random_state + 100_003,
        random_state + 100_104,
        random_state + 100_205,
    ]
    for split_repetition, split_seed in enumerate(split_seeds, start=1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=split_seed
        )
        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(sample[PREDICTIVE_SPECS["extended"]], y, groups), start=1
        ):
            train, test = sample.iloc[train_idx], sample.iloc[test_idx]
            train_dyads, test_dyads = set(train["dyad_id"]), set(test["dyad_id"])
            if train_dyads & test_dyads:
                raise RuntimeError(
                    f"Dyad leakage detected in split {split_repetition}, fold {fold_id}."
                )
            train_conflicts = set(train["conflict_id"])
            test_conflicts = set(test["conflict_id"])
            shared_conflicts = train_conflicts & test_conflicts
            overlap_rows.append({
                "split_repetition": split_repetition,
                "split_seed": split_seed,
                "fold": fold_id,
                "train_conflicts": len(train_conflicts),
                "test_conflicts": len(test_conflicts),
                "shared_conflicts": len(shared_conflicts),
                "shared_conflict_ids": "|".join(
                    str(value) for value in sorted(shared_conflicts, key=str)
                ),
            })
            for spec_name, features in PREDICTIVE_SPECS.items():
                for model_name in PREDICTIVE_MODELS:
                    for estimator_repetition, seed in enumerate(
                        estimator_seeds, start=1
                    ):
                        base = {
                            "split_repetition": split_repetition,
                            "split_seed": split_seed,
                            "fold": fold_id,
                            "specification": spec_name,
                            "model": model_name,
                            "repetition": estimator_repetition,
                            "status": "not_estimable",
                            "train_n": len(train),
                            "test_n": len(test),
                            "train_dyads": len(train_dyads),
                            "test_dyads": len(test_dyads),
                            "train_n_war": int(train["war_t"].sum()),
                            "train_n_minor": int((train["war_t"] == 0).sum()),
                            "test_n_war": int(test["war_t"].sum()),
                            "test_n_minor": int((test["war_t"] == 0).sum()),
                            "test_war_prevalence": float(test["war_t"].mean()),
                        }
                        try:
                            if train["war_t"].nunique() < 2:
                                folds.append(base)
                                continue
                            estimator = _predictive_estimator(model_name, seed)
                            estimator.fit(train[features], train["war_t"].astype(int))
                            probability = estimator.predict_proba(test[features])[:, 1]
                            result = binary_metrics(test["war_t"].astype(int), probability)
                            base.update({
                                "status": "ok",
                                **{key: value for key, value in result.items()
                                   if key not in {"n", "n_war", "n_minor",
                                                  "war_prevalence"}},
                            })
                            for row_id, actual, predicted in zip(
                                test.index, test["war_t"], probability
                            ):
                                prediction_rows.append({
                                    "split_repetition": split_repetition,
                                    "split_seed": split_seed,
                                    "fold": fold_id,
                                    "specification": spec_name,
                                    "model": model_name,
                                    "repetition": estimator_repetition,
                                    "row_index": row_id,
                                    "dyad_id": test.loc[row_id, "dyad_id"],
                                    "conflict_id": test.loc[row_id, "conflict_id"],
                                    "year": test.loc[row_id, "year"],
                                    "war_t": int(actual),
                                    "predicted_probability": float(predicted),
                                })
                            if (
                                split_repetition == 1
                                and spec_name == "extended"
                                and model_name == "xgboost"
                                and estimator_repetition == 1
                            ):
                                values = shap.TreeExplainer(estimator).shap_values(
                                    test[features]
                                )
                                if isinstance(values, list):
                                    values = values[-1]
                                frame = pd.DataFrame(values, columns=features)
                                frame.insert(0, "fold", fold_id)
                                frame.insert(1, "dyad_id", test["dyad_id"].to_numpy())
                                if "year" not in frame.columns:
                                    frame.insert(2, "year", test["year"].to_numpy())
                                else:
                                    frame.insert(2, "observation_year", test["year"].to_numpy())
                                shap_rows.append(frame)
                            if hasattr(estimator, "feature_importances_"):
                                importance_rows.extend(
                                    {"split_repetition": split_repetition,
                                     "split_seed": split_seed, "fold": fold_id,
                                     "specification": spec_name, "model": model_name,
                                     "feature": feature, "importance": float(value)}
                                    for feature, value in zip(
                                        features, estimator.feature_importances_
                                    )
                                )
                        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                            base["error"] = str(exc)
                        folds.append(base)
    seed_frame = pd.DataFrame(folds)
    seed_frame.to_csv(output_dir / "grouped_cv_seed_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "grouped_cv_predictions.csv", index=False
    )
    pd.DataFrame(overlap_rows).to_csv(
        output_dir / "grouped_cv_conflict_overlap.csv", index=False
    )
    if seed_frame.empty:
        return [], None, None
    group_keys = ["split_repetition", "split_seed", "fold", "specification", "model"]
    fold_frame = seed_frame.groupby(group_keys, as_index=False).agg(
        train_n=("train_n", "first"), test_n=("test_n", "first"),
        train_dyads=("train_dyads", "first"), test_dyads=("test_dyads", "first"),
        train_n_war=("train_n_war", "first"), train_n_minor=("train_n_minor", "first"),
        test_n_war=("test_n_war", "first"), test_n_minor=("test_n_minor", "first"),
        test_war_prevalence=("test_war_prevalence", "first"),
        valid_estimator_seeds=("status", lambda values: int((values == "ok").sum())),
        roc_auc=("roc_auc", "mean"), average_precision=("average_precision", "mean"),
        brier=("brier", "mean"), accuracy=("accuracy", "mean"),
    )
    fold_frame["status"] = np.where(
        fold_frame["valid_estimator_seeds"] == len(estimator_seeds),
        "ok",
        np.where(fold_frame["valid_estimator_seeds"] > 0, "partial", "not_estimable"),
    )
    fold_frame.to_csv(output_dir / "grouped_cv_fold_metrics.csv", index=False)
    repetition_frame = fold_frame.groupby(
        ["split_repetition", "split_seed", "specification", "model"], as_index=False
    ).agg(
        folds=("fold", "nunique"), valid_folds=("status", lambda values: int((values != "not_estimable").sum())),
        train_n_mean=("train_n", "mean"), test_n_mean=("test_n", "mean"),
        test_dyads_mean=("test_dyads", "mean"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_sd=("average_precision", "std"),
        brier_mean=("brier", "mean"), brier_sd=("brier", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
    ).reset_index()
    repetition_frame.insert(2, "method", "predictive_group_cv_repetition")
    repetition_frame["status"] = np.where(
        repetition_frame["valid_folds"] == n_splits, "ok", "partial"
    )
    repetition_frame.to_csv(
        output_dir / "grouped_cv_repetition_metrics.csv", index=False
    )
    pivot = repetition_frame.pivot_table(
        index=["split_repetition", "split_seed"], columns=["specification", "model"],
        values=["average_precision_mean", "brier_mean"], aggfunc="first",
    )
    for model_name in PREDICTIVE_MODELS:
        extended_key = ("average_precision_mean", "extended", model_name)
        baseline_key = ("average_precision_mean", "baseline", model_name)
        extended_brier_key = ("brier_mean", "extended", model_name)
        baseline_brier_key = ("brier_mean", "baseline", model_name)
        if extended_key in pivot and baseline_key in pivot:
            repetition_frame.loc[
                (repetition_frame["specification"] == "extended")
                & (repetition_frame["model"] == model_name),
                "delta_ap_vs_baseline",
            ] = (
                repetition_frame.loc[
                    (repetition_frame["specification"] == "extended")
                    & (repetition_frame["model"] == model_name),
                    ["split_repetition", "split_seed"],
                ].set_index(["split_repetition", "split_seed"]).index.map(
                    (pivot[extended_key] - pivot[baseline_key]).to_dict()
                )
            )
            repetition_frame.loc[
                (repetition_frame["specification"] == "extended")
                & (repetition_frame["model"] == model_name),
                "delta_brier_vs_baseline",
            ] = (
                repetition_frame.loc[
                    (repetition_frame["specification"] == "extended")
                    & (repetition_frame["model"] == model_name),
                    ["split_repetition", "split_seed"],
                ].set_index(["split_repetition", "split_seed"]).index.map(
                    (pivot[extended_brier_key] - pivot[baseline_brier_key]).to_dict()
                )
            )
    summaries = repetition_frame.groupby(["specification", "model"]).agg(
        split_repetitions=("split_repetition", "nunique"),
        folds_per_repetition=("folds", "first"),
        train_n_mean=("train_n_mean", "mean"),
        test_n_mean=("test_n_mean", "mean"),
        test_dyads_mean=("test_dyads_mean", "mean"),
        roc_auc_mean=("roc_auc_mean", "mean"),
        roc_auc_sd_across_repetitions=("roc_auc_mean", "std"),
        average_precision_mean=("average_precision_mean", "mean"),
        average_precision_sd_across_repetitions=("average_precision_mean", "std"),
        brier_mean=("brier_mean", "mean"),
        brier_sd_across_repetitions=("brier_mean", "std"),
        accuracy_mean=("accuracy_mean", "mean"),
        accuracy_sd_across_repetitions=("accuracy_mean", "std"),
    ).reset_index()
    summaries.insert(1, "method", "predictive_group_cv")
    summaries["status"] = "ok"
    for model_name in PREDICTIVE_MODELS:
        deltas = repetition_frame[
            (repetition_frame["specification"] == "extended")
            & (repetition_frame["model"] == model_name)
        ]
        if not deltas.empty:
            summaries.loc[
                (summaries["model"] == model_name) & (summaries["specification"] == "extended"),
                "delta_ap_vs_baseline",
            ] = deltas["delta_ap_vs_baseline"].mean()
            summaries.loc[
                (summaries["model"] == model_name) & (summaries["specification"] == "extended"),
                "delta_brier_vs_baseline",
            ] = deltas["delta_brier_vs_baseline"].mean()
            summaries.loc[
                (summaries["model"] == model_name) & (summaries["specification"] == "extended"),
                "delta_ap_sd_across_repetitions",
            ] = deltas["delta_ap_vs_baseline"].std()
            summaries.loc[
                (summaries["model"] == model_name) & (summaries["specification"] == "extended"),
                "delta_brier_sd_across_repetitions",
            ] = deltas["delta_brier_vs_baseline"].std()
    shap_values = pd.concat(shap_rows, ignore_index=True) if shap_rows else None
    if shap_values is not None:
        shap_features = shap_values[PREDICTIVE_SPECS["extended"]]
        shap_matrix = shap_features.to_numpy()
        shap_values.to_csv(output_dir / "xgboost_shap_values.csv", index=False)
        pd.DataFrame({
            "feature": PREDICTIVE_SPECS["extended"],
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            output_dir / "xgboost_shap_importance.csv", index=False
        )
    pd.DataFrame(importance_rows).to_csv(
        output_dir / "tree_feature_importance.csv", index=False
    )
    return summaries.to_dict("records"), (
        shap_values[PREDICTIVE_SPECS["extended"]].to_numpy()
        if shap_values is not None else None
    ), (shap_values[PREDICTIVE_SPECS["extended"]] if shap_values is not None else None)


def fit_predictive_models(
    df: pd.DataFrame, output_dir: Path, test_start_year: int,
    random_state: int, cv_splits: int, run_temporal: bool = False,
    split_seeds: list[int] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray | None, pd.DataFrame | None]:
    """Run primary grouped CV; temporal validation is optional robustness analysis."""
    sample = complete_case(df, PREDICTIVE_SPECS["extended"])
    metrics, shap_values, shap_features = grouped_cross_validation(
        sample, output_dir, random_state, cv_splits, split_seeds
    )
    if run_temporal:
        metrics.extend(temporal_cross_validation(
            sample, PREDICTIVE_SPECS["extended"], output_dir, random_state, test_start_year
        ))
    return metrics, shap_values, shap_features


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_panel(ROOT / args.dataset)
    logistic_metrics = fit_logistic(df, output_dir)
    fit_support_hypotheses(df, output_dir)
    predictive_metrics, shap_values, shap_features = fit_predictive_models(
        df, output_dir, args.test_start_year, args.random_state, args.cv_splits,
        args.temporal_validation, args.cv_split_seeds)
    metrics = pd.DataFrame(logistic_metrics + predictive_metrics)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    coefficients = pd.read_csv(output_dir / "logistic_coefficients.csv")
    save_plots(output_dir, coefficients, metrics, df, shap_values, shap_features)
    (output_dir / "run_summary.json").write_text(json.dumps({
        "dataset": str(ROOT / args.dataset),
        "rows": int(len(df)),
        "test_start_year": args.test_start_year if args.temporal_validation else None,
        "random_state": args.random_state,
        "cv_splits": args.cv_splits,
        "cv_split_seeds": args.cv_split_seeds or [
            args.random_state, args.random_state + 1009, args.random_state + 2017
        ],
        "primary_evaluation": "dyad-grouped stratified CV",
        "grouping_variable": "dyad_id",
        "predictive_specifications": list(PREDICTIVE_SPECS),
        "predictive_algorithms": list(PREDICTIVE_MODELS),
        "temporal_validation": bool(args.temporal_validation),
        "logistic_models": list(MODEL_FEATURES),
        "support_hypotheses": ["H2_support_type", "H3_support_configuration",
                               "H3_supported_dyads", "minor_to_war_escalation"],
        "predictive_features": {
            "baseline": PREDICTIVE_SPECS["baseline"],
            "extended": PREDICTIVE_SPECS["extended"],
        },
        "cv_outputs": {
            "seed_metrics": "grouped_cv_seed_metrics.csv",
            "fold_metrics": "grouped_cv_fold_metrics.csv",
            "repetition_metrics": "grouped_cv_repetition_metrics.csv",
            "predictions": "grouped_cv_predictions.csv",
            "conflict_overlap": "grouped_cv_conflict_overlap.csv",
        },
        "note": "Logistic models use cluster-robust standard errors by dyad. "
                "Predictive models use complete-case data and dyad-held-out stratified "
                "grouped cross-validation; temporal validation is optional. Repeated estimator "
                "seeds assess algorithmic sensitivity, not additional independent observations. "
                "Split-seed repetitions are summarized hierarchically and are not additional "
                "observations. Conflict overlap between train and test is reported diagnostically. "
                "SHAP values describe prediction, not causal effects.",
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nAnalysis outputs: {output_dir}")


if __name__ == "__main__":
    main()
