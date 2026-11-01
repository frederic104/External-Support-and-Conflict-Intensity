"""Fit the planned inferential and predictive models on the integrated panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).resolve().parents[1]

MODEL_FEATURES: dict[str, list[str]] = {
    "M0": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year"],
    "M1": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "n_support_types_confirmed"],
    "M2": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "n_support_types_confirmed", "cinc", "population"],
    "M3": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "n_support_types_confirmed", "cinc", "population", "gdp_per_capita"],
    "M4": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "n_support_types_confirmed", "cinc", "population", "gdp_per_capita",
           "v2x_polyarchy"],
}
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
        ax.set_title("Predictive performance across temporal validation folds")
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
    result = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, probability)),
    }
    result["roc_auc"] = float(roc_auc_score(y, probability)) if y.nunique() == 2 else np.nan
    result["average_precision"] = (
        float(average_precision_score(y, probability)) if y.nunique() == 2 else np.nan
    )
    return result


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
        "odds_ratio": float(np.exp(estimate)),
        "or_ci_lower": float(np.exp(ci.loc[term].iloc[0])),
        "or_ci_upper": float(np.exp(ci.loc[term].iloc[1])),
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
    m4_sample = complete_case(df, MODEL_FEATURES["M4"])
    specifications = [
        ({"M0": MODEL_FEATURES["M0"], "M1": MODEL_FEATURES["M1"]},
         m1_sample, "primary_m1_common"),
        ({f"{name}_adjusted": features for name, features in MODEL_FEATURES.items()},
         m4_sample, "adjusted_m4_common"),
    ]
    for model_set, sample, specification in specifications:
        for model_name, features in model_set.items():
            rows, metric = _fit_logit_sample(sample, features, model_name, specification)
            coefficients.extend(rows)
            metrics.append(metric)
    pd.DataFrame(coefficients).to_csv(output_dir / "logistic_coefficients.csv", index=False)
    return metrics


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
        fit = sm.Logit(y, X).fit(
            disp=False, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": sample["dyad_id"]},
            warn_convergence=False,
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return [], {"model": model_name, "hypothesis": hypothesis, "status": "failed",
                    "n": len(sample), "error": str(exc)}
    converged = bool(fit.mle_retvals.get("converged", True))
    ci = fit.conf_int()
    rows = [{
        "model": model_name,
        "hypothesis": hypothesis,
        "term": term,
        "estimate": float(estimate),
        "odds_ratio": float(np.exp(estimate)),
        "or_ci_lower": float(np.exp(ci.loc[term].iloc[0])),
        "or_ci_upper": float(np.exp(ci.loc[term].iloc[1])),
        "std_error": float(stderr),
        "p_value": float(pvalue),
        "n": len(sample),
        "converged": converged,
    } for term, estimate, stderr, pvalue in zip(
        fit.params.index, fit.params, fit.bse, fit.pvalues
    )]
    metric = {
        "model": model_name, "hypothesis": hypothesis,
        "status": "ok" if converged else "nonconverged",
        "n": len(sample), "pseudo_r2": float(fit.prsquared), "aic": float(fit.aic),
        "converged": converged,
        "warning": "" if converged else "maximum_likelihood_not_converged",
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
    excluded = (~df["support_configuration_confirmed"].isin(valid_categories)).sum()
    config = df[df["support_configuration_confirmed"].isin(valid_categories)].copy()
    config["support_one_side"] = (config["support_configuration_confirmed"] == "one_side_only").astype(int)
    config["support_both_sides"] = (config["support_configuration_confirmed"] == "both_sides").astype(int)
    rows, metric = _fit_binary_specification(
        config, baseline + ["support_one_side", "support_both_sides"],
        "war_t", "H3_support_configuration", "support_configuration",
        {"h3_excluded_invalid_categories": int(excluded),
         "h3_reference_category": "none"},
    )
    coefficient_rows.extend(rows)
    metric_rows.append(metric)
    h2_rows = [row for row in coefficient_rows
               if row["hypothesis"] == "support_type"
               and row["term"] in SUPPORT_TYPE_FEATURES]
    if h2_rows:
        _, corrected, _, _ = multipletests(
            [row["p_value"] for row in h2_rows], method="fdr_bh"
        )
        correction = {row["term"]: float(value) for row, value in zip(h2_rows, corrected)}
        for row in coefficient_rows:
            if row["hypothesis"] == "support_type" and row["term"] in correction:
                row["p_value_fdr_bh"] = correction[row["term"]]

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


def fit_predictive_models(
    df: pd.DataFrame, output_dir: Path, test_start_year: int,
    random_state: int, cv_splits: int,
) -> tuple[list[dict[str, Any]], np.ndarray | None, pd.DataFrame | None]:
    features = MODEL_FEATURES["M4"]
    sample = complete_case(df, features)
    train = sample[sample["year"] < test_start_year].copy()
    test = sample[sample["year"] >= test_start_year].copy()
    if train.empty or test.empty or train["war_t"].nunique() < 2 or test["war_t"].nunique() < 2:
        raise ValueError(
            f"Time split at {test_start_year} must leave both classes in train and test; "
            f"train={len(train)}, test={len(test)}."
        )
    X_train, X_test = train[features], test[features]
    y_train, y_test = train["war_t"].astype(int), test["war_t"].astype(int)
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
            objective="binary:logistic", eval_metric="logloss",
            random_state=random_state, n_jobs=1,
        ),
    }
    metrics: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    shap_values_out = None
    shap_features_out = None
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        probability = estimator.predict_proba(X_test)[:, 1]
        metrics.append({
            "model": name,
            "method": "predictive",
            "status": "ok",
            "train_n": len(train),
            "test_n": len(test),
            "test_year_min": int(test["year"].min()),
            "test_year_max": int(test["year"].max()),
            **{f"test_{key}": value for key, value in binary_metrics(y_test, probability).items()
               if key != "n"},
        })
        for feature, value in zip(features, estimator.feature_importances_):
            importance_rows.append({"model": name, "feature": feature, "importance": float(value)})
        if name == "xgboost":
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            shap_frame = pd.DataFrame(shap_values, columns=features)
            shap_frame.insert(0, "dyad_id", test["dyad_id"].to_numpy())
            if "year" not in shap_frame.columns:
                shap_frame.insert(1, "year", test["year"].to_numpy())
            shap_frame.to_csv(output_dir / "xgboost_shap_values.csv", index=False)
            shap_values_out = shap_values
            shap_features_out = X_test.copy()
            shap_importance = pd.DataFrame({
                "feature": features,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            shap_importance.to_csv(output_dir / "xgboost_shap_importance.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output_dir / "tree_feature_importance.csv", index=False)
    cv_summary = grouped_cross_validation(
        sample, features, output_dir, random_state, n_splits=cv_splits
    )
    cv_summary.extend(temporal_cross_validation(
        sample, features, output_dir, random_state, test_start_year
    ))
    metrics.extend(cv_summary)
    return metrics, shap_values_out, shap_features_out


def _predictive_estimator(model_name: str, seed: int) -> Any:
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
    sample: pd.DataFrame, features: list[str], output_dir: Path,
    random_state: int, n_splits: int,
) -> list[dict[str, Any]]:
    """Evaluate models on dyad-held-out, class-stratified folds."""
    if n_splits < 2 or sample["dyad_id"].nunique() < n_splits:
        raise ValueError("Grouped CV requires at least as many dyads as folds.")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    rows: list[dict[str, Any]] = []
    X, y, groups = sample[features], sample["war_t"].astype(int), sample["dyad_id"]
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        train, test = sample.iloc[train_idx], sample.iloc[test_idx]
        for model_name in ("random_forest", "xgboost"):
            for repetition, seed in enumerate(
                [random_state, random_state + 101, random_state + 202], start=1
            ):
                estimator = _predictive_estimator(model_name, seed)
                estimator.fit(train[features], train["war_t"].astype(int))
                probability = estimator.predict_proba(test[features])[:, 1]
                result = binary_metrics(test["war_t"].astype(int), probability)
                rows.append({
                    "fold": fold_id,
                    "model": model_name,
                    "repetition": repetition,
                    "train_n": len(train),
                    "test_n": len(test),
                    "train_dyads": train["dyad_id"].nunique(),
                    "test_dyads": test["dyad_id"].nunique(),
                    **{key: value for key, value in result.items() if key != "n"},
                })
    fold_frame = pd.DataFrame(rows)
    fold_frame.to_csv(output_dir / "grouped_cv_metrics.csv", index=False)
    summary = fold_frame.groupby("model").agg(
        folds=("fold", "nunique"),
        repetitions=("repetition", "count"),
        train_n_mean=("train_n", "mean"),
        test_n_mean=("test_n", "mean"),
        test_dyads_mean=("test_dyads", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_sd=("average_precision", "std"),
        brier_mean=("brier", "mean"),
        brier_sd=("brier", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
    ).reset_index()
    summary.insert(1, "method", "predictive_group_cv")
    summary["status"] = "ok"
    return summary.to_dict("records")


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


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_panel(ROOT / args.dataset)
    logistic_metrics = fit_logistic(df, output_dir)
    fit_support_hypotheses(df, output_dir)
    predictive_metrics, shap_values, shap_features = fit_predictive_models(
        df, output_dir, args.test_start_year, args.random_state, args.cv_splits)
    metrics = pd.DataFrame(logistic_metrics + predictive_metrics)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    coefficients = pd.read_csv(output_dir / "logistic_coefficients.csv")
    save_plots(output_dir, coefficients, metrics, df, shap_values, shap_features)
    (output_dir / "run_summary.json").write_text(json.dumps({
        "dataset": str(ROOT / args.dataset),
        "rows": int(len(df)),
        "test_start_year": args.test_start_year,
        "random_state": args.random_state,
        "cv_splits": args.cv_splits,
        "logistic_models": list(MODEL_FEATURES),
        "support_hypotheses": ["H2_support_type", "H3_support_configuration",
                               "minor_to_war_escalation"],
        "predictive_features": MODEL_FEATURES["M4"],
        "note": "Logistic models use cluster-robust standard errors by dyad. "
                "Tree models use complete-case data, dyad-held-out stratified grouped "
                "cross-validation, and repeated expanding-window temporal validation; "
                "repeated seeds assess "
                "algorithmic sensitivity, not additional independent observations. "
                "SHAP values describe prediction, not causal effects.",
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nAnalysis outputs: {output_dir}")


if __name__ == "__main__":
    main()
