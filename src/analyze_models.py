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
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model dataset not found: {path}. Run src/pipeline.py first.")
    df = pd.read_csv(path)
    required = {"war_t", "year", "dyad_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Model dataset is missing required columns: {missing}")
    if "n_supporters_confirmed_raw" not in df.columns:
        raise ValueError(
            "Model dataset is missing required support-count column: "
            "n_supporters_confirmed_raw"
        )
    required_support = {"n_support_types_confirmed", "support_configuration_confirmed"}
    missing_support = sorted(required_support - set(df.columns))
    if missing_support:
        raise ValueError(f"Model dataset is missing support-analysis columns: {missing_support}")
    df["log_supporters_confirmed"] = np.log1p(df["n_supporters_confirmed_raw"])
    return df


def complete_case(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"Model dataset is missing model columns: {missing}")
    columns = list(dict.fromkeys(["war_t", "year", "dyad_id", *features]))
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

    predictive = metrics[metrics["method"].eq("predictive")].copy()
    if not predictive.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        for ax, metric, label, higher_is_better in [
            (axes[0], "test_roc_auc", "ROC-AUC (higher is better)", True),
            (axes[1], "test_brier", "Brier score (lower is better)", False),
        ]:
            part = predictive[["model", metric]].dropna().sort_values(metric)
            ax.barh(part["model"].str.replace("_", " ").str.title(), part[metric],
                    color="#2a6f97")
            for y, value in enumerate(part[metric]):
                ax.text(value + (0.01 if higher_is_better else 0.002), y,
                        f"{value:.3f}", va="center", fontsize=9)
            ax.set_title(label)
            ax.set_xlim(0, min(1, max(part[metric].max() * 1.18, 0.1)))
            ax.grid(axis="x", alpha=0.25)
        fig.suptitle("Out-of-time predictive performance (test years)", y=1.02, fontsize=14)
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


def fit_logistic(df: pd.DataFrame, output_dir: Path) -> list[dict[str, Any]]:
    coefficients: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for model_name, features in MODEL_FEATURES.items():
        sample = complete_case(df, features)
        X = sm.add_constant(sample[features], has_constant="add")
        y = sample["war_t"].astype(int)
        try:
            fit = sm.Logit(y, X).fit(
                disp=False,
                maxiter=200,
                cov_type="cluster",
                cov_kwds={"groups": sample["dyad_id"]},
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            metrics.append({"model": model_name, "method": "logistic", "status": "failed",
                            "n": len(sample), "error": str(exc)})
            continue
        probability = fit.predict(X)
        metrics.append({"model": model_name, "method": "logistic", "status": "ok",
                        **binary_metrics(y, probability),
                        "pseudo_r2": float(fit.prsquared),
                        "aic": float(fit.aic)})
        for term, estimate, stderr, pvalue in zip(
                fit.params.index, fit.params, fit.bse, fit.pvalues):
            coefficients.append({
                "model": model_name,
                "term": term,
                "estimate": float(estimate),
                "odds_ratio": float(np.exp(estimate)),
                "std_error": float(stderr),
                "p_value": float(pvalue),
                "n": len(sample),
            })
    pd.DataFrame(coefficients).to_csv(output_dir / "logistic_coefficients.csv", index=False)
    return metrics


def _fit_binary_specification(
    sample: pd.DataFrame, features: list[str], outcome: str, model_name: str,
    hypothesis: str,
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
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return [], {"model": model_name, "hypothesis": hypothesis, "status": "failed",
                    "n": len(sample), "error": str(exc)}
    rows = [{
        "model": model_name,
        "hypothesis": hypothesis,
        "term": term,
        "estimate": float(estimate),
        "odds_ratio": float(np.exp(estimate)),
        "std_error": float(stderr),
        "p_value": float(pvalue),
        "n": len(sample),
    } for term, estimate, stderr, pvalue in zip(
        fit.params.index, fit.params, fit.bse, fit.pvalues
    )]
    return rows, {
        "model": model_name, "hypothesis": hypothesis, "status": "ok",
        "n": len(sample), "pseudo_r2": float(fit.prsquared), "aic": float(fit.aic),
    }


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

    config = df.copy()
    config["support_one_side"] = (config["support_configuration_confirmed"] == "one_side_only").astype(int)
    config["support_both_sides"] = (config["support_configuration_confirmed"] == "both_sides").astype(int)
    rows, metric = _fit_binary_specification(
        config, baseline + ["support_one_side", "support_both_sides"],
        "war_t", "H3_support_configuration", "support_configuration",
    )
    coefficient_rows.extend(rows)
    metric_rows.append(metric)

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
    df: pd.DataFrame, output_dir: Path, test_start_year: int, random_state: int
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
    return metrics, shap_values_out, shap_features_out


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_panel(ROOT / args.dataset)
    logistic_metrics = fit_logistic(df, output_dir)
    fit_support_hypotheses(df, output_dir)
    predictive_metrics, shap_values, shap_features = fit_predictive_models(
        df, output_dir, args.test_start_year, args.random_state)
    metrics = pd.DataFrame(logistic_metrics + predictive_metrics)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    coefficients = pd.read_csv(output_dir / "logistic_coefficients.csv")
    save_plots(output_dir, coefficients, metrics, df, shap_values, shap_features)
    (output_dir / "run_summary.json").write_text(json.dumps({
        "dataset": str(ROOT / args.dataset),
        "rows": int(len(df)),
        "test_start_year": args.test_start_year,
        "random_state": args.random_state,
        "logistic_models": list(MODEL_FEATURES),
        "support_hypotheses": ["H2_support_type", "H3_support_configuration",
                               "minor_to_war_escalation"],
        "predictive_features": MODEL_FEATURES["M4"],
        "note": "Logistic models use cluster-robust standard errors by dyad. "
                "Tree models use complete-case data and a fixed time-ordered split; "
                "SHAP values describe prediction, not causal effects.",
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nAnalysis outputs: {output_dir}")


if __name__ == "__main__":
    main()
