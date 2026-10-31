"""Fit the planned inferential and predictive models on the integrated panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).resolve().parents[1]

MODEL_FEATURES: dict[str, list[str]] = {
    "M0": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year"],
    "M1": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "any_support_main"],
    "M2": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "any_support_main", "cinc", "population"],
    "M3": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "any_support_main", "cinc", "population", "gdp_per_capita"],
    "M4": ["lag_intensity", "lag_conflict_type", "lag_episode_age_years", "year",
           "any_support_main", "cinc", "population", "gdp_per_capita",
           "v2x_polyarchy"],
}


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
    return df


def complete_case(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"Model dataset is missing model columns: {missing}")
    out = df[["war_t", "year", "dyad_id", *features]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out["war_t"].nunique() < 2:
        raise ValueError("A model sample must contain both war_t classes.")
    return out


def binary_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    result = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, probability)),
    }
    result["roc_auc"] = float(roc_auc_score(y, probability)) if y.nunique() == 2 else np.nan
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


def fit_predictive_models(
    df: pd.DataFrame, output_dir: Path, test_start_year: int, random_state: int
) -> list[dict[str, Any]]:
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
            shap_frame.insert(1, "year", test["year"].to_numpy())
            shap_frame.to_csv(output_dir / "xgboost_shap_values.csv", index=False)
            pd.DataFrame({
                "feature": features,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False).to_csv(
                output_dir / "xgboost_shap_importance.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output_dir / "tree_feature_importance.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_panel(ROOT / args.dataset)
    logistic_metrics = fit_logistic(df, output_dir)
    predictive_metrics = fit_predictive_models(
        df, output_dir, args.test_start_year, args.random_state)
    metrics = pd.DataFrame(logistic_metrics + predictive_metrics)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    (output_dir / "run_summary.json").write_text(json.dumps({
        "dataset": str(ROOT / args.dataset),
        "rows": int(len(df)),
        "test_start_year": args.test_start_year,
        "random_state": args.random_state,
        "logistic_models": list(MODEL_FEATURES),
        "predictive_features": MODEL_FEATURES["M4"],
        "note": "Logistic models use cluster-robust standard errors by dyad. "
                "Tree models use complete-case data and a fixed time-ordered split; "
                "SHAP values describe prediction, not causal effects.",
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nAnalysis outputs: {output_dir}")


if __name__ == "__main__":
    main()
