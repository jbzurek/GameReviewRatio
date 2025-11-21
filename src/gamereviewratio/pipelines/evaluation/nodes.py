"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 1.0.0
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import wandb
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from autogluon.tabular import TabularPredictor


def load_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df


def _parse_list_cell(x: Any) -> List[str]:
    if isinstance(x, list):
        return x

    if isinstance(x, str):
        x_str = x.strip()
        if not x_str:
            return []

        try:
            val = ast.literal_eval(x_str)

            if isinstance(val, list):
                return val

            if isinstance(val, dict):
                return list(val.keys())

        except Exception:
            pass

        return [s.strip() for s in x_str.split(",") if s.strip()]

    return []


def basic_clean(df: pd.DataFrame, clean: Dict[str, Any], target: str) -> pd.DataFrame:
    df = df.copy()

    threshold = float(clean.get("threshold_missing", 0.3))
    bin_flag_cols: List[str] = list(clean.get("bin_flag_cols", []))
    platform_cols: List[str] = list(clean.get("platform_cols", []))
    drop_cols: List[str] = list(clean.get("drop_cols", []))
    mlb_cols: List[str] = list(clean.get("mlb_cols", []))
    top_n: int = int(clean.get("top_n_labels", 50))

    to_drop: List[str] = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].isnull().mean() > threshold:
            to_drop.append(col)
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors="ignore")

    for col in ["positive", "negative"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
        df["release_month"] = df["release_date"].dt.month
        df.drop(columns=["release_date"], inplace=True)

    for col in bin_flag_cols:
        if col in df.columns:
            df[f"has_{col}"] = df[col].notnull().astype(int)
            df.drop(columns=[col], inplace=True)

    for col in platform_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    for col in mlb_cols:
        if col not in df.columns:
            continue

        series = df[col].apply(_parse_list_cell)
        counts: Dict[str, int] = {}
        for lst in series.dropna():
            for lab in lst:
                counts[lab] = counts.get(lab, 0) + 1

        top_labels = sorted(counts, key=counts.get, reverse=True)[:top_n]
        top_set = set(top_labels)

        if not top_set:
            df.drop(columns=[col], inplace=True)
            continue

        series_top = series.apply(lambda lst: [x for x in lst if x in top_set])
        mlb = MultiLabelBinarizer(classes=sorted(top_set))
        encoded = pd.DataFrame(
            mlb.fit_transform(series_top),
            columns=[f"{col}_{c}" for c in mlb.classes_],
            index=df.index,
        )
        df = pd.concat([df.drop(columns=[col]), encoded], axis=1)

    return df


def split_data(
    df: pd.DataFrame,
    target: str,
    split: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = float(split.get("test_size", 0.2))
    random_state = int(split.get("random_state", 42))

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataframe")

    y = pd.to_numeric(df[target], errors="coerce")
    x = df.drop(columns=[target])
    x = pd.get_dummies(x, drop_first=True)

    mask = y.notnull()
    x = x.loc[mask]
    y = y.loc[mask]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    y_train_df = y_train.to_frame(name=target)
    y_test_df = y_test.to_frame(name=target)
    return x_train, x_test, y_train_df, y_test_df


def train_baseline(
    x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame, model: dict
) -> RandomForestRegressor:
    wandb.init(
        project="gamereviewratio",
        job_type="train-baseline",
        reinit=True,
        config=model,
    )

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()
    y_train = y_train.to_numpy().ravel()

    params = {
        "random_state": model.get("random_state", 42),
        "n_estimators": model.get("n_estimators", 200),
        "n_jobs": model.get("n_jobs", -1),
    }

    mdl = RandomForestRegressor(**params)
    start_train = time.time()
    mdl.fit(x_train, y_train)
    train_time_s = time.time() - start_train

    model_path = Path("data/06_models/model_baseline.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl, model_path)

    try:
        wandb.log({"train_time_s": train_time_s})
    except Exception:
        pass

    try:
        art = wandb.Artifact("model_baseline", type="model")
        art.add_file(str(model_path))
        wandb.log_artifact(art, aliases=["candidate"])
    except Exception:
        pass

    return mdl


def evaluate(
    mdl: RandomForestRegressor, x_test: pd.DataFrame, y_test: pd.DataFrame | pd.Series
) -> dict:
    if isinstance(y_test, pd.DataFrame):
        y_true = y_test.iloc[:, 0]
    else:
        y_true = y_test

    y_true = pd.to_numeric(y_true, errors="coerce")
    mask = y_true.notnull()
    y_true = y_true[mask]
    x_eval = x_test.loc[mask]

    start_pred = time.time()
    y_pred = mdl.predict(x_eval)
    inference_time_s = time.time() - start_pred
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {"rmse": rmse, "inference_time_s": inference_time_s}

    try:
        Path("data/09_tracking").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        wandb.log(metrics)
    except Exception:
        pass

    try:
        wandb.finish()
    except Exception:
        pass

    return metrics


def train_autogluon(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    ag_params: dict,
) -> TabularPredictor:
    label = ag_params.get("label")
    problem_type = ag_params.get("problem_type", "regression")
    eval_metric = ag_params.get("eval_metric", "rmse")
    time_limit = int(ag_params.get("time_limit", 60))
    presets = ag_params.get("presets", "medium_quality_faster_train")
    random_state = int(ag_params.get("random_state", 42))

    if isinstance(y_train, pd.DataFrame):
        y_series = y_train.iloc[:, 0]
    else:
        y_series = y_train

    train_df = x_train.copy()
    train_df[label] = pd.to_numeric(y_series, errors="coerce")

    train_df = train_df.dropna(subset=[label])

    wandb_config = {
        "label": label,
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "time_limit": time_limit,
        "presets": presets,
        "random_state": random_state,
    }
    wandb_config.update(
        {f"feature_{col}": str(train_df[col].dtype) for col in x_train.columns}
    )

    wandb.init(
        project="gamereviewratio",
        job_type="ag-train",
        reinit=True,
        config=wandb_config,
    )

    start_time = time.time()

    predictor = None
    try:
        predictor = TabularPredictor(
            label=label,
            problem_type=problem_type,
            eval_metric=eval_metric,
            verbosity=2,
        ).fit(
            train_data=train_df,
            time_limit=time_limit,
            presets=presets,
            hyperparameters={
                "GBM": {},  # LightGBM
                "RF": [{"criterion": "squared_error"}],
            },
            raise_on_no_models_fitted=False,
        )
    except Exception:
        predictor = None

    train_time_s = time.time() - start_time

    if predictor is not None:
        output_path = Path("data/06_models")
        output_path.mkdir(parents=True, exist_ok=True)
        predictor.save(str(output_path))
        pkl_path = Path("data/06_models/ag_production.pkl")
        joblib.dump(predictor, pkl_path)

    try:
        wandb.log({"train_time_s": train_time_s})
    except Exception:
        pass

    if predictor is not None:
        try:
            art = wandb.Artifact("ag_model", type="model")
            art.add_file(str(pkl_path))
            wandb.log_artifact(art, aliases=["candidate"])
        except Exception:
            pass

    return predictor


def evaluate_autogluon(
    predictor: TabularPredictor,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
    ag_params: dict,
) -> Dict[str, float]:
    if isinstance(y_test, pd.DataFrame):
        y_true = y_test.iloc[:, 0]
    else:
        y_true = y_test

    y_true = pd.to_numeric(y_true, errors="coerce")
    mask = y_true.notnull()
    y_true = y_true[mask]
    x_eval = x_test.loc[mask]

    start_pred = time.time()
    y_pred = predictor.predict(x_eval)
    inference_time_s = time.time() - start_pred

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {"rmse": rmse, "inference_time_s": inference_time_s}

    try:
        Path("data/09_tracking").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        wandb.log(metrics)
    except Exception:
        pass

    try:
        wandb.finish()
    except Exception:
        pass

    return metrics


def select_production_model(
    ag_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    selection_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    params = selection_params or {}
    primary_metric = params.get("primary_metric", "rmse")
    tie_breaker = params.get("tie_breaker", "inference_time_s")
    higher_is_better = bool(params.get("higher_is_better", False))
    enable_wandb = bool(params.get("enable_wandb", True))
    raw_dataset_path = params.get("raw_dataset_path", "data/01_raw/sample_100.csv")

    def _safe_val(d: Dict[str, Any], key: str) -> float:
        v = d.get(key)
        try:
            return float(v)
        except Exception:
            return float("inf") if not higher_is_better else float("-inf")

    ag_val = _safe_val(ag_metrics, primary_metric)
    base_val = _safe_val(baseline_metrics, primary_metric)

    def _better(a: float, b: float) -> bool:
        return a > b if higher_is_better else a < b

    if _better(ag_val, base_val):
        chosen = "ag_model"
    elif _better(base_val, ag_val):
        chosen = "model_baseline"
    else:
        ag_tb = _safe_val(ag_metrics, tie_breaker)
        base_tb = _safe_val(baseline_metrics, tie_breaker)
        chosen = "ag_model" if ag_tb <= base_tb else "model_baseline"
    try:
        raw_path = Path(raw_dataset_path)
        if raw_path.exists():
            data_version = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(raw_path.stat().st_mtime)
            )
        else:
            data_version = "missing"
    except Exception:
        data_version = "error"

    info = {
        "primary_metric": primary_metric,
        "tie_breaker": tie_breaker,
        "higher_is_better": higher_is_better,
        "ag_metrics": ag_metrics,
        "baseline_metrics": baseline_metrics,
        "chosen_model": chosen,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_version": data_version,
    }

    if enable_wandb:
        try:
            wandb.init(
                project="gamereviewratio",
                job_type="select-production",
                reinit=True,
                config={
                    "primary_metric": primary_metric,
                    "tie_breaker": tie_breaker,
                    "higher_is_better": higher_is_better,
                    "data_version": data_version,
                },
            )
            from wandb import Api

            api = Api()
            artifact_name = "ag_model" if chosen == "ag_model" else "model_baseline"
            candidate_ref = f"{artifact_name}:candidate"
            try:
                art = api.artifact(candidate_ref)
            except Exception:
                art = api.artifact(f"{artifact_name}:latest")
            if art is not None and "production" not in art.aliases:
                art.aliases.append("production")
                art.save()
            wandb.log({"chosen_model": chosen})
            wandb.finish()
        except Exception:
            pass

    return info
