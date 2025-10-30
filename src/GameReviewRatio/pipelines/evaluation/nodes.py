"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 1.0.0

"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer

def load_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df

def _parse_list_cell(x: Any) -> List[str]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list):
                return val
        except Exception:
            return [s.strip() for s in x.split(",") if s.strip()]
    return []

def basic_clean(
    df: pd.DataFrame,
    clean: Dict[str, Any],
    target: str,
) -> pd.DataFrame:
    df = df.copy()

    threshold = float(clean.get("threshold_missing", 0.3))
    bin_flag_cols: List[str] = list(clean.get("bin_flag_cols", []))
    platform_cols: List[str] = list(clean.get("platform_cols", []))
    drop_cols: List[str] = list(clean.get("drop_cols", []))
    mlb_cols: List[str] = list(clean.get("mlb_cols", []))
    top_n: int = int(clean.get("top_n_labels", 50))

    to_drop = []
    for col in df.columns:
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    test_size = float(split.get("test_size", 0.2))
    random_state = int(split.get("random_state", 42))

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataframe")

    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target])

    X = pd.get_dummies(X, drop_first=True)

    mask = y.notnull()
    X = X.loc[mask]
    y = y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    y_train_df = y_train.to_frame(name=target)
    y_test_df = y_test.to_frame(name=target)
    return X_train, X_test, y_train_df, y_test_df

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Dict[str, Any],
) -> RandomForestRegressor:
    params = {
        "random_state": model.get("random_state", 42),
        "n_estimators": model.get("n_estimators", 200),
        "n_jobs": model.get("n_jobs", -1),
    }
    mdl = RandomForestRegressor(**params)
    mdl.fit(X_train, y_train)
    return mdl

def evaluate(
    mdl: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    if isinstance(y_test, pd.DataFrame):
        y_true = pd.to_numeric(y_test.iloc[:, 0], errors="coerce")
    else:
        y_true = pd.to_numeric(y_test, errors="coerce")

    y_pred = mdl.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return pd.DataFrame({"rmse": [rmse]})
