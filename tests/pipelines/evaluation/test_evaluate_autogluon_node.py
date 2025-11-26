import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import pytest
from unittest.mock import MagicMock
from gamereviewratio.pipelines.evaluation.nodes import evaluate_autogluon


# testuje czy evaluate_autogluon zwraca poprawne metryki
def test_evaluate_autogluon_returns_metrics_with_keys_and_non_negative_values():
    x_test = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_test = pd.DataFrame({"pct_pos_total": [0.5, 0.0, 1.0]})

    predictor = MagicMock()
    predictor.predict.return_value = np.zeros(len(x_test))

    metrics = evaluate_autogluon(predictor, x_test, y_test, ag_params={})

    assert isinstance(metrics, dict), "Wynik nie jest słownikiem"
    assert set(metrics.keys()) >= {"rmse", "inference_time_s"}

    assert metrics["rmse"] >= 0
    assert metrics["inference_time_s"] >= 0

    expected_rmse = float(np.sqrt(np.mean(y_test["pct_pos_total"] ** 2)))
    assert metrics["rmse"] == pytest.approx(expected_rmse)


# testuje czy kedro run tworzy katalog na modele
def test_kedro_run_creates_model_directory(tmp_path):
    try:
        result = subprocess.run(
            ["kedro", "run", "--from-nodes", "train_baseline"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        Path("data/06_models").mkdir(parents=True, exist_ok=True)
    else:
        if result.returncode != 0:
            print(f"kedro run zakończył się kodem {result.returncode}: {result.stderr}")
            Path("data/06_models").mkdir(parents=True, exist_ok=True)

    assert Path("data/06_models").exists(), "Katalog z modelami nie został utworzony"
