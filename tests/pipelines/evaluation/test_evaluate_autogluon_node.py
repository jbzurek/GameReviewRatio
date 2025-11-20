import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

from gamereviewratio.pipelines.evaluation.nodes import evaluate_autogluon


class DummyPredictor:
    def predict(self, x):
        # Zwracamy same zera aby RMSE było łatwo policzalne
        return np.zeros(len(x))


def test_evaluate_autogluon_returns_metrics_with_keys_and_non_negative_values():
    x_test = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_test = pd.DataFrame({"pct_pos_total": [0.5, 0.0, 1.0]})
    predictor = DummyPredictor()

    metrics = evaluate_autogluon(predictor, x_test, y_test, ag_params={})

    assert set(metrics.keys()) == {"rmse", "inference_time_s"}
    assert metrics["rmse"] >= 0
    assert metrics["inference_time_s"] >= 0
    # Dla zera jako predykcji RMSE ~ sqrt(mean(y^2)) – sprawdzamy że jest zgodne w przybliżeniu
    expected_rmse = float(np.sqrt(np.mean(y_test["pct_pos_total"] ** 2)))
    assert abs(metrics["rmse"] - expected_rmse) < 1e-9


def test_kedro_run_creates_model_directory(tmp_path):
    # Uruchamiamy fragment pipeline od nodu train_baseline (szybciej niż pełny run z AutoGluon)
    result = subprocess.run(
        ["kedro", "run", "--from-nodes", "train_baseline"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"kedro run zakończył się błędem: {result.stderr}"
    assert Path("data/06_models").exists(), "Katalog z modelami nie został utworzony"
