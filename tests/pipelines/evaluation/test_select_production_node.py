from gamereviewratio.pipelines.evaluation.nodes import select_production_model


def test_select_prefers_lower_rmse():
    ag = {"rmse": 0.10, "inference_time_s": 0.10}
    base = {"rmse": 0.12, "inference_time_s": 0.05}
    info = select_production_model(
        ag,
        base,
        {"primary_metric": "rmse", "higher_is_better": False, "enable_wandb": False},
    )
    assert info["chosen_model"] == "ag_model"


def test_select_tie_uses_inference_time():
    ag = {"rmse": 0.10, "inference_time_s": 0.10}
    base = {"rmse": 0.10, "inference_time_s": 0.05}
    info = select_production_model(
        ag,
        base,
        {"primary_metric": "rmse", "higher_is_better": False, "enable_wandb": False},
    )
    assert info["chosen_model"] == "model_baseline"  # niższy czas


def test_select_records_data_version():
    ag = {"rmse": 0.11, "inference_time_s": 0.06}
    base = {"rmse": 0.13, "inference_time_s": 0.04}
    info = select_production_model(
        ag,
        base,
        {"primary_metric": "rmse", "higher_is_better": False, "enable_wandb": False},
    )
    assert "data_version" in info
