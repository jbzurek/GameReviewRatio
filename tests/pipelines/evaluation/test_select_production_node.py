from gamereviewratio.pipelines.evaluation.nodes import choose_best_model


# testuje czy wybierany jest model z niższym rmse
def test_select_prefers_lower_rmse():
    ag = {"rmse": 0.10, "inference_time_s": 0.10}
    base = {"rmse": 0.12, "inference_time_s": 0.05}

    chosen = choose_best_model(ag, base)

    assert chosen == "ag_model"
