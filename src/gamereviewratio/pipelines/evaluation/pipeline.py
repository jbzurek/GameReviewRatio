from kedro.pipeline import Pipeline, node
from .nodes import (
    load_raw,
    basic_clean,
    split_data,
    train_baseline,
    evaluate,
    train_autogluon,
    evaluate_autogluon,
    choose_best_model,
    log_ag_metrics,
    log_baseline_metrics,
)


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(load_raw, "raw_data", "raw_df", name="load_raw"),
            node(
                basic_clean,
                ["raw_df", "params:clean", "params:target"],
                "clean_df",
                name="basic_clean",
            ),
            node(
                split_data,
                ["clean_df", "params:target", "params:split"],
                ["x_train", "x_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                train_autogluon,
                ["x_train", "y_train", "params:autogluon"],
                "ag_model",
                name="train_autogluon",
            ),
            node(
                evaluate_autogluon,
                ["ag_model", "x_test", "y_test"],
                "ag_metrics_local",
                name="evaluate_autogluon",
            ),
            node(
                log_ag_metrics,
                "ag_metrics_local",
                "ag_metrics",
                name="log_ag_metrics",
            ),
            node(
                train_baseline,
                ["x_train", "y_train", "params:model"],
                "baseline_model",
                name="train_baseline",
            ),
            node(
                evaluate,
                ["baseline_model", "x_test", "y_test"],
                "metrics_baseline_local",
                name="evaluate",
            ),
            node(
                log_baseline_metrics,
                "metrics_baseline_local",
                "metrics_baseline",
                name="log_baseline_metrics",
            ),
            node(
                choose_best_model,
                ["ag_metrics_local", "metrics_baseline_local"],
                "best_model_name",
                name="choose_best_model",
            ),
        ]
    )
