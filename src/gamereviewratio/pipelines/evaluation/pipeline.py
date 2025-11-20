from kedro.pipeline import Pipeline, node
from .nodes import (
    load_raw,
    basic_clean,
    split_data,
    train_baseline,
    evaluate,
    train_autogluon,
    evaluate_autogluon,
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
                ["ag_model", "x_test", "y_test", "params:autogluon"],
                "ag_metrics",
                name="evaluate_autogluon",
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
                "metrics_baseline",
                name="evaluate",
            ),
        ]
    )
