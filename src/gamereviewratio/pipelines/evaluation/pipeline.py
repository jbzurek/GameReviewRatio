from kedro.pipeline import Pipeline, node
from .nodes import load_raw, basic_clean, split_data, train_baseline, evaluate


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
                ["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                train_baseline,
                ["X_train", "y_train", "params:model"],
                "baseline_model",
                name="train_baseline",
            ),
            node(
                evaluate,
                ["baseline_model", "X_test", "y_test"],
                "metrics_baseline",
                name="evaluate",
            ),
        ]
    )
