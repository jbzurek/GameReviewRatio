"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 1.0.0

"""
from kedro.pipeline import node, Pipeline

from .nodes import load_raw, basic_clean, split_data, train_baseline, evaluate


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_raw,
                inputs="raw_data",
                outputs="raw_df",
                name="load_raw",
            ),
            node(
                func=basic_clean,
                inputs=["raw_df", "params:clean", "params:target"],
                outputs="clean_df",
                name="basic_clean",
            ),
            node(
                func=split_data,
                inputs=["clean_df", "params:target", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=train_baseline,
                inputs=["X_train", "y_train", "params:model"],
                outputs="baseline_model",
                name="train_baseline",
            ),
            node(
                func=evaluate,
                inputs=["baseline_model", "X_test", "y_test"],
                outputs="rf_metrics",
                name="evaluate",
            ),
        ]
    )
