import pandas as pd
from gamereviewratio.pipelines.evaluation.nodes import split_data


# testuje poprawność podziału danych
def test_split_returns_y_as_dataframe_and_no_target_in_x():
    df = pd.DataFrame(
        {"f1": [1, 2, 3, 4, 5], "pct_pos_total": [0.1, 0.2, 0.3, 0.4, 0.5]}
    )
    X_train, X_test, y_train, y_test = split_data(
        df, "pct_pos_total", {"test_size": 0.4, "random_state": 42}
    )
    assert "pct_pos_total" not in X_train.columns
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert list(y_train.columns) == ["pct_pos_total"]
    assert list(y_test.columns) == ["pct_pos_total"]


# testuje obsługę błędu
def test_split_raises_if_target_missing():
    df = pd.DataFrame({"f1": [1, 2, 3]})
    try:
        split_data(df, "pct_pos_total", {"test_size": 0.2, "random_state": 42})
        assert False
    except ValueError as e:
        assert "Target 'pct_pos_total' not in dataframe" in str(e)
