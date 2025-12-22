import pandas as pd
from src.preprocessing import get_feature_columns, build_preprocessor

def test_preprocessor_output_shape():
    df = pd.DataFrame({
        "age": [25, 45, 35],
        "income": [50000, 60000, 55000],
        "job": ["a", "b", "a"],
        "default": [0, 1, 0],
    })

    num_cols, cat_cols = get_feature_columns(df, target_col="default")
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X = df.drop(columns=["default"])
    X_p = preprocessor.fit_transform(X)

    assert X_p.shape[0] == df.shape[0]
