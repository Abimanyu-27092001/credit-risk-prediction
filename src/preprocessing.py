from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_feature_columns(df, target_col):
    """
    Returns lists of numeric and categorical feature columns.
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    cat_cols = [c for c in df.columns if c not in num_cols + [target_col]]

    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    """
    Builds a ColumnTransformer for numeric + categorical features.
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor

