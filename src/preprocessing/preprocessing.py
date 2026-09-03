from typing import Tuple
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def split_features_target(
    train: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split training data into features and binary target."""

    X = train.drop(columns=[target])

    y = train[target].map({
        "No": 0,
        "Yes": 1,
    })

    return X, y

def get_feature_types(
    X: DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical features.

    Parameters
    ----------
    X : DataFrame
        Feature dataset.

    Returns
    -------
    tuple[list[str], list[str]]
        Numerical and categorical feature names.
    """
    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features

def create_one_hot_preprocessor(
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Create a preprocessor that one-hot encodes categorical features
    and leaves numerical features unchanged.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )