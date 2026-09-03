from typing import Tuple
import pandas as pd

def split_features_target(
    train: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split training data into features and target."""

    X = train.drop(columns=[target])
    y = train[target]

    return X, y