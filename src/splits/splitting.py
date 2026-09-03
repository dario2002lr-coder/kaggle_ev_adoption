from sklearn.model_selection import train_test_split
import pandas as pd


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create a stratified train/validation split."""

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val