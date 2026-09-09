from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preprocessing.preprocessing import create_svm_preprocessor


def build_svm(
    numeric_features: list[str],
    categorical_features: list[str],
    **params,
) -> Pipeline:
    """
    Build an SVM classification pipeline with preprocessing.
    """

    default_params = {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "random_state": 42,
    }

    default_params.update(params)

    preprocessor = create_svm_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model = SVC(**default_params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )