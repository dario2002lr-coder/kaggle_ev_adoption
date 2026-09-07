from lightgbm import LGBMClassifier

def build_lightgbm(**params):

    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    default_params.update(params)

    return LGBMClassifier(**default_params)