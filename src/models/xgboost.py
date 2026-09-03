from xgboost import XGBClassifier

def build_xgboost(**params):
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "auc",
    }

    default_params.update(params)

    return XGBClassifier(**default_params)