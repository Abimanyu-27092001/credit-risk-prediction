import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE


def apply_smote(X_train, y_train, random_state=42):
    """
    Applies SMOTE to balance the training data.
    Returns resampled X and y.
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def train_lightgbm(X, y, random_state=42):
    """
    Train LightGBM with CV hyperparameter tuning.
    Returns best model and best parameters.
    """

    param_grid = {
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_model = lgb.LGBMClassifier(
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
    )

    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X, y)

    best_model = search.best_estimator_
    best_params = search.best_params_

    return best_model, best_params

