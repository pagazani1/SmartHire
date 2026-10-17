import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import config

def train_logistic_regression(X_train, y_train, force_retrain: bool = False) -> LogisticRegression:
    """
    Trains or loads Logistic Regression model with balanced class weights.
    """
    if not force_retrain and config.LOGISTIC_REGRESSION_PATH.exists():
        print(f"[MODELS] Loading cached Logistic Regression model from '{config.LOGISTIC_REGRESSION_PATH}'...")
        model = joblib.load(config.LOGISTIC_REGRESSION_PATH)
        return model

    print("[MODELS] Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=config.LOGREG_MAX_ITER,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)
    joblib.dump(model, config.LOGISTIC_REGRESSION_PATH)
    print(f"[MODELS] Saved Logistic Regression model to '{config.LOGISTIC_REGRESSION_PATH}'")
    return model

def train_linear_svc(X_train, y_train, force_retrain: bool = False) -> LinearSVC:
    """
    Trains or loads LinearSVC model with balanced class weights.
    """
    if not force_retrain and config.LINEAR_SVC_PATH.exists():
        print(f"[MODELS] Loading cached LinearSVC model from '{config.LINEAR_SVC_PATH}'...")
        model = joblib.load(config.LINEAR_SVC_PATH)
        return model

    print("[MODELS] Training LinearSVC model...")
    model = LinearSVC(
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)
    joblib.dump(model, config.LINEAR_SVC_PATH)
    print(f"[MODELS] Saved LinearSVC model to '{config.LINEAR_SVC_PATH}'")
    return model
