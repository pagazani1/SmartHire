import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import config

def evaluate_baseline_level0(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Evaluates Level 0 Majority Class Baseline.
    Predicts majority training class for all test instances.
    """
    majority_class = train_df["label"].mode()[0]
    y_test = test_df["label"]
    y_pred = [majority_class] * len(test_df)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "="*50)
    print("Level 0 - Majority Class Baseline")
    print("="*50)
    print(f"Majority Class: {majority_class}")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"model": "Level 0 - Majority", "accuracy": acc, "macro_f1": f1}

def evaluate_baseline_level1(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Evaluates Level 1 Domain Heuristic Baseline.
    Predicts domain-specific majority class.
    """
    majority_class = train_df["label"].mode()[0]
    level1_mapping = train_df.groupby("resume_domain")["label"].agg(lambda x: x.mode()[0])

    y_test = test_df["label"]
    y_pred = test_df["resume_domain"].map(level1_mapping).fillna(majority_class)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "="*50)
    print("Level 1 - Domain Heuristic Baseline")
    print("="*50)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"model": "Level 1 - Resume Domain", "accuracy": acc, "macro_f1": f1}

def evaluate_model(model, X_test, y_test, model_name: str, save_cm: bool = True) -> dict:
    """
    Evaluates a trained classifier on test data and plots confusion matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "="*50)
    print(f"Model Evaluation: {model_name}")
    print("="*50)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    labels = ["Good Fit", "Potential Fit", "No Fit"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    if save_cm:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues")
        plt.title(f"{model_name} - Confusion Matrix")
        save_path = config.MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_cm.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[EVALUATION] Confusion matrix saved to '{save_path}'")

    return {"model": model_name, "accuracy": acc, "macro_f1": f1}

def run_cross_validation(X_train, y_train: pd.Series, sample_size: int = 20000, cv_folds: int = 3):
    """
    Executes 3-Fold Stratified Cross-Validation on a representative training sample.
    Slices X_train sparse matrix directly to save memory.
    """
    print("\n" + "="*50)
    print(f"Running {cv_folds}-Fold Cross-Validation (Sample Size: {min(sample_size, X_train.shape[0])})")
    print("="*50)

    if X_train.shape[0] > sample_size:
        sample_indices = y_train.sample(n=sample_size, random_state=config.RANDOM_STATE).index
        # Get positional integer indices
        pos_indices = y_train.index.get_indexer(sample_indices)
        X_cv = X_train[pos_indices]
        y_cv = y_train.iloc[pos_indices]
    else:
        X_cv = X_train
        y_cv = y_train

    print("Sample label distribution:")
    print(y_cv.value_counts())

    # Logistic Regression CV
    lr = LogisticRegression(max_iter=config.LOGREG_MAX_ITER, class_weight=config.CLASS_WEIGHT, random_state=config.RANDOM_STATE)
    lr_scores = cross_val_score(lr, X_cv, y_cv, cv=cv_folds, scoring="f1_macro")
    print(f"\nLogistic Regression {cv_folds}-Fold Macro-F1 Scores: {lr_scores}")
    print(f"Mean: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")

    # LinearSVC CV
    svc = LinearSVC(class_weight=config.CLASS_WEIGHT, random_state=config.RANDOM_STATE)
    svc_scores = cross_val_score(svc, X_cv, y_cv, cv=cv_folds, scoring="f1_macro")
    print(f"\nLinearSVC {cv_folds}-Fold Macro-F1 Scores: {svc_scores}")
    print(f"Mean: {svc_scores.mean():.4f} +/- {svc_scores.std():.4f}")

    return {
        "lr_cv_mean": lr_scores.mean(),
        "lr_cv_std": lr_scores.std(),
        "svc_cv_mean": svc_scores.mean(),
        "svc_cv_std": svc_scores.std()
    }
