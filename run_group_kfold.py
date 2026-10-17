import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import config

def run_group_kfold_experiment():
    print("="*65)
    print("SmartHire - GroupKFold Validation Experiment (Grouped by 'jd')")
    print("="*65)

    # 1. Load raw train data for 'jd' grouping column
    print(f"[LOAD] Loading training data from '{config.RAW_TRAIN_PATH}'...")
    train_df = pd.read_parquet(config.RAW_TRAIN_PATH)
    groups = train_df["jd"]
    y_train = train_df["label"]

    # 2. Load pre-computed TF-IDF sparse matrix
    print(f"[LOAD] Loading sparse feature matrix from '{config.X_TRAIN_SPARSE_PATH}'...")
    X_train = sp.load_npz(config.X_TRAIN_SPARSE_PATH)

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] Total Unique JDs: {groups.nunique()}")

    gkf = GroupKFold(n_splits=3)

    # Instantiate models
    lr_model = LogisticRegression(
        max_iter=config.LOGREG_MAX_ITER,
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE
    )

    svc_model = LinearSVC(
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_STATE
    )

    lr_fold_acc, lr_fold_f1 = [], []
    svc_fold_acc, svc_fold_f1 = [], []

    print("\nExecuting GroupKFold (3 Folds)...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups), 1):
        X_tr, y_tr = X_train[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train[val_idx], y_train.iloc[val_idx]
        val_jds = groups.iloc[val_idx].nunique()

        # Fit & Predict Logistic Regression
        lr_model.fit(X_tr, y_tr)
        lr_pred = lr_model.predict(X_va)
        lr_acc = accuracy_score(y_va, lr_pred)
        lr_f1 = f1_score(y_va, lr_pred, average="macro")
        lr_fold_acc.append(lr_acc)
        lr_fold_f1.append(lr_f1)

        # Fit & Predict LinearSVC
        svc_model.fit(X_tr, y_tr)
        svc_pred = svc_model.predict(X_va)
        svc_acc = accuracy_score(y_va, svc_pred)
        svc_f1 = f1_score(y_va, svc_pred, average="macro")
        svc_fold_acc.append(svc_acc)
        svc_fold_f1.append(svc_f1)

        print(f"\n--- Fold {fold} (Validation JDs: {val_jds}, Samples: {len(val_idx)}) ---")
        print(f"Logistic Regression -> Acc: {lr_acc:.6f} | Macro-F1: {lr_f1:.6f}")
        print(f"LinearSVC           -> Acc: {svc_acc:.6f} | Macro-F1: {svc_f1:.6f}")

    # Summary table
    print("\n" + "="*65)
    print("GROUPKFOLD (3-FOLD BY JD) EXPERIMENT RESULTS")
    print("="*65)

    results_data = {
        "Metric": [
            "Fold 1 Macro-F1",
            "Fold 2 Macro-F1",
            "Fold 3 Macro-F1",
            "Mean Macro-F1",
            "Std Macro-F1",
            "Mean Accuracy",
            "Std Accuracy"
        ],
        "Logistic Regression": [
            f"{lr_fold_f1[0]:.6f}",
            f"{lr_fold_f1[1]:.6f}",
            f"{lr_fold_f1[2]:.6f}",
            f"{np.mean(lr_fold_f1):.6f}",
            f"{np.std(lr_fold_f1):.6f}",
            f"{np.mean(lr_fold_acc):.6f}",
            f"{np.std(lr_fold_acc):.6f}"
        ],
        "LinearSVC": [
            f"{svc_fold_f1[0]:.6f}",
            f"{svc_fold_f1[1]:.6f}",
            f"{svc_fold_f1[2]:.6f}",
            f"{np.mean(svc_fold_f1):.6f}",
            f"{np.std(svc_fold_f1):.6f}",
            f"{np.mean(svc_fold_acc):.6f}",
            f"{np.std(svc_fold_acc):.6f}"
        ]
    }

    df_results = pd.DataFrame(results_data)
    print(df_results.to_string(index=False))
    print("="*65)

if __name__ == "__main__":
    run_group_kfold_experiment()
