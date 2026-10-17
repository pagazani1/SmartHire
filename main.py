import argparse
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data, get_tfidf_features
from src.models import train_logistic_regression, train_linear_svc
from src.evaluate import (
    evaluate_baseline_level0,
    evaluate_baseline_level1,
    evaluate_model,
    run_cross_validation
)

def run_pipeline(
    force_download: bool = False,
    force_reprocess: bool = False,
    force_revectorize: bool = False,
    force_retrain: bool = False,
    skip_cv: bool = False
):
    print("="*60)
    print("SmartHire - ML Pipeline Execution")
    print("="*60)

    # 1. Load Data
    train_df, test_df = load_data(force_download=force_download)

    # 2. Preprocess Text
    train_df, test_df = preprocess_data(train_df, test_df, force_reprocess=force_reprocess)

    # 3. Extract TF-IDF Features
    X_train, X_test, y_train, y_test, tfidf = get_tfidf_features(
        train_df, test_df, force_revectorize=force_revectorize
    )

    # 4. Evaluate Baselines
    results = []
    results.append(evaluate_baseline_level0(train_df, test_df))
    results.append(evaluate_baseline_level1(train_df, test_df))

    # 5. Train & Evaluate Models
    lr_model = train_logistic_regression(X_train, y_train, force_retrain=force_retrain)
    results.append(evaluate_model(lr_model, X_test, y_test, "Logistic Regression"))

    svc_model = train_linear_svc(X_train, y_train, force_retrain=force_retrain)
    results.append(evaluate_model(svc_model, X_test, y_test, "LinearSVC"))

    # 6. Model Comparison Summary
    summary_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))

    # 7. Cross-Validation (Fixed NameError & MemoryError)
    if not skip_cv:
        run_cross_validation(X_train, y_train, sample_size=20000, cv_folds=3)

    print("\n[SUCCESS] SmartHire ML Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartHire Persistent ML Pipeline")
    parser.add_argument("--force-download", action="store_true", help="Force dataset download from Hugging Face")
    parser.add_argument("--force-reprocess", action="store_true", help="Force text cleaning re-execution")
    parser.add_argument("--force-revectorize", action="store_true", help="Force TF-IDF vectorizer re-fitting")
    parser.add_argument("--force-retrain", action="store_true", help="Force re-training of ML models")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 3-fold cross validation")

    args = parser.parse_args()
    run_pipeline(
        force_download=args.force_download,
        force_reprocess=args.force_reprocess,
        force_revectorize=args.force_revectorize,
        force_retrain=args.force_retrain,
        skip_cv=args.skip_cv
    )
