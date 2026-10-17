import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Config
HF_DATASET_NAME = "med2425/resume-job-fit-merged-v1"
RAW_TRAIN_PATH = RAW_DATA_DIR / "raw_train.parquet"
RAW_TEST_PATH = RAW_DATA_DIR / "raw_test.parquet"

# Processed Data Paths
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "clean_train.parquet"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "clean_test.parquet"

# Feature Extractor & Model Artifact Paths
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
X_TRAIN_SPARSE_PATH = MODELS_DIR / "X_train.npz"
X_TEST_SPARSE_PATH = MODELS_DIR / "X_test.npz"
LOGISTIC_REGRESSION_PATH = MODELS_DIR / "logistic_regression.joblib"
LINEAR_SVC_PATH = MODELS_DIR / "linear_svc.joblib"

# Model Hyperparameters
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
LOGREG_MAX_ITER = 1000
