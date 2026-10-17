import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import config

def clean_text(text: str) -> str:
    """
    Cleans raw text: lowercases, replaces double quotes, and normalizes extra whitespace.
    """
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, force_reprocess: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans resume and job description text and constructs combined_text feature column.
    Caches results to disk in data/processed/.
    """
    if not force_reprocess and config.PROCESSED_TRAIN_PATH.exists() and config.PROCESSED_TEST_PATH.exists():
        print(f"[PREPROCESSING] Loading cached clean data from '{config.PROCESSED_DATA_DIR}'...")
        train_df = pd.read_parquet(config.PROCESSED_TRAIN_PATH)
        test_df = pd.read_parquet(config.PROCESSED_TEST_PATH)
        print(f"[PREPROCESSING] Loaded Processed Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df

    print("[PREPROCESSING] Cleaning text fields (resume & jd)...")
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["resume_clean"] = train_df["resume"].apply(clean_text)
    train_df["jd_clean"] = train_df["jd"].apply(clean_text)
    train_df["combined_text"] = train_df["resume_clean"] + " " + train_df["jd_clean"]

    test_df["resume_clean"] = test_df["resume"].apply(clean_text)
    test_df["jd_clean"] = test_df["jd"].apply(clean_text)
    test_df["combined_text"] = test_df["resume_clean"] + " " + test_df["jd_clean"]

    # Select essential columns to optimize memory & file size
    cols_to_keep = ["combined_text", "label", "resume_domain", "jd_domain"]
    
    train_clean_df = train_df[cols_to_keep]
    test_clean_df = test_df[cols_to_keep]

    print(f"[PREPROCESSING] Saving optimized clean dataset to '{config.PROCESSED_DATA_DIR}'...")
    train_clean_df.to_parquet(config.PROCESSED_TRAIN_PATH, index=False)
    test_clean_df.to_parquet(config.PROCESSED_TEST_PATH, index=False)
    print("[PREPROCESSING] Clean dataset saved successfully.")

    return train_clean_df, test_clean_df

import scipy.sparse as sp

def get_tfidf_features(train_df: pd.DataFrame, test_df: pd.DataFrame, force_revectorize: bool = False):
    """
    Extracts TF-IDF features for training and testing datasets.
    Saves/loads sparse matrices (X_train.npz, X_test.npz) and fitted TfidfVectorizer.
    """
    if (
        not force_revectorize 
        and config.TFIDF_VECTORIZER_PATH.exists()
        and config.X_TRAIN_SPARSE_PATH.exists()
        and config.X_TEST_SPARSE_PATH.exists()
    ):
        print(f"[PREPROCESSING] Loading cached sparse matrices from '{config.MODELS_DIR}'...")
        X_train = sp.load_npz(config.X_TRAIN_SPARSE_PATH)
        X_test = sp.load_npz(config.X_TEST_SPARSE_PATH)
        tfidf = joblib.load(config.TFIDF_VECTORIZER_PATH)
        y_train = train_df["label"]
        y_test = test_df["label"]
        print(f"[PREPROCESSING] Loaded X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test, tfidf

    print("[PREPROCESSING] Fitting TfidfVectorizer on training set...")
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF
    )
    
    X_train = tfidf.fit_transform(train_df["combined_text"])
    X_test = tfidf.transform(test_df["combined_text"])

    print(f"[PREPROCESSING] Saving sparse matrices to '{config.MODELS_DIR}'...")
    sp.save_npz(config.X_TRAIN_SPARSE_PATH, X_train)
    sp.save_npz(config.X_TEST_SPARSE_PATH, X_test)
    joblib.dump(tfidf, config.TFIDF_VECTORIZER_PATH)
    print(f"[PREPROCESSING] Saved TfidfVectorizer and sparse feature matrices successfully.")

    y_train = train_df["label"]
    y_test = test_df["label"]

    print(f"[PREPROCESSING] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, tfidf
