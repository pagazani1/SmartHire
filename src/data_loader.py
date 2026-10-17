import os
import pandas as pd
from datasets import load_dataset
import config

def load_data(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw train and test DataFrames.
    Checks for local cached Parquet files in data/raw/.
    If cached files are missing or force_download is True, fetches from Hugging Face.
    """
    if not force_download and config.RAW_TRAIN_PATH.exists() and config.RAW_TEST_PATH.exists():
        print(f"[DATA LOADER] Loading cached raw data from '{config.RAW_DATA_DIR}'...")
        train_df = pd.read_parquet(config.RAW_TRAIN_PATH)
        test_df = pd.read_parquet(config.RAW_TEST_PATH)
        print(f"[DATA LOADER] Loaded Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df

    print(f"[DATA LOADER] Fetching dataset '{config.HF_DATASET_NAME}' from Hugging Face Hub...")
    hf_dataset = load_dataset(config.HF_DATASET_NAME)
    
    train_df = hf_dataset["train"].to_pandas()
    test_df = hf_dataset["test"].to_pandas()

    # Cache raw datasets locally
    print(f"[DATA LOADER] Caching raw data to '{config.RAW_DATA_DIR}'...")
    train_df.to_parquet(config.RAW_TRAIN_PATH, index=False)
    test_df.to_parquet(config.RAW_TEST_PATH, index=False)
    print(f"[DATA LOADER] Cached Train: {train_df.shape}, Test: {test_df.shape}")

    return train_df, test_df

if __name__ == "__main__":
    load_data()
