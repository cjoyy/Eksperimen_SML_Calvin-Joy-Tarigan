"""
Preprocessing Automation for Credit Card Fraud Detection Dataset.
This script replicates the preprocessing steps from the notebook
Eksperimen_Calvin-Joy-Tarigan.ipynb as a reusable standalone script.

Usage:
    python automate_Calvin-Joy-Tarigan.py

Output:
    Writes processed CSV files to creditcard_preprocessing/.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def preprocess_data(input_path: str, output_path: str) -> None:
    """Load raw data, apply cleaning/scaling/split/SMOTE,
    save processed train/test sets to output_path.

    Args:
        input_path: Path to raw creditcard.csv
        output_path: Directory to save processed files
    """
    # --- 1. Load data ---
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- 2. Handle duplicates ---
    dups_before = df.duplicated().sum()
    df_clean = df.drop_duplicates()
    logger.info(f"Duplicates removed: {dups_before} -> {df_clean.duplicated().sum()}")
    logger.info(f"Shape after dedup: {df_clean.shape}")

    # --- 3. Scale Time and Amount ---
    X = df_clean.copy()

    scaler_time = StandardScaler()
    X['Time'] = scaler_time.fit_transform(X[['Time']])
    logger.info(f"Time scaled via StandardScaler (mean~{X['Time'].mean():.4f})")

    scaler_amount = RobustScaler()
    X['Amount'] = scaler_amount.fit_transform(X[['Amount']])
    logger.info(f"Amount scaled via RobustScaler (median~{X['Amount'].median():.4f})")

    # V1-V28 are already PCA-normalized, no scaling needed
    logger.info("V1-V28 left unchanged (already PCA-normalized)")

    # --- 4. Separate features and target ---
    y = X['Class']
    X_features = X.drop('Class', axis=1)
    logger.info(f"Features: {X_features.shape[1]}, Target fraud rate: {y.mean()*100:.4f}%")

    # --- 5. Stratified train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Train fraud%: {y_train.mean()*100:.4f}%, Test fraud%: {y_test.mean()*100:.4f}%")

    # --- 6. SMOTE on training set only ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    train_res_counts = y_train_res.value_counts()
    logger.info(f"After SMOTE: {train_res_counts.to_dict()}, "
                f"ratio {train_res_counts[0]/train_res_counts[1]:.2f}:1")

    # --- 7. Save all outputs ---
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Saving files to {output_path}")

    pd.DataFrame(X_train, columns=X_features.columns).to_csv(
        os.path.join(output_path, 'X_train.csv'), index=False)
    pd.DataFrame(X_test, columns=X_features.columns).to_csv(
        os.path.join(output_path, 'X_test.csv'), index=False)
    pd.DataFrame(X_train_res, columns=X_features.columns).to_csv(
        os.path.join(output_path, 'X_train_resampled.csv'), index=False)
    pd.DataFrame(y_train, columns=['Class']).to_csv(
        os.path.join(output_path, 'y_train.csv'), index=False)
    pd.DataFrame(y_test, columns=['Class']).to_csv(
        os.path.join(output_path, 'y_test.csv'), index=False)
    pd.DataFrame(y_train_res, columns=['Class']).to_csv(
        os.path.join(output_path, 'y_train_resampled.csv'), index=False)

    logger.info(f"All files saved successfully.")

    # Quick verification
    for fname in ['X_train.csv', 'X_test.csv', 'X_train_resampled.csv',
                  'y_train.csv', 'y_test.csv', 'y_train_resampled.csv']:
        fpath = os.path.join(output_path, fname)
        nrows = pd.read_csv(fpath).shape[0]
        logger.info(f"  Verified: {fname} -> {nrows} rows")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(os.path.dirname(script_dir), 'creditcard_raw', 'creditcard.csv')
    default_output = os.path.join(script_dir, 'creditcard_preprocessing')

    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    logger.info("=== Credit Card Fraud Preprocessing Pipeline ===")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    preprocess_data(input_path, output_path)
    logger.info("=== Pipeline completed successfully ===")
