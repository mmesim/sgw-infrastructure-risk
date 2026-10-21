import os
import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"The dataset at {file_path} is empty.")
    return df


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in numerical columns with median and categorical with mode."""
    df_clean = df.copy()
    
    # Impute missing numerical values using median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Impute missing categorical values using mode
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    return df_clean


def create_binary_target(df: pd.DataFrame, target_col: str = 'Component Health') -> pd.DataFrame:
    """Maps critical component health states ('Faulty', 'Overheated') to 1, and 'Normal' to 0."""
    df_out = df.copy()
    if target_col not in df_out.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
        
    df_out['is_critical'] = df_out[target_col].map({'Normal': 0, 'Faulty': 1, 'Overheated': 1}).fillna(0).astype(int)
    return df_out