import pandas as pd
import numpy as np

def encode_categorical_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Encodes categorical variables using one-hot encoding with mandatory inputs and strict column validation."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("A valid pandas DataFrame must be provided for 'df'.")
    if cat_cols is None or not isinstance(cat_cols, list):
        raise ValueError("A mandatory list of categorical columns must be provided for 'cat_cols'.")
    
    # Check if all specified categorical columns exist in the DataFrame to prevent silent failures
    missing_cols = [col for col in cat_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following categorical columns do not exist in the DataFrame: {missing_cols}")
        
    df_out = df.copy()
    if cat_cols:
        df_out = pd.get_dummies(df_out, columns=cat_cols, drop_first=True)
        
    return df_out



def engineer_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates domain-specific interaction features for utility infrastructure risk."""
    df_out = df.copy()
    
    # Example domain interactions if columns exist
    if 'Voltage' in df_out.columns and 'Power Load' in df_out.columns:
        df_out['voltage_load_interaction'] = df_out['Voltage'] * df_out['Power Load']
        
    if 'Wind Speed' in df_out.columns and 'Age' in df_out.columns:
        df_out['environmental_stress_index'] = df_out['Wind Speed'] * df_out['Age']
        
    return df_out


def prepare_feature_matrix(df: pd.DataFrame, target_col: str, exclude_cols: list = None) -> tuple:
    """Separates the DataFrame into feature matrix X and target vector y, requiring explicit inputs and supporting optional exclusions."""
    if df is None:
        raise ValueError("DataFrame input is required.")
    if not target_col:
        raise ValueError("Target column name must be provided.")
        
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
        
    # Build exclusion set dynamically
    exclude_list = [target_col]
    if exclude_cols is not None:
        exclude_list.extend(list(exclude_cols))
        
    all_exclude = set(exclude_list)
    
    feature_cols = [col for col in df.columns if col not in all_exclude]
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y