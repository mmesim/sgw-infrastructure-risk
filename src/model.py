import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_xgboost_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits data into train/test sets, scales numerical features post-split, 
    trains an XGBoost classifier, and returns the model, scaler, and evaluation metrics including confusion matrix.
    """
    if X is None or y is None:
        raise ValueError("Feature matrix X and target vector y must be provided.")
        
    # 1. Split data into training and test subsets (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. Normalize numerical features AFTER splitting to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames to preserve feature names for XGBoost
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    # 3. Initialize and train XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        random_state=random_state,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 4. Evaluate performance on the scaled test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, scaler, X_test_scaled, y_test, auc_score, report, conf_matrix


def generate_risk_scores(model, scaler, X: pd.DataFrame) -> np.ndarray:
    """Scales incoming features using the training scaler and converts failure probabilities into a 0–100 risk score."""
    if model is None or scaler is None or X is None:
        raise ValueError("Trained model, fitted scaler, and feature matrix X must be provided.")
        
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    failure_probabilities = model.predict_proba(X_scaled_df)[:, 1]
    risk_scores = failure_probabilities * 100
    
    return risk_scores

