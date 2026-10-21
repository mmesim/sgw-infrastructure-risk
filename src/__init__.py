# This file makes the folder a package
# You can "lift" the functions here so they are easier to import

# Version of the package
__version__ = "0.0.1"

#Data Preparation
from .data_prep import load_data, clean_and_impute, create_binary_target

#Feature Engineering
from .features import encode_categorical_features, engineer_risk_features, prepare_feature_matrix

# Modeling
from .model import train_xgboost_model, generate_risk_scores
__all__ = [
    "load_data", 
    "clean_and_impute", 
    "create_binary_target",
    "encode_categorical_features",
    "engineer_risk_features",
    "prepare_feature_matrix",
    "train_xgboost_model",
    "generate_risk_scores"

]