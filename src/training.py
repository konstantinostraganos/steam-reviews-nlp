"""
Model training utilities for sentiment classification.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression with basic hyperparameter tuning."""
    param_grid = {"C": [0.1, 1, 10]}
    
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    grid = GridSearchCV(lr, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best C: {grid.best_params_['C']} | Best CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_xgboost(X_train, y_train, random_state=42):
    """Train XGBoost with hyperparameter tuning."""
    param_grid = {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    }
    
    xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False
    )
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best params: {grid.best_params_} | Best CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to balance training data."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Before SMOTE: {np.bincount(y_train)}")
    print(f"After SMOTE:  {np.bincount(y_resampled)}")
    return X_resampled, y_resampled