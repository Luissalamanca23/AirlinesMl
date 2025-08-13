"""
Machine Learning pipeline nodes for Airlines price prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


def prepare_features(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Prepare features for ML models by encoding categorical variables.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset  
        test_data: Test dataset
        
    Returns:
        Tuple of (processed_train, processed_val, processed_test, encoders)
    """
    logger.info("Preparing features for ML models")
    
    # Copy data to avoid modifying originals
    train = train_data.copy()
    val = val_data.copy()
    test = test_data.copy()
    
    # Identify categorical and numerical columns
    categorical_cols = ['airline', 'flight', 'source_city', 'departure_time', 
                       'stops', 'arrival_time', 'destination_city', 'class']
    numerical_cols = ['duration', 'days_left']
    target_col = 'price'
    
    # Initialize encoders dictionary
    encoders = {}
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in train.columns:
            logger.info(f"Encoding column: {col}")
            le = LabelEncoder()
            
            # Combine all data to get all possible categories
            combined_data = pd.concat([
                train[col].astype(str),
                val[col].astype(str),
                test[col].astype(str)
            ])
            
            # Fit on combined data to include all categories
            le.fit(combined_data)
            
            # Transform all datasets
            train[col] = le.transform(train[col].astype(str))
            val[col] = le.transform(val[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
            
            encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    for col in numerical_cols:
        if col in train.columns:
            logger.info(f"Scaling column: {col}")
            train[col] = scaler.fit_transform(train[[col]])
            val[col] = scaler.transform(val[[col]])
            test[col] = scaler.transform(test[[col]])
    
    encoders['scaler'] = scaler
    
    logger.info(f"Feature preparation completed. Train shape: {train.shape}")
    return train, val, test, encoders


def train_linear_regression(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> LinearRegression:
    """Train Linear Regression model.
    
    Args:
        train_data: Training dataset
        parameters: Model parameters
        
    Returns:
        Trained Linear Regression model
    """
    logger.info("Training Linear Regression model")
    
    # Separate features and target
    X = train_data.drop('price', axis=1)
    y = train_data['price']
    
    # Initialize and train model
    model = LinearRegression()
    model.fit(X, y)
    
    logger.info("Linear Regression training completed")
    return model


def train_random_forest(train_data: pd.DataFrame, parameters: Dict[str, Any]) -> RandomForestRegressor:
    """Train Random Forest model.
    
    Args:
        train_data: Training dataset
        parameters: Model parameters
        
    Returns:
        Trained Random Forest model
    """
    logger.info("Training Random Forest model")
    
    # Separate features and target
    X = train_data.drop('price', axis=1)
    y = train_data['price']
    
    # Get model parameters
    rf_params = parameters.get('random_forest', {})
    
    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=rf_params.get('n_estimators', 100),
        max_depth=rf_params.get('max_depth', None),
        random_state=42
    )
    model.fit(X, y)
    
    logger.info("Random Forest training completed")
    return model


def train_xgboost(train_data: pd.DataFrame, val_data: pd.DataFrame, parameters: Dict[str, Any]) -> xgb.XGBRegressor:
    """Train XGBoost model.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        parameters: Model parameters
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model")
    
    # Separate features and target
    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    X_val = val_data.drop('price', axis=1)
    y_val = val_data['price']
    
    # Get model parameters
    xgb_params = parameters.get('xgboost', {})
    
    # Initialize and train model
    model = xgb.XGBRegressor(
        n_estimators=xgb_params.get('n_estimators', 100),
        max_depth=xgb_params.get('max_depth', 6),
        learning_rate=xgb_params.get('learning_rate', 0.1),
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    logger.info("XGBoost training completed")
    return model


def train_lightgbm(train_data: pd.DataFrame, val_data: pd.DataFrame, parameters: Dict[str, Any]) -> lgb.LGBMRegressor:
    """Train LightGBM model.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        parameters: Model parameters
        
    Returns:
        Trained LightGBM model
    """
    logger.info("Training LightGBM model")
    
    # Separate features and target
    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    X_val = val_data.drop('price', axis=1)
    y_val = val_data['price']
    
    # Get model parameters
    lgb_params = parameters.get('lightgbm', {})
    
    # Initialize and train model
    model = lgb.LGBMRegressor(
        n_estimators=lgb_params.get('n_estimators', 100),
        max_depth=lgb_params.get('max_depth', 6),
        learning_rate=lgb_params.get('learning_rate', 0.1),
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    logger.info("LightGBM training completed")
    return model


def evaluate_model(model: Any, test_data: pd.DataFrame, model_name: str) -> Dict[str, float]:
    """Evaluate model performance on test data.
    
    Args:
        model: Trained model
        test_data: Test dataset
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name} model")
    
    # Separate features and target
    X_test = test_data.drop('price', axis=1)
    y_test = test_data['price']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    logger.info(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    return metrics


def compare_models(lr_metrics: Dict, rf_metrics: Dict, xgb_metrics: Dict, lgb_metrics: Dict) -> pd.DataFrame:
    """Compare all model performances.
    
    Args:
        lr_metrics: Linear Regression metrics
        rf_metrics: Random Forest metrics
        xgb_metrics: XGBoost metrics
        lgb_metrics: LightGBM metrics
        
    Returns:
        DataFrame with model comparison
    """
    logger.info("Comparing model performances")
    
    # Create comparison dataframe
    comparison = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics, lgb_metrics])
    comparison = comparison.sort_values('rmse')
    
    logger.info("Model comparison:")
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    return comparison
