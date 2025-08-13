"""
Data processing nodes for Airlines ML project.
"""

import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def preprocess_airlines_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the airlines dataset.
    
    Args:
        raw_data: Raw airlines flight data
        
    Returns:
        Cleaned and preprocessed data
    """
    logger.info("Starting data preprocessing")
    
    # Create a copy to avoid modifying original data
    data = raw_data.copy()
    
    # Basic info about the dataset
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Columns: {list(data.columns)}")
    
    # Handle missing values if any
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Found missing values: {missing_values[missing_values > 0]}")
        data = data.dropna()
    
    # Basic feature engineering
    # Convert categorical variables to proper format
    categorical_cols = ['Airline', 'Flight', 'Source_City', 'Departure_Time', 
                       'Stops', 'Arrival_Time', 'Destination_City', 'Class']
    
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')
    
    logger.info(f"Processed dataset shape: {data.shape}")
    return data


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.
    
    Args:
        data: Preprocessed data
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Splitting data into train/val/test sets")
    
    # First split: train+val vs test (80% vs 20%)
    train_val, test = train_test_split(data, test_size=0.2, random_state=42)
    
    # Second split: train vs val (80% vs 20% of train_val, which gives us 64%/16%/20% split)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    
    logger.info(f"Train set: {len(train)} samples")
    logger.info(f"Validation set: {len(val)} samples") 
    logger.info(f"Test set: {len(test)} samples")
    
    return train, val, test
