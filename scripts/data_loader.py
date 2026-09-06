"""
Data loading module for credit card fraud detection
"""

import pandas as pd


def load_data():
    """
    Load the credit card fraud detection dataset from local CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("Loading dataset from local file...")
    
    data_df = pd.read_csv('../data/creditcard.csv')
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {data_df.shape}")
    print(f"Rows: {data_df.shape[0]}, Columns: {data_df.shape[1]}")
    
    return data_df


def get_data_summary(data_df):
    """
    Get summary statistics of the dataset
    
    Args:
        data_df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary containing summary information
    """
    summary = {
        'shape': data_df.shape,
        'columns': data_df.columns.tolist(),
        'missing_values': data_df.isnull().sum().sum(),
        'fraud_count': data_df['Class'].sum(),
        'fraud_percentage': (data_df['Class'].sum() / len(data_df)) * 100
    }
    
    return summary