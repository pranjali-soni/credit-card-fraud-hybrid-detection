"""
SMOTE (Synthetic Minority Oversampling Technique) handler
Used to balance the training data ONLY - never applied to validation/test data,
to avoid data leakage and maintain realistic evaluation conditions.
"""

from imblearn.over_sampling import SMOTE
from config import TARGET, PREDICTORS, RANDOM_STATE
import pandas as pd


def apply_smote(train_df):
    """
    Apply SMOTE to balance the training dataset only.
    
    Args:
        train_df (pd.DataFrame): Original imbalanced training data
        
    Returns:
        pd.DataFrame: Balanced training dataframe with synthetic fraud samples added
    """
    print("\n" + "="*50)
    print("Applying SMOTE to Training Data...")
    print("="*50)
    
    X_train = train_df[PREDICTORS]
    y_train = train_df[TARGET]
    
    print(f"Before SMOTE -> Genuine: {(y_train == 0).sum():,} | Fraud: {(y_train == 1).sum():,}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE  -> Genuine: {(y_resampled == 0).sum():,} | Fraud: {(y_resampled == 1).sum():,}")
    
    # Reconstruct a dataframe in the same format as train_df
    train_df_smote = pd.DataFrame(X_resampled, columns=PREDICTORS)
    train_df_smote[TARGET] = y_resampled
    
    print("✓ SMOTE applied successfully - training data is now balanced")
    
    return train_df_smote