"""
Hybrid Fraud Detection Model
Combines Isolation Forest (unsupervised anomaly detection) with 
XGBoost (supervised classification) to catch both known and novel fraud patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from config import TARGET, PREDICTORS, RANDOM_STATE, PLOTS_DIR, MAX_ROUNDS, EARLY_STOP, VERBOSE_EVAL
from model_training import calculate_metrics, plot_confusion_matrix


def train_isolation_forest(train_df, contamination=0.002):
    """
    Train Isolation Forest on GENUINE transactions only.
    It learns what "normal" behavior looks like, then flags deviations.
    
    Args:
        train_df (pd.DataFrame): Training data
        contamination (float): Expected proportion of anomalies (roughly matches fraud rate)
        
    Returns:
        IsolationForest: Trained model
    """
    print("\n" + "="*50)
    print("Training Isolation Forest (Stage 1 - Anomaly Detection)...")
    print("="*50)
    
    # Train ONLY on genuine transactions - this is the key idea:
    # the model learns what "normal" looks like, without ever seeing fraud examples
    genuine_only = train_df[train_df[TARGET] == 0]
    
    print(f"Training on {len(genuine_only):,} genuine transactions only (no fraud examples shown)")
    
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    iso_forest.fit(genuine_only[PREDICTORS])
    
    print("✓ Isolation Forest trained successfully")
    
    return iso_forest


def get_anomaly_scores(iso_forest, df):
    """
    Get anomaly scores from Isolation Forest, normalized to 0-1 range
    where higher = more anomalous (more likely fraud).
    
    Args:
        iso_forest: Trained Isolation Forest model
        df (pd.DataFrame): Data to score
        
    Returns:
        np.array: Normalized anomaly scores (0 to 1)
    """
    # decision_function gives higher values for "normal" points, lower for anomalies
    # We flip and normalize it so higher = more anomalous (matches fraud probability intuition)
    raw_scores = iso_forest.decision_function(df[PREDICTORS])
    
    # Flip sign so higher = more anomalous
    anomaly_scores = -raw_scores
    
    # Normalize to 0-1 range
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
    
    return normalized_scores


def train_xgboost_for_hybrid(train_df, valid_df):
    """
    Train the supervised XGBoost component of the hybrid model.
    Note: trained WITHOUT SMOTE, since our experiments showed SMOTE
    actually reduced XGBoost's AUPRC (0.8269 -> 0.7488).
    
    Returns:
        tuple: (model, dvalid for later prediction)
    """
    print("\n" + "="*50)
    print("Training XGBoost (Stage 2 - Supervised Classification)...")
    print("="*50)
    
    dtrain = xgb.DMatrix(train_df[PREDICTORS], train_df[TARGET].values)
    dvalid = xgb.DMatrix(valid_df[PREDICTORS], valid_df[TARGET].values)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    params = {
        'objective': 'binary:logistic',
        'eta': 0.039,
        'max_depth': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'eval_metric': 'auc',
        'random_state': RANDOM_STATE
    }
    
    model = xgb.train(
        params, dtrain, MAX_ROUNDS, watchlist,
        early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False
    )
    
    print("✓ XGBoost trained successfully")
    
    return model


def train_hybrid_model(train_df, valid_df, alpha=0.7):
    """
    Train the full Hybrid Model: Isolation Forest + XGBoost combined.
    
    Args:
        train_df, valid_df: Data splits
        alpha (float): Weight given to XGBoost's score vs Isolation Forest's anomaly score.
                       Final score = alpha * XGBoost_proba + (1-alpha) * anomaly_score
                       alpha=0.7 means XGBoost's opinion counts 70%, anomaly score counts 30%
        
    Returns:
        tuple: (iso_forest, xgb_model, predictions, metrics_dict)
    """
    print("\n" + "#"*60)
    print("# BUILDING HYBRID MODEL (Isolation Forest + XGBoost)")
    print("#"*60)
    
    # Stage 1: Isolation Forest
    iso_forest = train_isolation_forest(train_df)
    
    # Stage 2: XGBoost
    xgb_model = train_xgboost_for_hybrid(train_df, valid_df)
    
    # Get scores from both models on validation data
    print("\nCombining both models' scores...")
    anomaly_scores = get_anomaly_scores(iso_forest, valid_df)
    
    dvalid = xgb.DMatrix(valid_df[PREDICTORS])
    xgb_proba = xgb_model.predict(dvalid)
    
    # Combine: weighted average of both scores
    hybrid_proba = alpha * xgb_proba + (1 - alpha) * anomaly_scores
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, hybrid_proba)
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds, rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "HybridModel")
    
    print(f"\nHybrid Model (alpha={alpha}) -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return iso_forest, xgb_model, hybrid_proba, metrics


def compare_alpha_values(train_df, valid_df, alpha_values=[0.3, 0.5, 0.7, 0.9]):
    """
    Test different alpha values (weighting between XGBoost and Isolation Forest)
    to find the best combination.
    
    Args:
        train_df, valid_df: Data splits
        alpha_values (list): List of alpha values to test
        
    Returns:
        pd.DataFrame: Results for each alpha value
    """
    print("\n" + "#"*60)
    print("# TESTING DIFFERENT ALPHA VALUES FOR HYBRID MODEL")
    print("#"*60)
    
    # Train both components once
    iso_forest = train_isolation_forest(train_df)
    xgb_model = train_xgboost_for_hybrid(train_df, valid_df)
    
    anomaly_scores = get_anomaly_scores(iso_forest, valid_df)
    dvalid = xgb.DMatrix(valid_df[PREDICTORS])
    xgb_proba = xgb_model.predict(dvalid)
    
    results = []
    for alpha in alpha_values:
        hybrid_proba = alpha * xgb_proba + (1 - alpha) * anomaly_scores
        metrics, _ = calculate_metrics(valid_df[TARGET].values, hybrid_proba)
        metrics['Alpha'] = alpha
        results.append(metrics)
        print(f"Alpha={alpha} -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
              f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot alpha comparison
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Alpha'], results_df['F1-Score'], marker='o', label='F1-Score')
    plt.plot(results_df['Alpha'], results_df['AUPRC'], marker='s', label='AUPRC')
    plt.plot(results_df['Alpha'], results_df['Recall'], marker='^', label='Recall')
    plt.plot(results_df['Alpha'], results_df['Precision'], marker='d', label='Precision')
    plt.xlabel('Alpha (weight given to XGBoost vs Isolation Forest)')
    plt.ylabel('Score')
    plt.title('Hybrid Model Performance vs Alpha Weighting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f'{PLOTS_DIR}/hybrid_alpha_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Alpha comparison chart saved")
    
    return results_df