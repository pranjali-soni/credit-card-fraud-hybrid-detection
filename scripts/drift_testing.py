"""
Concept Drift Testing Module
Tests whether models trained on earlier transactions still perform well 
on later transactions, simulating real-world deployment where fraud 
patterns evolve over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import TARGET, PREDICTORS, PLOTS_DIR


def create_temporal_split(data_df, split_ratio=0.7):
    """
    Split data based on the 'Time' column instead of randomly.
    Earlier transactions become "train", later transactions become "test".
    This simulates real-world deployment where a model must generalize to future data.
    
    Args:
        data_df (pd.DataFrame): Full dataset (must contain 'Time' column)
        split_ratio (float): Proportion of data (by time) to use for training
        
    Returns:
        tuple: (early_df, late_df) - early portion for training, late portion for testing
    """
    print("\n" + "="*60)
    print("CREATING TEMPORAL (TIME-BASED) SPLIT FOR DRIFT TESTING")
    print("="*60)
    
    # Sort by time to ensure correct chronological ordering
    data_sorted = data_df.sort_values('Time').reset_index(drop=True)
    
    split_index = int(len(data_sorted) * split_ratio)
    
    early_df = data_sorted.iloc[:split_index].reset_index(drop=True)
    late_df = data_sorted.iloc[split_index:].reset_index(drop=True)
    
    early_fraud_rate = (early_df[TARGET].sum() / len(early_df)) * 100
    late_fraud_rate = (late_df[TARGET].sum() / len(late_df)) * 100
    
    print(f"Early period (training) -> {len(early_df):,} transactions, "
          f"{early_df[TARGET].sum()} fraud ({early_fraud_rate:.3f}%)")
    print(f"Late period (testing)   -> {len(late_df):,} transactions, "
          f"{late_df[TARGET].sum()} fraud ({late_fraud_rate:.3f}%)")
    
    return early_df, late_df


def evaluate_drift_performance(model_predict_fn, early_train_df, early_valid_df, 
                                late_test_df, model_name="Model"):
    """
    Compare a model's performance on same-period validation data 
    vs a completely different, later time period (drift test).
    
    Args:
        model_predict_fn: A function that takes (train_df, valid_df, test_df) and 
                          returns (model, predictions_proba, metrics_dict) for the test_df
        early_train_df, early_valid_df: Data from the early period
        late_test_df: Data from the later period (the drift test set)
        model_name (str): Name of the model being tested
        
    Returns:
        dict: Comparison of same-period vs drift performance
    """
    print(f"\n--- Drift Test: {model_name} ---")
    
    metrics_same_period, metrics_drift = model_predict_fn(
        early_train_df, early_valid_df, late_test_df
    )
    
    comparison = {
        'Model': model_name,
        'AUC_SamePeriod': metrics_same_period['AUC'],
        'AUC_LaterPeriod': metrics_drift['AUC'],
        'AUC_Drop': metrics_same_period['AUC'] - metrics_drift['AUC'],
        'F1_SamePeriod': metrics_same_period['F1-Score'],
        'F1_LaterPeriod': metrics_drift['F1-Score'],
        'F1_Drop': metrics_same_period['F1-Score'] - metrics_drift['F1-Score'],
        'Recall_SamePeriod': metrics_same_period['Recall'],
        'Recall_LaterPeriod': metrics_drift['Recall'],
        'Recall_Drop': metrics_same_period['Recall'] - metrics_drift['Recall'],
    }
    
    print(f"  Same-period AUC: {metrics_same_period['AUC']:.4f} | Later-period AUC: {metrics_drift['AUC']:.4f} "
          f"| Drop: {comparison['AUC_Drop']:.4f}")
    
    return comparison


def plot_drift_comparison(drift_results_df, save=True):
    """
    Plot AUC performance: same-period vs later-period (drift) for all tested models.
    
    Args:
        drift_results_df (pd.DataFrame): Results from multiple evaluate_drift_performance calls
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(drift_results_df))
    width = 0.35
    
    ax.bar(x - width/2, drift_results_df['AUC_SamePeriod'], width, 
           label='Same-Period Performance', color='#2ecc71')
    ax.bar(x + width/2, drift_results_df['AUC_LaterPeriod'], width, 
           label='Later-Period Performance (Drift Test)', color='#e74c3c')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('AUC Score')
    ax.set_title('Concept Drift Test: Same-Period vs Later-Period Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(drift_results_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/drift_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_population_stability_index(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) for a single feature,
    comparing its distribution in an early period vs a later period.
    PSI < 0.1: no significant drift
    PSI 0.1-0.25: moderate drift
    PSI > 0.25: significant drift
    
    Args:
        expected (array): Feature values from the early/training period
        actual (array): Feature values from the later/test period
        buckets (int): Number of bins to divide the data into
        
    Returns:
        float: PSI value
    """
    def scale_range(data, buckets):
        breakpoints = np.arange(0, buckets + 1) / buckets * 100
        return np.percentile(data, breakpoints)
    
    breakpoints = scale_range(expected, buckets)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero / log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi_value


def analyze_feature_drift(early_df, late_df, top_n=10):
    """
    Calculate PSI for all features to identify which ones have drifted the most
    between the early and later time periods.
    
    Args:
        early_df, late_df: Early and late period dataframes
        top_n (int): Number of most-drifted features to display
        
    Returns:
        pd.DataFrame: Features sorted by PSI (most drifted first)
    """
    print("\n" + "="*60)
    print("FEATURE-LEVEL DRIFT ANALYSIS (Population Stability Index)")
    print("="*60)
    
    psi_results = []
    for feature in PREDICTORS:
        psi = calculate_population_stability_index(
            early_df[feature].values, late_df[feature].values
        )
        psi_results.append({'Feature': feature, 'PSI': psi})
    
    psi_df = pd.DataFrame(psi_results).sort_values('PSI', ascending=False).reset_index(drop=True)
    
    def interpret_psi(psi):
        if psi < 0.1:
            return "Stable"
        elif psi < 0.25:
            return "Moderate Drift"
        else:
            return "Significant Drift"
    
    psi_df['Interpretation'] = psi_df['PSI'].apply(interpret_psi)
    
    print(f"\nTop {top_n} most drifted features:")
    print(psi_df.head(top_n).to_string(index=False))
    
    return psi_df