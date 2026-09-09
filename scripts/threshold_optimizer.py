"""
Cost-Sensitive Threshold Optimization
Instead of using a fixed 0.5 cutoff, this finds the threshold that minimizes
the real-world cost of mistakes: missed fraud (False Negative) vs false alarms (False Positive)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import PLOTS_DIR
import os


def calculate_cost(y_true, y_proba, threshold, cost_fn=5000, cost_fp=50):
    """
    Calculate the total cost of using a specific threshold.
    
    Args:
        y_true (array): Actual labels (0 or 1)
        y_proba (array): Predicted probability of fraud
        threshold (float): Decision threshold to test
        cost_fn (float): Cost of a False Negative (missed fraud) - default ₹5000
        cost_fp (float): Cost of a False Positive (false alarm) - default ₹50
        
    Returns:
        dict: Cost breakdown
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Count each type of outcome
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))  # Missed fraud
    false_positives = np.sum((y_true == 0) & (y_pred == 1))  # False alarms
    true_positives = np.sum((y_true == 1) & (y_pred == 1))   # Caught fraud
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))   # Correctly approved
    
    total_cost = (false_negatives * cost_fn) + (false_positives * cost_fp)
    
    return {
        'threshold': threshold,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'total_cost': total_cost
    }


def find_optimal_threshold(y_true, y_proba, cost_fn=5000, cost_fp=50, 
                           thresholds=None):
    """
    Test multiple thresholds and find the one that minimizes total cost.
    
    Args:
        y_true (array): Actual labels
        y_proba (array): Predicted probabilities
        cost_fn (float): Cost of missing a fraud case
        cost_fp (float): Cost of a false alarm
        thresholds (array): Thresholds to test (default: 0.01 to 0.99 in steps of 0.01)
        
    Returns:
        tuple: (best_threshold, results_dataframe)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    
    print("\n" + "="*60)
    print("COST-SENSITIVE THRESHOLD OPTIMIZATION")
    print("="*60)
    print(f"Cost of missed fraud (False Negative): ₹{cost_fn}")
    print(f"Cost of false alarm (False Positive): ₹{cost_fp}")
    print(f"Testing {len(thresholds)} threshold values...\n")
    
    results = []
    for t in thresholds:
        cost_info = calculate_cost(y_true, y_proba, t, cost_fn, cost_fp)
        results.append(cost_info)
    
    results_df = pd.DataFrame(results)
    
    # Find the threshold with minimum total cost
    best_row = results_df.loc[results_df['total_cost'].idxmin()]
    best_threshold = best_row['threshold']
    
    # Compare to default 0.5 threshold
    default_cost_info = calculate_cost(y_true, y_proba, 0.5, cost_fn, cost_fp)
    
    print(f"Default threshold (0.5) -> Total Cost: ₹{default_cost_info['total_cost']:,.0f} "
          f"(Missed Fraud: {default_cost_info['false_negatives']}, False Alarms: {default_cost_info['false_positives']})")
    
    print(f"Optimal threshold ({best_threshold:.2f}) -> Total Cost: ₹{best_row['total_cost']:,.0f} "
          f"(Missed Fraud: {best_row['false_negatives']}, False Alarms: {best_row['false_positives']})")
    
    savings = default_cost_info['total_cost'] - best_row['total_cost']
    savings_pct = (savings / default_cost_info['total_cost']) * 100 if default_cost_info['total_cost'] > 0 else 0
    
    print(f"\n💰 Potential Savings: ₹{savings:,.0f} ({savings_pct:.1f}% reduction in total cost)")
    
    return best_threshold, results_df


def plot_cost_curve(results_df, model_name="Model", save=True):
    """
    Plot the cost curve showing total cost at each threshold value.
    
    Args:
        results_df (pd.DataFrame): Results from find_optimal_threshold
        model_name (str): Name of the model (for plot title/filename)
        save (bool): Whether to save the plot
    """
    best_row = results_df.loc[results_df['total_cost'].idxmin()]
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['total_cost'], color='#2c3e50', linewidth=2)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Default Threshold (0.5)')
    plt.axvline(x=best_row['threshold'], color='green', linestyle='--', alpha=0.7, 
               label=f"Optimal Threshold ({best_row['threshold']:.2f})")
    plt.scatter([best_row['threshold']], [best_row['total_cost']], color='green', s=100, zorder=5)
    
    plt.title(f'Cost Curve - {model_name}\n(Cost of Missed Fraud vs False Alarms at Different Thresholds)', fontsize=13)
    plt.xlabel('Decision Threshold')
    plt.ylabel('Total Cost (₹)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        safe_name = model_name.lower().replace(' ', '_')
        plt.savefig(f'{PLOTS_DIR}/cost_curve_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()