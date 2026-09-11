"""
Multi-Model Consensus Scoring
Uses agreement/disagreement across multiple models as a confidence signal
for practical transaction triage: APPROVE / REVIEW / BLOCK
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import PLOTS_DIR


def calculate_consensus_score(predictions_dict, threshold=0.5):
    """
    Given predictions (probabilities) from multiple models, calculate 
    how many models "vote" for fraud on each transaction.
    
    Args:
        predictions_dict (dict): {model_name: array_of_probabilities}
                                 All arrays must be the same length (same transactions)
        threshold (float): Probability cutoff to count as a "fraud vote"
        
    Returns:
        pd.DataFrame: One row per transaction, showing vote count and consensus level
    """
    print("\n" + "="*60)
    print("MULTI-MODEL CONSENSUS SCORING")
    print("="*60)
    
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    n_transactions = len(predictions_dict[model_names[0]])
    
    print(f"Combining votes from {n_models} models across {n_transactions:,} transactions")
    
    # Build a matrix: rows = transactions, columns = models, values = 0/1 vote
    votes_matrix = np.zeros((n_transactions, n_models))
    for i, model_name in enumerate(model_names):
        votes_matrix[:, i] = (predictions_dict[model_name] >= threshold).astype(int)
    
    vote_counts = votes_matrix.sum(axis=1)
    vote_fraction = vote_counts / n_models
    
    results_df = pd.DataFrame({
        'Fraud_Votes': vote_counts.astype(int),
        'Total_Models': n_models,
        'Consensus_Fraction': vote_fraction
    })
    
    # Assign a triage decision based on consensus level
    def assign_triage(fraction):
        if fraction >= 0.7:
            return 'BLOCK'
        elif fraction >= 0.3:
            return 'REVIEW'
        else:
            return 'APPROVE'
    
    results_df['Triage_Decision'] = results_df['Consensus_Fraction'].apply(assign_triage)
    
    print(f"\nTriage Distribution:")
    print(results_df['Triage_Decision'].value_counts().to_string())
    
    return results_df


def evaluate_consensus_triage(consensus_df, y_true):
    """
    Evaluate how well the consensus-based triage system performs,
    specifically checking: how many actual frauds fall into each triage category?
    
    Args:
        consensus_df (pd.DataFrame): Output from calculate_consensus_score
        y_true (array): Actual labels (0 or 1)
        
    Returns:
        pd.DataFrame: Breakdown of actual fraud/genuine counts per triage category
    """
    consensus_df = consensus_df.copy()
    consensus_df['Actual'] = y_true
    
    breakdown = consensus_df.groupby('Triage_Decision')['Actual'].agg(
        Total_Transactions='count',
        Actual_Fraud_Count='sum'
    ).reset_index()
    
    breakdown['Fraud_Rate_in_Category'] = (breakdown['Actual_Fraud_Count'] / breakdown['Total_Transactions'] * 100).round(3)
    
    print("\n" + "="*60)
    print("CONSENSUS TRIAGE EVALUATION")
    print("="*60)
    print(breakdown.to_string(index=False))
    
    # Calculate: what % of ALL fraud got caught in BLOCK or REVIEW (not silently approved)?
    total_fraud = y_true.sum()
    fraud_in_approve = breakdown[breakdown['Triage_Decision'] == 'APPROVE']['Actual_Fraud_Count'].sum() if 'APPROVE' in breakdown['Triage_Decision'].values else 0
    fraud_caught = total_fraud - fraud_in_approve
    catch_rate = (fraud_caught / total_fraud * 100) if total_fraud > 0 else 0
    
    print(f"\n💡 {catch_rate:.1f}% of all fraud was flagged for BLOCK or REVIEW (not silently approved)")
    print(f"   Only {fraud_in_approve} fraud case(s) slipped through to APPROVE")
    
    return breakdown


def plot_consensus_distribution(consensus_df, save=True):
    """
    Visualize the distribution of triage decisions.
    
    Args:
        consensus_df (pd.DataFrame): Output from calculate_consensus_score
        save (bool): Whether to save the plot
    """
    plt.figure(figsize=(8, 6))
    triage_counts = consensus_df['Triage_Decision'].value_counts()
    colors = {'APPROVE': '#2ecc71', 'REVIEW': '#f39c12', 'BLOCK': '#e74c3c'}
    plot_colors = [colors.get(cat, '#95a5a6') for cat in triage_counts.index]
    
    plt.bar(triage_counts.index, triage_counts.values, color=plot_colors)
    plt.title('Multi-Model Consensus Triage Distribution', fontsize=14)
    plt.xlabel('Triage Decision')
    plt.ylabel('Number of Transactions')
    
    for i, v in enumerate(triage_counts.values):
        plt.text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/consensus_triage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()