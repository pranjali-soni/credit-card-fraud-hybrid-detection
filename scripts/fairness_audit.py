"""
Fairness Audit Module
Checks whether the model's False Positive Rate differs significantly 
across transaction amount brackets (a proxy for fairness, since this 
dataset lacks demographic information).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import PLOTS_DIR


def create_amount_brackets(amounts):
    """
    Categorize transaction amounts into Low, Medium, High brackets.
    
    Args:
        amounts (array): Transaction amount values
        
    Returns:
        array: Bracket labels
    """
    def bracket(amount):
        if amount < 50:
            return 'Low (<$50)'
        elif amount < 500:
            return 'Medium ($50-$500)'
        else:
            return 'High (>$500)'
    
    return np.array([bracket(a) for a in amounts])


def audit_fairness_by_amount(y_true, y_pred, amounts):
    """
    Calculate False Positive Rate and False Negative Rate separately 
    for each transaction amount bracket, to check for fairness disparities.
    
    Args:
        y_true (array): Actual labels
        y_pred (array): Predicted labels (0 or 1, after threshold applied)
        amounts (array): Transaction amounts
        
    Returns:
        pd.DataFrame: Fairness metrics per bracket
    """
    print("\n" + "="*60)
    print("FAIRNESS AUDIT: PERFORMANCE ACROSS TRANSACTION AMOUNT BRACKETS")
    print("="*60)
    
    brackets = create_amount_brackets(amounts)
    
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Bracket': brackets
    })
    
    results = []
    for bracket_name in ['Low (<$50)', 'Medium ($50-$500)', 'High (>$500)']:
        bracket_df = df[df['Bracket'] == bracket_name]
        
        if len(bracket_df) == 0:
            continue
        
        # Genuine transactions in this bracket
        genuine = bracket_df[bracket_df['Actual'] == 0]
        fraud = bracket_df[bracket_df['Actual'] == 1]
        
        # False Positive Rate: of genuine transactions, how many wrongly flagged as fraud?
        fpr = (genuine['Predicted'] == 1).sum() / len(genuine) if len(genuine) > 0 else 0
        
        # False Negative Rate: of actual fraud, how many wrongly approved?
        fnr = (fraud['Predicted'] == 0).sum() / len(fraud) if len(fraud) > 0 else 0
        
        results.append({
            'Bracket': bracket_name,
            'Total_Transactions': len(bracket_df),
            'Genuine_Count': len(genuine),
            'Fraud_Count': len(fraud),
            'False_Positive_Rate': fpr,
            'False_Negative_Rate': fnr
        })
    
    results_df = pd.DataFrame(results)
    
    print(results_df.to_string(index=False))
    
    # Check for disparity
    if len(results_df) > 1:
        fpr_range = results_df['False_Positive_Rate'].max() - results_df['False_Positive_Rate'].min()
        print(f"\n💡 False Positive Rate range across brackets: {fpr_range:.4f}")
        if fpr_range > 0.02:
            print("⚠️  Notable disparity detected - some transaction amount groups face higher false alarm rates than others")
        else:
            print("✓ False Positive Rates are relatively consistent across amount brackets")
    
    return results_df


def plot_fairness_audit(fairness_df, save=True):
    """
    Visualize False Positive Rate and False Negative Rate across brackets.
    
    Args:
        fairness_df (pd.DataFrame): Output from audit_fairness_by_amount
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(fairness_df))
    width = 0.35
    
    ax.bar(x - width/2, fairness_df['False_Positive_Rate'], width, 
           label='False Positive Rate', color='#e74c3c')
    ax.bar(x + width/2, fairness_df['False_Negative_Rate'], width, 
           label='False Negative Rate', color='#3498db')
    
    ax.set_xlabel('Transaction Amount Bracket')
    ax.set_ylabel('Rate')
    ax.set_title('Fairness Audit: Error Rates Across Transaction Amount Brackets')
    ax.set_xticks(x)
    ax.set_xticklabels(fairness_df['Bracket'])
    ax.legend()
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/fairness_audit.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()