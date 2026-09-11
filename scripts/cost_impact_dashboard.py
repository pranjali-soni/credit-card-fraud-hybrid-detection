"""
Business Cost-Impact Dashboard
Translates raw model metrics (Precision, Recall) into actual monetary 
business impact, making results interpretable for non-technical stakeholders.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import PLOTS_DIR


def calculate_business_impact(y_true, y_pred, avg_fraud_amount=5000, 
                               false_alarm_cost=50):
    """
    Calculate the real-world monetary impact of a model's predictions.
    
    Args:
        y_true (array): Actual labels
        y_pred (array): Predicted labels (0 or 1)
        avg_fraud_amount (float): Average financial loss per missed fraud case
        false_alarm_cost (float): Average cost per false alarm (customer service, review time)
        
    Returns:
        dict: Business impact breakdown
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))   # Fraud caught
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))  # Fraud missed
    false_positives = np.sum((y_true == 0) & (y_pred == 1))  # False alarms
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))   # Correctly approved
    
    money_saved = true_positives * avg_fraud_amount
    money_lost_to_missed_fraud = false_negatives * avg_fraud_amount
    cost_of_false_alarms = false_positives * false_alarm_cost
    
    net_impact = money_saved - cost_of_false_alarms
    
    return {
        'Fraud_Caught': int(true_positives),
        'Fraud_Missed': int(false_negatives),
        'False_Alarms': int(false_positives),
        'Correctly_Approved': int(true_negatives),
        'Money_Saved': money_saved,
        'Money_Lost_to_Missed_Fraud': money_lost_to_missed_fraud,
        'Cost_of_False_Alarms': cost_of_false_alarms,
        'Net_Business_Impact': net_impact
    }


def compare_business_impact_across_models(predictions_dict, y_true, 
                                          avg_fraud_amount=5000, false_alarm_cost=50,
                                          threshold=0.5):
    """
    Compare the business impact of multiple models side by side.
    
    Args:
        predictions_dict (dict): {model_name: array_of_probabilities}
        y_true (array): Actual labels
        avg_fraud_amount (float): Average loss per missed fraud
        false_alarm_cost (float): Average cost per false alarm
        threshold (float): Decision threshold to apply
        
    Returns:
        pd.DataFrame: Business impact comparison across models
    """
    print("\n" + "="*70)
    print("BUSINESS COST-IMPACT DASHBOARD")
    print("="*70)
    print(f"Assumptions: Avg fraud loss = ₹{avg_fraud_amount:,} | False alarm cost = ₹{false_alarm_cost}")
    print("-"*70)
    
    results = []
    for model_name, proba in predictions_dict.items():
        y_pred = (proba >= threshold).astype(int)
        impact = calculate_business_impact(y_true, y_pred, avg_fraud_amount, false_alarm_cost)
        impact['Model'] = model_name
        results.append(impact)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Net_Business_Impact', ascending=False).reset_index(drop=True)
    
    # Reorder columns for readability
    cols = ['Model', 'Fraud_Caught', 'Fraud_Missed', 'False_Alarms', 
            'Money_Saved', 'Money_Lost_to_Missed_Fraud', 'Cost_of_False_Alarms', 
            'Net_Business_Impact']
    results_df = results_df[cols]
    
    print("\nBusiness Impact Ranking (by Net Impact):")
    for i, row in results_df.iterrows():
        print(f"{i+1}. {row['Model']:<20} -> Net Impact: ₹{row['Net_Business_Impact']:,.0f} "
              f"(Caught: {row['Fraud_Caught']}, Missed: {row['Fraud_Missed']}, False Alarms: {row['False_Alarms']})")
    
    return results_df


def plot_business_impact(results_df, save=True):
    """
    Visualize the net business impact across models.
    
    Args:
        results_df (pd.DataFrame): Output from compare_business_impact_across_models
        save (bool): Whether to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Net impact comparison
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in results_df['Net_Business_Impact']]
    axes[0].barh(results_df['Model'], results_df['Net_Business_Impact'], color=colors)
    axes[0].set_xlabel('Net Business Impact (₹)')
    axes[0].set_title('Net Financial Impact by Model')
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    
    # Breakdown: Money saved vs Cost of false alarms vs Money lost
    x = np.arange(len(results_df))
    width = 0.6
    axes[1].bar(x, results_df['Money_Saved'], width, label='Money Saved (Fraud Caught)', color='#2ecc71')
    axes[1].bar(x, -results_df['Cost_of_False_Alarms'], width, label='Cost of False Alarms', color='#f39c12')
    axes[1].bar(x, -results_df['Money_Lost_to_Missed_Fraud'], width, 
                bottom=-results_df['Cost_of_False_Alarms'], label='Money Lost (Missed Fraud)', color='#e74c3c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[1].set_ylabel('₹ Amount')
    axes[1].set_title('Cost Breakdown by Model')
    axes[1].legend()
    axes[1].axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/business_cost_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()