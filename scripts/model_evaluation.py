"""
Model evaluation and comparison module
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import PLOTS_DIR


def create_results_summary(results_dict):
    """
    Create a summary dataframe of all model results
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metrics dicts as values
                              e.g. {'XGBoost': {'Accuracy':..., 'Precision':..., 'Recall':..., 'F1-Score':..., 'AUC':..., 'AUPRC':...}}
        
    Returns:
        pd.DataFrame: Summary dataframe with one row per model, one column per metric
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('AUC', ascending=False).reset_index(drop=True)
    
    return df


def plot_model_comparison(results_df, save=True):
    """
    Plot comparison of all models across AUC and AUPRC
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        save (bool): Whether to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # AUC comparison
    sns.barplot(data=results_df, x='AUC', y='Model', palette='viridis', ax=axes[0])
    axes[0].set_title('Model Comparison - ROC-AUC Score', fontsize=14)
    axes[0].set_xlabel('ROC-AUC Score')
    axes[0].set_xlim(0.0, 1.0)
    for i, row in results_df.iterrows():
        axes[0].text(row['AUC'] - 0.05, i, f"{row['AUC']:.4f}",
                     va='center', fontsize=9, color='white', weight='bold')
    
    # AUPRC comparison
    df_sorted_auprc = results_df.sort_values('AUPRC', ascending=False).reset_index(drop=True)
    sns.barplot(data=df_sorted_auprc, x='AUPRC', y='Model', palette='magma', ax=axes[1])
    axes[1].set_title('Model Comparison - AUPRC (Precision-Recall AUC)', fontsize=14)
    axes[1].set_xlabel('AUPRC')
    axes[1].set_xlim(0.0, 1.0)
    for i, row in df_sorted_auprc.iterrows():
        axes[1].text(row['AUPRC'] - 0.05, i, f"{row['AUPRC']:.4f}",
                     va='center', fontsize=9, color='white', weight='bold')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_evaluation_report(results_dict):
    """
    Generate a comprehensive evaluation report covering all metrics
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metrics dicts as values
        
    Returns:
        tuple: (report string, results dataframe)
    """
    results_df = create_results_summary(results_dict)
    
    report = "\n" + "="*100 + "\n"
    report += "MODEL EVALUATION SUMMARY (sorted by AUC)\n"
    report += "="*100 + "\n\n"
    
    header = f"{'Rank':<6}{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'AUC':<10}{'AUPRC':<10}\n"
    report += header
    report += "-" * 100 + "\n"
    
    for i, row in results_df.iterrows():
        rank_emoji = "1st" if i == 0 else "2nd" if i == 1 else "3rd" if i == 2 else f"{i+1}th"
        report += (f"{rank_emoji:<6}{row['Model']:<20}{row['Accuracy']:<12.4f}"
                   f"{row['Precision']:<12.4f}{row['Recall']:<12.4f}{row['F1-Score']:<12.4f}"
                   f"{row['AUC']:<10.4f}{row['AUPRC']:<10.4f}\n")
    
    report += "\n" + "="*100 + "\n"
    report += f"Best Model (by AUC): {results_df.iloc[0]['Model']} with AUC = {results_df.iloc[0]['AUC']:.4f}\n"
    
    best_recall_row = results_df.sort_values('Recall', ascending=False).iloc[0]
    report += f"Best Model (by Recall - catches most fraud): {best_recall_row['Model']} with Recall = {best_recall_row['Recall']:.4f}\n"
    
    best_auprc_row = results_df.sort_values('AUPRC', ascending=False).iloc[0]
    report += f"Best Model (by AUPRC - most reliable on imbalanced data): {best_auprc_row['Model']} with AUPRC = {best_auprc_row['AUPRC']:.4f}\n"
    report += "="*100 + "\n"
    
    return report, results_df


def save_results_to_csv(results_df, filename='model_results.csv'):
    """
    Save results to CSV file
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        filename (str): Output filename
    """
    output_path = f'{PLOTS_DIR}/{filename}'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def plot_before_after_smote(results_df_no_smote, results_df_with_smote, metric='F1-Score', save=True):
    """
    Plot a before/after SMOTE comparison for a given metric across all models.
    
    Args:
        results_df_no_smote (pd.DataFrame): Results without SMOTE
        results_df_with_smote (pd.DataFrame): Results with SMOTE
        metric (str): Which metric to compare (e.g., 'F1-Score', 'Recall', 'Precision', 'AUPRC')
        save (bool): Whether to save the plot
    """
    merged = pd.merge(
        results_df_no_smote[['Model', metric]].rename(columns={metric: 'Without SMOTE'}),
        results_df_with_smote[['Model', metric]].rename(columns={metric: 'With SMOTE'}),
        on='Model'
    )
    
    merged_melted = merged.melt(id_vars='Model', var_name='Condition', value_name=metric)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=merged_melted, x=metric, y='Model', hue='Condition', palette=['#e74c3c', '#2ecc71'])
    plt.title(f'Before vs After SMOTE - {metric} Comparison', fontsize=15)
    plt.xlabel(metric, fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.legend(title='')
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        safe_metric_name = metric.lower().replace(' ', '_').replace('-', '_')
        plt.savefig(f'{PLOTS_DIR}/smote_comparison_{safe_metric_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()