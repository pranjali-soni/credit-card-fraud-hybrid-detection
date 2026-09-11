"""
SHAP (SHapley Additive exPlanations) Explainability Module
Provides both global (overall feature importance) and local (individual 
prediction) explanations for fraud detection decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import xgboost as xgb

from config import TARGET, PREDICTORS, PLOTS_DIR


def create_shap_explainer(xgb_model):
    """
    Create a SHAP TreeExplainer for an XGBoost model.
    TreeExplainer is fast and exact for tree-based models like XGBoost.
    
    Args:
        xgb_model: Trained XGBoost model (Booster object)
        
    Returns:
        shap.TreeExplainer: The explainer object
    """
    print("\n" + "="*60)
    print("Creating SHAP Explainer...")
    print("="*60)
    
    explainer = shap.TreeExplainer(xgb_model)
    
    print("✓ SHAP Explainer created successfully")
    
    return explainer


def generate_global_explanation(explainer, X_sample, save=True):
    """
    Generate a global SHAP summary plot showing which features matter 
    most across all predictions in the dataset.
    
    Args:
        explainer: SHAP explainer object
        X_sample (pd.DataFrame): Sample of data to explain (use a subset for speed)
        save (bool): Whether to save the plot
        
    Returns:
        array: SHAP values for the sample
    """
    print("\nCalculating SHAP values for global explanation (this may take a moment)...")
    
    shap_values = explainer.shap_values(X_sample)
    
    print("✓ SHAP values calculated")
    
    # Summary plot (bar chart - overall feature importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance - Which Features Drive Fraud Predictions Overall", fontsize=12)
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/shap_global_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Global importance bar chart saved to {PLOTS_DIR}/shap_global_importance_bar.png")
    
    # Summary plot (beeswarm - shows direction and magnitude of impact)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot - Feature Impact Direction and Magnitude", fontsize=12)
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{PLOTS_DIR}/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Summary beeswarm plot saved to {PLOTS_DIR}/shap_summary_beeswarm.png")
    
    return shap_values


def explain_single_transaction(explainer, X_sample, shap_values, transaction_index, 
                                 actual_label=None, save=True):
    """
    Generate a local (individual) SHAP explanation for one specific transaction,
    showing exactly which features pushed it toward "fraud" or "genuine".
    
    Args:
        explainer: SHAP explainer object
        X_sample (pd.DataFrame): The sample data used
        shap_values (array): Precomputed SHAP values for X_sample
        transaction_index (int): Which row (by position) to explain
        actual_label: The true label for this transaction (0 or 1), if known
        save (bool): Whether to save the plot
        
    Returns:
        dict: Explanation details (top contributing features)
    """
    print(f"\n--- Explaining Transaction #{transaction_index} ---")
    if actual_label is not None:
        print(f"Actual label: {'FRAUD' if actual_label == 1 else 'GENUINE'}")
    
    # Get SHAP values for this specific transaction
    transaction_shap = shap_values[transaction_index]
    transaction_features = X_sample.iloc[transaction_index]
    
    # Build a dataframe of feature contributions, sorted by absolute impact
    contributions = pd.DataFrame({
        'Feature': PREDICTORS,
        'Value': transaction_features.values,
        'SHAP_Contribution': transaction_shap
    })
    contributions['Abs_Contribution'] = contributions['SHAP_Contribution'].abs()
    contributions = contributions.sort_values('Abs_Contribution', ascending=False).reset_index(drop=True)
    
    print("\nTop 5 features influencing this prediction:")
    for i, row in contributions.head(5).iterrows():
        direction = "→ INCREASES fraud risk" if row['SHAP_Contribution'] > 0 else "→ DECREASES fraud risk"
        print(f"  {row['Feature']}: value={row['Value']:.3f} | contribution={row['SHAP_Contribution']:.4f} {direction}")
    
    # Waterfall plot for this transaction
    plt.figure(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, 
        transaction_shap, 
        feature_names=PREDICTORS,
        max_display=10,
        show=False
    )
    plt.title(f"SHAP Explanation - Transaction #{transaction_index} "
              f"(Actual: {'FRAUD' if actual_label == 1 else 'GENUINE' if actual_label == 0 else 'Unknown'})",
              fontsize=11)
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/shap_explanation_transaction_{transaction_index}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Waterfall plot saved to {PLOTS_DIR}/shap_explanation_transaction_{transaction_index}.png")
    
    return contributions.head(5).to_dict('records')


def generate_natural_language_explanation(contributions, prediction_label):
    """
    Convert SHAP contributions into a plain-English explanation.
    This is the "explainability for non-technical users" piece.
    
    Args:
        contributions (list): Top feature contributions (from explain_single_transaction)
        prediction_label (str): "FRAUD" or "GENUINE"
        
    Returns:
        str: Natural language explanation
    """
    top_features = contributions[:3]
    
    reasons = []
    for feat in top_features:
        direction = "an unusually high risk signal" if feat['SHAP_Contribution'] > 0 else "a reduced risk signal"
        reasons.append(f"{feat['Feature']} (value: {feat['Value']:.2f}, {direction})")
    
    explanation = (
        f"This transaction was classified as {prediction_label} primarily due to: "
        f"{', '.join(reasons)}."
    )
    
    return explanation