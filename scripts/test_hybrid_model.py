"""
Quick isolated test for the Hybrid Model (Isolation Forest + XGBoost)
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data
from data_preprocessing import prepare_data
from hybrid_model import train_hybrid_model, compare_alpha_values
import time

print("Loading data...")
data_df = load_data()

print("Preparing data...")
train_df, valid_df, test_df = prepare_data(data_df)

start = time.time()

# Test 1: Train the hybrid model with default alpha
print("\n" + "="*60)
print("TEST 1: Default Hybrid Model (alpha=0.7)")
print("="*60)
iso_forest, xgb_model, hybrid_proba, metrics = train_hybrid_model(train_df, valid_df, alpha=0.7)
print(f"\nDefault Hybrid Metrics: {metrics}")

# Test 2: Compare different alpha values to find the best combination
print("\n" + "="*60)
print("TEST 2: Comparing Different Alpha Values")
print("="*60)
alpha_results = compare_alpha_values(train_df, valid_df, alpha_values=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

print("\n\nAlpha Comparison Results:")
print(alpha_results[['Alpha', 'Precision', 'Recall', 'F1-Score', 'AUC', 'AUPRC']].to_string(index=False))

elapsed = time.time() - start
print(f"\n\nTotal test time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print("✓ If you see this message with no errors, the Hybrid Model works correctly!")