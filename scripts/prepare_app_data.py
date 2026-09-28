"""
Pre-compute and save all data needed for the full-depth Streamlit app:
- SHAP values for a sample (for global explanation)
- Cost curve data
- Drift test results
- Alpha tuning results
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from data_loader import load_data
from data_preprocessing import prepare_data
from hybrid_model import train_isolation_forest, train_xgboost_for_hybrid, get_anomaly_scores, compare_alpha_values
from threshold_optimizer import find_optimal_threshold
from drift_testing import create_temporal_split, analyze_feature_drift
from model_training import calculate_metrics
from config import TARGET, PREDICTORS, RANDOM_STATE, MAX_ROUNDS, EARLY_STOP
import os

os.makedirs('../saved_models', exist_ok=True)

print("Loading data...")
data_df = load_data()
train_df, valid_df, test_df = prepare_data(data_df)

# --- 1. SHAP sample data for global explanation ---
print("\nPreparing SHAP sample data...")
sample_df = valid_df.sample(n=1000, random_state=RANDOM_STATE).reset_index(drop=True)
sample_df.to_csv('../saved_models/shap_sample.csv', index=False)

# --- 2. Cost curve data ---
print("Computing cost curve data...")
dtrain = xgb.DMatrix(train_df[PREDICTORS], train_df[TARGET].values)
dvalid = xgb.DMatrix(valid_df[PREDICTORS], valid_df[TARGET].values)
params = {'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
          'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc', 'random_state': RANDOM_STATE}
xgb_model_full = xgb.train(params, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid, 'valid')],
                            early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)
proba_full = xgb_model_full.predict(dvalid)
best_threshold, cost_results_df = find_optimal_threshold(valid_df[TARGET].values, proba_full, cost_fn=5000, cost_fp=50)
cost_results_df.to_csv('../saved_models/cost_curve_data.csv', index=False)

# --- 3. Drift test data ---
print("Computing drift test data...")
early_df, late_df = create_temporal_split(data_df, split_ratio=0.7)
early_train, early_valid = train_test_split(early_df, test_size=0.2, random_state=RANDOM_STATE, stratify=early_df[TARGET])

dtrain_d = xgb.DMatrix(early_train[PREDICTORS], early_train[TARGET].values)
dvalid_d = xgb.DMatrix(early_valid[PREDICTORS], early_valid[TARGET].values)
dtest_d = xgb.DMatrix(late_df[PREDICTORS], late_df[TARGET].values)
xgb_drift_model = xgb.train(params, dtrain_d, MAX_ROUNDS, [(dtrain_d, 'train'), (dvalid_d, 'valid')],
                             early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)
proba_same = xgb_drift_model.predict(dvalid_d)
proba_drift = xgb_drift_model.predict(dtest_d)
metrics_same, _ = calculate_metrics(early_valid[TARGET].values, proba_same)
metrics_drift, _ = calculate_metrics(late_df[TARGET].values, proba_drift)

drift_summary = pd.DataFrame([
    {'Period': 'Same-Period (Validation)', 'AUC': metrics_same['AUC'], 'F1': metrics_same['F1-Score'], 'Recall': metrics_same['Recall']},
    {'Period': 'Later-Period (Drift Test)', 'AUC': metrics_drift['AUC'], 'F1': metrics_drift['F1-Score'], 'Recall': metrics_drift['Recall']}
])
drift_summary.to_csv('../saved_models/drift_summary.csv', index=False)

psi_df = analyze_feature_drift(early_df, late_df, top_n=10)
psi_df.to_csv('../saved_models/psi_results.csv', index=False)

# --- 4. Alpha tuning results ---
print("Computing alpha tuning results...")
alpha_results = compare_alpha_values(train_df, valid_df, alpha_values=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
alpha_results.to_csv('../saved_models/alpha_results.csv', index=False)

print("\n✓ All app data prepared and saved to ../saved_models/")