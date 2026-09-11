"""
Test script for SHAP Explainability
Trains an XGBoost model, generates global feature importance,
and explains a few individual fraud/genuine transactions in detail.
"""

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import numpy as np

from data_loader import load_data
from data_preprocessing import prepare_data
from shap_explainer import (create_shap_explainer, generate_global_explanation,
                            explain_single_transaction, generate_natural_language_explanation)
from config import TARGET, PREDICTORS, RANDOM_STATE, MAX_ROUNDS, EARLY_STOP

print("Loading data...")
data_df = load_data()

print("Preparing data...")
train_df, valid_df, test_df = prepare_data(data_df)

print("\nTraining XGBoost model for SHAP analysis...")
dtrain = xgb.DMatrix(train_df[PREDICTORS], train_df[TARGET].values)
dvalid = xgb.DMatrix(valid_df[PREDICTORS], valid_df[TARGET].values)

params = {
    'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
    'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc',
    'random_state': RANDOM_STATE
}

model = xgb.train(params, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid, 'valid')],
                   early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)

print("✓ Model trained")

# Create SHAP explainer
explainer = create_shap_explainer(model)

# Take a sample of the validation set for SHAP analysis (SHAP can be slow on huge datasets)
sample_size = 1000
X_sample = valid_df[PREDICTORS].sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
y_sample = valid_df.loc[X_sample.index, TARGET].reset_index(drop=True) if False else None

# Get corresponding labels properly (re-sample together to keep alignment)
sample_df = valid_df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
X_sample = sample_df[PREDICTORS]
y_sample = sample_df[TARGET]

print(f"\nUsing a sample of {sample_size} transactions for SHAP analysis...")

# Generate global explanation
shap_values = generate_global_explanation(explainer, X_sample)

# Find a few interesting transactions to explain individually:
# 1. A transaction that IS fraud
# 2. A transaction that is genuine
fraud_indices = sample_df[sample_df[TARGET] == 1].index.tolist()
genuine_indices = sample_df[sample_df[TARGET] == 0].index.tolist()

print(f"\nFound {len(fraud_indices)} fraud transactions and {len(genuine_indices)} genuine transactions in this sample")

if len(fraud_indices) > 0:
    fraud_idx = fraud_indices[0]
    print("\n" + "="*60)
    print("EXAMPLE 1: Explaining a FRAUD transaction")
    print("="*60)
    contributions = explain_single_transaction(
        explainer, X_sample, shap_values, fraud_idx, actual_label=1
    )
    explanation_text = generate_natural_language_explanation(contributions, "FRAUD")
    print(f"\n📝 Natural Language Explanation:\n{explanation_text}")

if len(genuine_indices) > 0:
    genuine_idx = genuine_indices[0]
    print("\n" + "="*60)
    print("EXAMPLE 2: Explaining a GENUINE transaction")
    print("="*60)
    contributions = explain_single_transaction(
        explainer, X_sample, shap_values, genuine_idx, actual_label=0
    )
    explanation_text = generate_natural_language_explanation(contributions, "GENUINE")
    print(f"\n📝 Natural Language Explanation:\n{explanation_text}")

print("\n\n✓ SHAP explainability test complete!")
print(f"Check the plots/ folder for: shap_global_importance_bar.png, shap_summary_beeswarm.png, "
      f"and individual transaction waterfall plots")