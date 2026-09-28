"""
Train and save the final models to disk for use in the Streamlit demo.
This avoids retraining every time the app starts.
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from data_preprocessing import prepare_data
from hybrid_model import train_isolation_forest, train_xgboost_for_hybrid
from shap_explainer import create_shap_explainer
from config import TARGET, PREDICTORS, RANDOM_STATE, MAX_ROUNDS, EARLY_STOP
import os

print("Loading data...")
data_df = load_data()

print("Preparing data...")
train_df, valid_df, test_df = prepare_data(data_df)

print("\nTraining final XGBoost model...")
dtrain = xgb.DMatrix(train_df[PREDICTORS], train_df[TARGET].values)
dvalid = xgb.DMatrix(valid_df[PREDICTORS], valid_df[TARGET].values)

params = {
    'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
    'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc',
    'random_state': RANDOM_STATE
}

xgb_model = xgb.train(params, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid, 'valid')],
                       early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)

print("✓ XGBoost trained")

print("\nTraining Isolation Forest...")
iso_forest = train_isolation_forest(train_df)

print("\nCreating SHAP explainer...")
explainer = create_shap_explainer(xgb_model)

# Create a saved_models directory
os.makedirs('../saved_models', exist_ok=True)

print("\nSaving models to disk...")

xgb_model.save_model('../saved_models/xgb_model.json')

with open('../saved_models/iso_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)

with open('../saved_models/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

# Save a sample of test data for the demo to use (so we have realistic transactions to pick from)
sample_for_demo = test_df.sample(n=500, random_state=RANDOM_STATE)
sample_for_demo.to_csv('../saved_models/demo_sample.csv', index=False)

print("\n✓ All models saved successfully to ../saved_models/")
print("  - xgb_model.json")
print("  - iso_forest.pkl")
print("  - shap_explainer.pkl")
print("  - demo_sample.csv (500 sample transactions for the demo)")