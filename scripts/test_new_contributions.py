"""
Test script for the 3 new contributions:
1. Multi-Model Consensus Scoring
2. Fairness Audit
3. Business Cost-Impact Dashboard
"""

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from data_preprocessing import prepare_data
from consensus_scoring import calculate_consensus_score, evaluate_consensus_triage, plot_consensus_distribution
from fairness_audit import audit_fairness_by_amount, plot_fairness_audit
from cost_impact_dashboard import compare_business_impact_across_models, plot_business_impact
from config import TARGET, PREDICTORS, RANDOM_STATE, MAX_ROUNDS, EARLY_STOP

print("Loading and preparing data...")
data_df = load_data()
train_df, valid_df, test_df = prepare_data(data_df)

y_valid = valid_df[TARGET].values
amounts_valid = valid_df['Amount'].values

# ============================================================
# Train 3 quick models to get diverse predictions for consensus scoring
# ============================================================
print("\nTraining 3 models to generate predictions for consensus scoring...")

# Model 1: XGBoost
dtrain = xgb.DMatrix(train_df[PREDICTORS], train_df[TARGET].values)
dvalid = xgb.DMatrix(valid_df[PREDICTORS], valid_df[TARGET].values)
params = {'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
          'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc', 'random_state': RANDOM_STATE}
xgb_model = xgb.train(params, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid, 'valid')],
                       early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)
xgb_proba = xgb_model.predict(dvalid)

# Model 2: Random Forest
rf = RandomForestClassifier(n_jobs=4, random_state=RANDOM_STATE, n_estimators=100)
rf.fit(train_df[PREDICTORS], train_df[TARGET].values)
rf_proba = rf.predict_proba(valid_df[PREDICTORS])[:, 1]

# Model 3: Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[PREDICTORS])
X_valid_scaled = scaler.transform(valid_df[PREDICTORS])
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, train_df[TARGET].values)
lr_proba = lr.predict_proba(X_valid_scaled)[:, 1]

predictions_dict = {
    'XGBoost': xgb_proba,
    'RandomForest': rf_proba,
    'LogisticRegression': lr_proba
}

print("✓ All 3 models trained")

# ============================================================
# CONTRIBUTION 1: Multi-Model Consensus Scoring
# ============================================================
consensus_df = calculate_consensus_score(predictions_dict, threshold=0.5)
triage_breakdown = evaluate_consensus_triage(consensus_df, y_valid)
plot_consensus_distribution(consensus_df)
print("✓ Consensus scoring complete, chart saved")

# ============================================================
# CONTRIBUTION 2: Fairness Audit (using XGBoost predictions)
# ============================================================
xgb_pred_labels = (xgb_proba >= 0.5).astype(int)
fairness_df = audit_fairness_by_amount(y_valid, xgb_pred_labels, amounts_valid)
plot_fairness_audit(fairness_df)
print("✓ Fairness audit complete, chart saved")

# ============================================================
# CONTRIBUTION 3: Business Cost-Impact Dashboard
# ============================================================
business_df = compare_business_impact_across_models(
    predictions_dict, y_valid, avg_fraud_amount=5000, false_alarm_cost=50
)
plot_business_impact(business_df)
print("✓ Business cost-impact dashboard complete, chart saved")

print("\n\n✓✓✓ ALL 3 NEW CONTRIBUTIONS TESTED SUCCESSFULLY ✓✓✓")