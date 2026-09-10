"""
Concept Drift Test: Compare XGBoost vs Hybrid Model performance
on same-period data vs a later, unseen time period.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from data_loader import load_data
from drift_testing import (create_temporal_split, plot_drift_comparison, 
                           analyze_feature_drift)
from model_training import calculate_metrics
from hybrid_model import train_isolation_forest, get_anomaly_scores, train_xgboost_for_hybrid
from config import TARGET, PREDICTORS, RANDOM_STATE, MAX_ROUNDS, EARLY_STOP


print("Loading data...")
data_df = load_data()

# Step 1: Create temporal split (early 70% vs late 30%)
early_df, late_df = create_temporal_split(data_df, split_ratio=0.7)

# Step 2: Split the early period into train/valid (same-period test)
early_train, early_valid = train_test_split(
    early_df, test_size=0.2, random_state=RANDOM_STATE, stratify=early_df[TARGET]
)

print(f"\nEarly Train: {len(early_train):,} | Early Valid (same-period test): {len(early_valid):,} "
      f"| Late Test (drift test): {len(late_df):,}")

# ============================================================
# TEST 1: XGBoost Alone - Same Period vs Drift
# ============================================================
print("\n" + "#"*60)
print("# TESTING: XGBoost Alone")
print("#"*60)

dtrain = xgb.DMatrix(early_train[PREDICTORS], early_train[TARGET].values)
dvalid_same = xgb.DMatrix(early_valid[PREDICTORS], early_valid[TARGET].values)
dtest_drift = xgb.DMatrix(late_df[PREDICTORS], late_df[TARGET].values)

params = {
    'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
    'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc',
    'random_state': RANDOM_STATE
}

xgb_model = xgb.train(params, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid_same, 'valid')],
                       early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)

proba_same = xgb_model.predict(dvalid_same)
proba_drift = xgb_model.predict(dtest_drift)

metrics_xgb_same, _ = calculate_metrics(early_valid[TARGET].values, proba_same)
metrics_xgb_drift, _ = calculate_metrics(late_df[TARGET].values, proba_drift)

print(f"\nXGBoost Same-Period  -> AUC: {metrics_xgb_same['AUC']:.4f} | F1: {metrics_xgb_same['F1-Score']:.4f} | Recall: {metrics_xgb_same['Recall']:.4f}")
print(f"XGBoost Later-Period -> AUC: {metrics_xgb_drift['AUC']:.4f} | F1: {metrics_xgb_drift['F1-Score']:.4f} | Recall: {metrics_xgb_drift['Recall']:.4f}")
print(f"AUC Drop: {metrics_xgb_same['AUC'] - metrics_xgb_drift['AUC']:.4f}")

# ============================================================
# TEST 2: Hybrid Model - Same Period vs Drift
# ============================================================
print("\n" + "#"*60)
print("# TESTING: Hybrid Model (Isolation Forest + XGBoost)")
print("#"*60)

iso_forest = train_isolation_forest(early_train)
xgb_hybrid_model = train_xgboost_for_hybrid(early_train, early_valid)

# Same-period predictions
anomaly_same = get_anomaly_scores(iso_forest, early_valid)
xgb_proba_same = xgb_hybrid_model.predict(xgb.DMatrix(early_valid[PREDICTORS]))
hybrid_proba_same = 0.7 * xgb_proba_same + 0.3 * anomaly_same

# Later-period (drift) predictions
anomaly_drift = get_anomaly_scores(iso_forest, late_df)
xgb_proba_drift = xgb_hybrid_model.predict(xgb.DMatrix(late_df[PREDICTORS]))
hybrid_proba_drift = 0.7 * xgb_proba_drift + 0.3 * anomaly_drift

metrics_hybrid_same, _ = calculate_metrics(early_valid[TARGET].values, hybrid_proba_same)
metrics_hybrid_drift, _ = calculate_metrics(late_df[TARGET].values, hybrid_proba_drift)

print(f"\nHybrid Same-Period  -> AUC: {metrics_hybrid_same['AUC']:.4f} | F1: {metrics_hybrid_same['F1-Score']:.4f} | Recall: {metrics_hybrid_same['Recall']:.4f}")
print(f"Hybrid Later-Period -> AUC: {metrics_hybrid_drift['AUC']:.4f} | F1: {metrics_hybrid_drift['F1-Score']:.4f} | Recall: {metrics_hybrid_drift['Recall']:.4f}")
print(f"AUC Drop: {metrics_hybrid_same['AUC'] - metrics_hybrid_drift['AUC']:.4f}")

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "#"*60)
print("# FINAL DRIFT ROBUSTNESS COMPARISON")
print("#"*60)

comparison_df = pd.DataFrame([
    {'Model': 'XGBoost (alone)', 
     'AUC_SamePeriod': metrics_xgb_same['AUC'], 'AUC_LaterPeriod': metrics_xgb_drift['AUC'],
     'AUC_Drop': metrics_xgb_same['AUC'] - metrics_xgb_drift['AUC']},
    {'Model': 'Hybrid Model (Ours)', 
     'AUC_SamePeriod': metrics_hybrid_same['AUC'], 'AUC_LaterPeriod': metrics_hybrid_drift['AUC'],
     'AUC_Drop': metrics_hybrid_same['AUC'] - metrics_hybrid_drift['AUC']}
])

print(comparison_df.to_string(index=False))

plot_drift_comparison(comparison_df)
print("\n✓ Drift comparison chart saved to plots/drift_comparison.png")

# ============================================================
# BONUS: Feature-level drift analysis (PSI)
# ============================================================
psi_df = analyze_feature_drift(early_df, late_df, top_n=10)

if comparison_df.loc[0, 'AUC_Drop'] > comparison_df.loc[1, 'AUC_Drop']:
    print("\n🏆 RESULT: Hybrid Model degrades LESS than XGBoost alone under concept drift!")
else:
    print("\n📊 RESULT: XGBoost alone degrades less than the Hybrid Model under concept drift.")

print("\n✓ Drift testing complete!")