"""
Quick isolated test for Cost-Sensitive Threshold Optimization
Uses XGBoost's predictions as an example
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data
from data_preprocessing import prepare_data
from threshold_optimizer import find_optimal_threshold, plot_cost_curve
import xgboost as xgb
from config import PREDICTORS, TARGET, RANDOM_STATE, MAX_ROUNDS, EARLY_STOP

print("Loading data...")
data_df = load_data()

print("Preparing data...")
train_df, valid_df, test_df = prepare_data(data_df)

print("\nTraining a quick XGBoost model to get predictions...")
dtrain = xgb.DMatrix(train_df[PREDICTORS], train_df[TARGET].values)
dvalid = xgb.DMatrix(valid_df[PREDICTORS], valid_df[TARGET].values)

params = {
    'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
    'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc',
    'random_state': RANDOM_STATE
}

model = xgb.train(params, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid, 'valid')],
                   early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)

proba = model.predict(dvalid)
y_true = valid_df[TARGET].values

print("\nFinding optimal threshold...")
best_threshold, results_df = find_optimal_threshold(y_true, proba, cost_fn=5000, cost_fp=50)

plot_cost_curve(results_df, model_name="XGBoost_Test")

print(f"\n✓ Test complete! Optimal threshold found: {best_threshold:.2f}")
print("Check plots/cost_curve_xgboost_test.png to see the cost curve")