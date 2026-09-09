"""
Main script to run the complete credit card fraud detection analysis
Compares model performance WITHOUT SMOTE vs WITH SMOTE
"""

import warnings
warnings.filterwarnings('ignore')

from threshold_optimizer import find_optimal_threshold, plot_cost_curve
from hybrid_model import train_hybrid_model
from config import TARGET
from data_loader import load_data, get_data_summary
from eda import generate_all_eda_plots
from data_preprocessing import prepare_data
from smote_handler import apply_smote
from model_training import (train_random_forest, train_adaboost, train_catboost,
                            train_xgboost, train_lightgbm, train_lightgbm_cv,
                            train_logistic_regression, train_naive_bayes,
                            train_knn, train_decision_tree, train_svm)
from deep_learning_models import train_dnn, train_lstm
from model_evaluation import (create_results_summary, plot_model_comparison,
                              generate_evaluation_report, save_results_to_csv,
                              plot_before_after_smote)
import time


def train_all_models(train_df, valid_df, test_df, label="", include_deep_learning=True):
    """
    Train all models on the given training data and return their metrics.
    
    Args:
        train_df, valid_df, test_df: Data splits
        label (str): Label to identify this run (e.g., "No SMOTE" or "With SMOTE")
        include_deep_learning (bool): Whether to also train DNN and LSTM (slower)
        
    Returns:
        dict: Results dictionary {model_name: metrics_dict}
    """
    print(f"\n{'#'*70}")
    print(f"# TRAINING ALL MODELS - {label}")
    print(f"{'#'*70}")
    
    results = {}
    
    _, _, rf_metrics = train_random_forest(train_df, valid_df)
    results['RandomForest'] = rf_metrics
    
    _, _, ada_metrics = train_adaboost(train_df, valid_df)
    results['AdaBoost'] = ada_metrics
    
    _, _, cat_metrics = train_catboost(train_df, valid_df)
    results['CatBoost'] = cat_metrics
    
    _, _, xgb_metrics = train_xgboost(train_df, valid_df, test_df)
    results['XGBoost'] = xgb_metrics
    
    _, _, lgb_metrics = train_lightgbm(train_df, valid_df, test_df)
    results['LightGBM'] = lgb_metrics
    
    _, _, lgb_cv_metrics = train_lightgbm_cv(train_df, test_df)
    results['LightGBM_CV'] = lgb_cv_metrics
    
    _, _, lr_metrics = train_logistic_regression(train_df, valid_df)
    results['LogisticRegression'] = lr_metrics
    
    _, _, nb_metrics = train_naive_bayes(train_df, valid_df)
    results['NaiveBayes'] = nb_metrics
    
    _, _, knn_metrics = train_knn(train_df, valid_df)
    results['KNN'] = knn_metrics
    
    _, _, dt_metrics = train_decision_tree(train_df, valid_df)
    results['DecisionTree'] = dt_metrics
    
    _, _, svm_metrics = train_svm(train_df, valid_df)
    results['SVM'] = svm_metrics
    
    if include_deep_learning:
        _, _, dnn_metrics = train_dnn(train_df, valid_df)
        results['DNN'] = dnn_metrics
        
        _, _, lstm_metrics = train_lstm(train_df, valid_df)
        results['LSTM'] = lstm_metrics
    
    # Hybrid Model (our novel contribution)
    _, _, hybrid_proba, hybrid_metrics = train_hybrid_model(train_df, valid_df, alpha=0.7)
    results['HybridModel (Ours)'] = hybrid_metrics
    
    # Cost-Sensitive Threshold Optimization applied to our Hybrid Model
    print("\n[BONUS] Applying Cost-Sensitive Threshold Optimization to Hybrid Model...")
    y_valid_true = valid_df[TARGET].values
    best_threshold, cost_results_df = find_optimal_threshold(
        y_valid_true, hybrid_proba, cost_fn=5000, cost_fp=50
    )
    plot_cost_curve(cost_results_df, model_name=f"HybridModel_{label.replace(' ', '_')}")
    
    return results


def main():
    """
    Main function to execute the complete analysis pipeline
    """
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - PREDICTIVE MODELS")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Step 1: Load Data
    print("\n[STEP 1/7] Loading Data...")
    data_df = load_data()
    
    summary = get_data_summary(data_df)
    print(f"\nDataset Summary:")
    print(f"  - Total Transactions: {summary['shape'][0]:,}")
    print(f"  - Total Features: {summary['shape'][1]}")
    print(f"  - Fraudulent Transactions: {summary['fraud_count']:,} ({summary['fraud_percentage']:.3f}%)")
    print(f"  - Missing Values: {summary['missing_values']}")
    
    # Step 2: Exploratory Data Analysis
    print("\n[STEP 2/7] Performing Exploratory Data Analysis...")
    generate_all_eda_plots(data_df)
    
    # Step 3: Data Preprocessing
    print("\n[STEP 3/7] Preparing Data...")
    train_df, valid_df, test_df = prepare_data(data_df)
    
    # Step 4: Train models WITHOUT SMOTE (baseline)
    print("\n[STEP 4/7] Training Models WITHOUT SMOTE (Baseline)...")
    results_no_smote = train_all_models(train_df, valid_df, test_df, label="WITHOUT SMOTE")
    
    # Step 5: Apply SMOTE and train models again
    print("\n[STEP 5/7] Applying SMOTE and Retraining Models...")
    train_df_smote = apply_smote(train_df)
    results_with_smote = train_all_models(train_df_smote, valid_df, test_df, label="WITH SMOTE")
    
    # Step 6: Evaluate Both Sets of Results
    print("\n[STEP 6/7] Evaluating Models...")
    
    print("\n\n" + "#"*100)
    print("# RESULTS WITHOUT SMOTE")
    print("#"*100)
    report_no_smote, results_df_no_smote = generate_evaluation_report(results_no_smote)
    print(report_no_smote)
    
    print("\n\n" + "#"*100)
    print("# RESULTS WITH SMOTE")
    print("#"*100)
    report_with_smote, results_df_with_smote = generate_evaluation_report(results_with_smote)
    print(report_with_smote)
    
    # Step 7: Save Results
    print("\n[STEP 7/7] Saving Results...")
    
    plot_model_comparison(results_df_no_smote)
    save_results_to_csv(results_df_no_smote, filename='model_results_no_smote.csv')
    
    plot_model_comparison(results_df_with_smote)
    save_results_to_csv(results_df_with_smote, filename='model_results_with_smote.csv')
    
    plot_before_after_smote(results_df_no_smote, results_df_with_smote, metric='F1-Score')
    plot_before_after_smote(results_df_no_smote, results_df_with_smote, metric='Recall')
    plot_before_after_smote(results_df_no_smote, results_df_with_smote, metric='Precision')
    plot_before_after_smote(results_df_no_smote, results_df_with_smote, metric='AUPRC')
    print("  ✓ Before/After SMOTE comparison charts saved")
    
    # Final Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nTotal Execution Time: {elapsed_time/60:.2f} minutes")
    print(f"Best Model WITHOUT SMOTE: {results_df_no_smote.iloc[0]['Model']} (AUC: {results_df_no_smote.iloc[0]['AUC']:.4f})")
    print(f"Best Model WITH SMOTE: {results_df_with_smote.iloc[0]['Model']} (AUC: {results_df_with_smote.iloc[0]['AUC']:.4f})")
    print(f"\nAll plots saved to: ./plots/")
    print(f"Results saved to: ./plots/model_results_no_smote.csv and ./plots/model_results_with_smote.csv")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()