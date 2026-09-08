"""
Main script to run the complete credit card fraud detection analysis
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data, get_data_summary
from eda import generate_all_eda_plots
from data_preprocessing import prepare_data
from model_training import (train_random_forest, train_adaboost, train_catboost,
                            train_xgboost, train_lightgbm, train_lightgbm_cv)
from model_evaluation import (create_results_summary, plot_model_comparison,
                              generate_evaluation_report, save_results_to_csv)
import time


def main():
    """
    Main function to execute the complete analysis pipeline
    """
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - PREDICTIVE MODELS")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Step 1: Load Data
    print("\n[STEP 1/6] Loading Data...")
    data_df = load_data()
    
    summary = get_data_summary(data_df)
    print(f"\nDataset Summary:")
    print(f"  - Total Transactions: {summary['shape'][0]:,}")
    print(f"  - Total Features: {summary['shape'][1]}")
    print(f"  - Fraudulent Transactions: {summary['fraud_count']:,} ({summary['fraud_percentage']:.3f}%)")
    print(f"  - Missing Values: {summary['missing_values']}")
    
    # Step 2: Exploratory Data Analysis
    print("\n[STEP 2/6] Performing Exploratory Data Analysis...")
    generate_all_eda_plots(data_df)
    
    # Step 3: Data Preprocessing
    print("\n[STEP 3/6] Preparing Data...")
    train_df, valid_df, test_df = prepare_data(data_df)
    
    # Step 4: Model Training
    print("\n[STEP 4/6] Training Models...")
    results = {}
    
    # Train RandomForest
    _, _, rf_metrics = train_random_forest(train_df, valid_df)
    results['RandomForest'] = rf_metrics
    
    # Train AdaBoost
    _, _, ada_metrics = train_adaboost(train_df, valid_df)
    results['AdaBoost'] = ada_metrics
    
    # Train CatBoost
    _, _, cat_metrics = train_catboost(train_df, valid_df)
    results['CatBoost'] = cat_metrics
    
    # Train XGBoost
    _, _, xgb_metrics = train_xgboost(train_df, valid_df, test_df)
    results['XGBoost'] = xgb_metrics
    
    # Train LightGBM
    _, _, lgb_metrics = train_lightgbm(train_df, valid_df, test_df)
    results['LightGBM'] = lgb_metrics
    
    # Train LightGBM with CV
    _, _, lgb_cv_metrics = train_lightgbm_cv(train_df, test_df)
    results['LightGBM_CV'] = lgb_cv_metrics
    
    # Step 5: Model Evaluation
    print("\n[STEP 5/6] Evaluating Models...")
    report, results_df = generate_evaluation_report(results)
    print(report)
    
    # Step 6: Save Results
    print("\n[STEP 6/6] Saving Results...")
    plot_model_comparison(results_df)
    print("  ✓ Model comparison plot saved")
    
    save_results_to_csv(results_df)
    print("  ✓ Results CSV saved")
    
    # Final Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nTotal Execution Time: {elapsed_time/60:.2f} minutes")
    print(f"Best Model (by AUC): {results_df.iloc[0]['Model']} (AUC: {results_df.iloc[0]['AUC']:.4f})")
    print(f"\nAll plots saved to: ./plots/")
    print(f"Results saved to: ./plots/model_results.csv")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()