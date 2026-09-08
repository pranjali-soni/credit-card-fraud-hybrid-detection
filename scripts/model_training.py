"""
Model training module for all classifiers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                             f1_score, average_precision_score, accuracy_score)
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import gc
import os

from config import (RFC_METRIC, NUM_ESTIMATORS, NO_JOBS, RANDOM_STATE, 
                   MAX_ROUNDS, EARLY_STOP, VERBOSE_EVAL, NUMBER_KFOLDS,
                   TARGET, PREDICTORS, PLOTS_DIR)


def calculate_metrics(y_true, y_proba, threshold=0.5):
    """
    Calculate a full set of evaluation metrics from true labels and predicted probabilities.
    
    Args:
        y_true (array): Actual labels (0 or 1)
        y_proba (array): Predicted probability of fraud (class 1)
        threshold (float): Cutoff to convert probability into a hard 0/1 prediction
        
    Returns:
        dict: Dictionary containing Accuracy, Precision, Recall, F1, AUC, AUPRC
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_proba),
        'AUPRC': average_precision_score(y_true, y_proba)
    }
    
    return metrics, y_pred


def plot_confusion_matrix(cm, model_name, save=True):
    """
    Plot confusion matrix
    
    Args:
        cm (pd.DataFrame): Confusion matrix
        model_name (str): Name of the model
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, 
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'],
                annot=True, ax=ax,
                linewidths=.2, linecolor="Darkblue", cmap="Blues")
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importance(importances, features, model_name, save=True):
    """
    Plot feature importance
    
    Args:
        importances (array): Feature importance values
        features (list): Feature names
        model_name (str): Name of the model
        save (bool): Whether to save the plot
    """
    tmp = pd.DataFrame({'Feature': features, 'Importance': importances})
    tmp = tmp.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(7, 4))
    plt.title(f'Feature Importance - {model_name}', fontsize=14)
    s = sns.barplot(x='Feature', y='Importance', data=tmp)
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/feature_importance_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def train_random_forest(train_df, valid_df):
    """
    Train Random Forest Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training RandomForest Classifier...")
    print("="*50)
    
    clf = RandomForestClassifier(
        n_jobs=NO_JOBS,
        random_state=RANDOM_STATE,
        criterion=RFC_METRIC,
        n_estimators=NUM_ESTIMATORS,
        verbose=False
    )
    
    clf.fit(train_df[PREDICTORS], train_df[TARGET].values)
    proba = clf.predict_proba(valid_df[PREDICTORS])[:, 1]
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, proba)
    
    # Plot feature importance
    plot_feature_importance(clf.feature_importances_, PREDICTORS, "RandomForest")
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds, 
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "RandomForest")
    
    print(f"RandomForest -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return clf, preds, metrics


def train_adaboost(train_df, valid_df):
    """
    Train AdaBoost Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training AdaBoost Classifier...")
    print("="*50)
    
    clf = AdaBoostClassifier(
        random_state=RANDOM_STATE,
        learning_rate=0.8,
        n_estimators=NUM_ESTIMATORS
    )
    
    clf.fit(train_df[PREDICTORS], train_df[TARGET].values)
    proba = clf.predict_proba(valid_df[PREDICTORS])[:, 1]
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, proba)
    
    # Plot feature importance
    plot_feature_importance(clf.feature_importances_, PREDICTORS, "AdaBoost")
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "AdaBoost")
    
    print(f"AdaBoost -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return clf, preds, metrics


def train_catboost(train_df, valid_df):
    """
    Train CatBoost Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training CatBoost Classifier...")
    print("="*50)
    
    clf = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=12,
        eval_metric='AUC',
        random_seed=RANDOM_STATE,
        bagging_temperature=0.2,
        od_type='Iter',
        metric_period=VERBOSE_EVAL,
        od_wait=100,
        verbose=False
    )
    
    clf.fit(train_df[PREDICTORS], train_df[TARGET].values)
    proba = clf.predict_proba(valid_df[PREDICTORS])[:, 1]
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, proba)
    
    # Plot feature importance
    plot_feature_importance(clf.feature_importances_, PREDICTORS, "CatBoost")
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "CatBoost")
    
    print(f"CatBoost -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return clf, preds, metrics


def train_xgboost(train_df, valid_df, test_df):
    """
    Train XGBoost Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training XGBoost Classifier...")
    print("="*50)
    
    # Prepare DMatrix
    dtrain = xgb.DMatrix(train_df[PREDICTORS], train_df[TARGET].values)
    dvalid = xgb.DMatrix(valid_df[PREDICTORS], valid_df[TARGET].values)
    dtest = xgb.DMatrix(test_df[PREDICTORS], test_df[TARGET].values)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eta': 0.039,
        'max_depth': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'eval_metric': 'auc',
        'random_state': RANDOM_STATE
    }
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        MAX_ROUNDS,
        watchlist,
        early_stopping_rounds=EARLY_STOP,
        maximize=True,
        verbose_eval=VERBOSE_EVAL
    )
    
    # Predict on test set (these are already probabilities, since objective is binary:logistic)
    proba = model.predict(dtest)
    
    metrics, preds = calculate_metrics(test_df[TARGET].values, proba)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    xgb.plot_importance(model, height=0.8, title="Feature Importance - XGBoost", 
                       ax=ax, color="green")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f'{PLOTS_DIR}/feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    cm = pd.crosstab(test_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "XGBoost")
    
    print(f"XGBoost -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return model, preds, metrics


def train_lightgbm(train_df, valid_df, test_df):
    """
    Train LightGBM Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training LightGBM Classifier...")
    print("="*50)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 7,
        'max_depth': 4,
        'min_child_samples': 100,
        'max_bin': 100,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'min_child_weight': 0,
        'min_split_gain': 0,
        'nthread': 8,
        'verbose': -1,
        'scale_pos_weight': 150,
        'random_state': RANDOM_STATE,
        'bagging_seed': RANDOM_STATE,
        'feature_fraction_seed': RANDOM_STATE,
    }
    
    dtrain = lgb.Dataset(train_df[PREDICTORS].values,
                        label=train_df[TARGET].values,
                        feature_name=PREDICTORS)
    
    dvalid = lgb.Dataset(valid_df[PREDICTORS].values,
                        label=valid_df[TARGET].values,
                        feature_name=PREDICTORS)
    
    evals_results = {}
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        num_boost_round=MAX_ROUNDS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=2*EARLY_STOP),
            lgb.log_evaluation(period=VERBOSE_EVAL),
            lgb.record_evaluation(evals_results)
        ]
    )
    
    # Predict on test set (already probabilities)
    proba = model.predict(test_df[PREDICTORS])
    
    metrics, preds = calculate_metrics(test_df[TARGET].values, proba)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    lgb.plot_importance(model, height=0.8, title="Feature Importance - LightGBM",
                       ax=ax, color="red")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f'{PLOTS_DIR}/feature_importance_lightgbm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    cm = pd.crosstab(test_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "LightGBM")
    
    print(f"LightGBM -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return model, preds, metrics


def train_lightgbm_cv(train_df, test_df):
    """
    Train LightGBM with Cross-Validation
    
    Returns:
        tuple: (oof_predictions, test_predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training LightGBM with Cross-Validation...")
    print("="*50)
    
    kf = KFold(n_splits=NUMBER_KFOLDS, random_state=RANDOM_STATE, shuffle=True)
    
    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), 1):
        print(f"\nFold {fold}/{NUMBER_KFOLDS}")
        
        train_x = train_df[PREDICTORS].iloc[train_idx]
        train_y = train_df[TARGET].iloc[train_idx]
        valid_x = train_df[PREDICTORS].iloc[valid_idx]
        valid_y = train_df[TARGET].iloc[valid_idx]
        
        model = LGBMClassifier(
            nthread=-1,
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=80,
            colsample_bytree=0.98,
            subsample=0.78,
            reg_alpha=0.04,
            reg_lambda=0.073,
            subsample_for_bin=50,
            boosting_type='gbdt',
            is_unbalance=False,
            min_split_gain=0.025,
            min_child_weight=40,
            min_child_samples=510,
            objective='binary',
            metric='auc',
            verbose=-1,
            random_state=RANDOM_STATE
        )
        
        model.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc',
            callbacks=[
                lgb.log_evaluation(period=VERBOSE_EVAL),
                lgb.early_stopping(stopping_rounds=EARLY_STOP)
            ]
        )
        
        oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
        test_preds += model.predict_proba(test_df[PREDICTORS], num_iteration=model.best_iteration_)[:, 1] / NUMBER_KFOLDS
        
        fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
        print(f'Fold {fold} AUC: {fold_auc:.6f}')
        
        del model, train_x, train_y, valid_x, valid_y
        gc.collect()
    
    metrics, final_preds = calculate_metrics(test_df[TARGET].values, test_preds)
    
    # Plot confusion matrix
    cm = pd.crosstab(test_df[TARGET].values, final_preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "LightGBM_CV")
    
    print(f"\nLightGBM(CV) -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return oof_preds, test_preds, metrics
def train_logistic_regression(train_df, valid_df):
    """
    Train Logistic Regression Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training Logistic Regression Classifier...")
    print("="*50)
    
    # Scale features (Logistic Regression is sensitive to feature scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[PREDICTORS])
    X_valid_scaled = scaler.transform(valid_df[PREDICTORS])
    
    clf = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'
    )
    
    clf.fit(X_train_scaled, train_df[TARGET].values)
    proba = clf.predict_proba(X_valid_scaled)[:, 1]
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, proba)
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "LogisticRegression")
    
    print(f"Logistic Regression -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return clf, preds, metrics


def train_naive_bayes(train_df, valid_df):
    """
    Train Gaussian Naive Bayes Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training Naive Bayes Classifier...")
    print("="*50)
    
    clf = GaussianNB()
    
    clf.fit(train_df[PREDICTORS], train_df[TARGET].values)
    proba = clf.predict_proba(valid_df[PREDICTORS])[:, 1]
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, proba)
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "NaiveBayes")
    
    print(f"Naive Bayes -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return clf, preds, metrics


def train_knn(train_df, valid_df, sample_size=50000):
    """
    Train K-Nearest Neighbors Classifier
    Note: KNN is trained on a sample of the training data for speed, 
    since KNN is computationally expensive on large datasets.
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training KNN Classifier...")
    print("="*50)
    
    # Scale features (KNN uses distance, so scaling matters a lot)
    scaler = StandardScaler()
    
    # Sample training data for speed (KNN is slow on 180k+ rows)
    if len(train_df) > sample_size:
        train_sample = train_df.sample(n=sample_size, random_state=RANDOM_STATE)
    else:
        train_sample = train_df
    
    X_train_scaled = scaler.fit_transform(train_sample[PREDICTORS])
    X_valid_scaled = scaler.transform(valid_df[PREDICTORS])
    
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=NO_JOBS)
    
    clf.fit(X_train_scaled, train_sample[TARGET].values)
    proba = clf.predict_proba(X_valid_scaled)[:, 1]
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, proba)
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "KNN")
    
    print(f"KNN -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return clf, preds, metrics


def train_decision_tree(train_df, valid_df):
    """
    Train Decision Tree Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training Decision Tree Classifier...")
    print("="*50)
    
    clf = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=10,
        class_weight='balanced'
    )
    
    clf.fit(train_df[PREDICTORS], train_df[TARGET].values)
    proba = clf.predict_proba(valid_df[PREDICTORS])[:, 1]
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, proba)
    
    # Plot feature importance
    plot_feature_importance(clf.feature_importances_, PREDICTORS, "DecisionTree")
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "DecisionTree")
    
    print(f"Decision Tree -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return clf, preds, metrics


def train_svm(train_df, valid_df, sample_size=20000):
    """
    Train Support Vector Machine Classifier
    Note: SVM is trained on a sample of the training data for speed,
    since SVM scales poorly with large datasets.
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training SVM Classifier...")
    print("="*50)
    
    # Scale features (SVM is very sensitive to feature scale)
    scaler = StandardScaler()
    
    # Sample training data for speed (SVM is very slow on large datasets)
    if len(train_df) > sample_size:
        train_sample = train_df.sample(n=sample_size, random_state=RANDOM_STATE)
    else:
        train_sample = train_df
    
    X_train_scaled = scaler.fit_transform(train_sample[PREDICTORS])
    X_valid_scaled = scaler.transform(valid_df[PREDICTORS])
    
    clf = SVC(
        kernel='rbf',
        probability=True,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    
    clf.fit(X_train_scaled, train_sample[TARGET].values)
    proba = clf.predict_proba(X_valid_scaled)[:, 1]
    
    metrics, preds = calculate_metrics(valid_df[TARGET].values, proba)
    
    # Plot confusion matrix
    cm = pd.crosstab(valid_df[TARGET].values, preds,
                     rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "SVM")
    
    print(f"SVM -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return clf, preds, metrics