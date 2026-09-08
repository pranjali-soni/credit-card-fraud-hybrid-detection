"""
Deep Learning models: DNN and LSTM for credit card fraud detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

from config import TARGET, PREDICTORS, RANDOM_STATE, PLOTS_DIR
from model_training import calculate_metrics, plot_confusion_matrix

import tensorflow as tf
tf.random.set_seed(RANDOM_STATE)


def train_dnn(train_df, valid_df, epochs=20, batch_size=2048):
    """
    Train a Deep Neural Network (DNN) Classifier
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training Deep Neural Network (DNN)...")
    print("="*50)
    
    # Scale features (neural networks need scaled input)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[PREDICTORS])
    X_valid_scaled = scaler.transform(valid_df[PREDICTORS])
    
    y_train = train_df[TARGET].values
    y_valid = valid_df[TARGET].values
    
    # Build the DNN architecture
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: probability of fraud (0 to 1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train the model
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_valid_scaled, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Predict probabilities
    proba = model.predict(X_valid_scaled, verbose=0).flatten()
    
    metrics, preds = calculate_metrics(y_valid, proba)
    
    # Plot confusion matrix
    cm = pd.crosstab(y_valid, preds, rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "DNN")
    
    # Plot training history
    plot_training_history(history, "DNN")
    
    print(f"DNN -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return model, preds, metrics


def train_lstm(train_df, valid_df, epochs=15, batch_size=2048):
    """
    Train an LSTM (Long Short-Term Memory) Classifier
    LSTM expects sequential input, so we reshape each transaction's features
    into a sequence of length 1 (treating each feature as a "timestep").
    
    Returns:
        tuple: (model, predictions, metrics_dict)
    """
    print("\n" + "="*50)
    print("Training LSTM Classifier...")
    print("="*50)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[PREDICTORS])
    X_valid_scaled = scaler.transform(valid_df[PREDICTORS])
    
    y_train = train_df[TARGET].values
    y_valid = valid_df[TARGET].values
    
    # Reshape for LSTM: (samples, timesteps, features)
    # We treat each of the 30 features as one timestep in a sequence
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_valid_lstm = X_valid_scaled.reshape((X_valid_scaled.shape[0], X_valid_scaled.shape[1], 1))
    
    # Build LSTM architecture
    model = keras.Sequential([
        layers.Input(shape=(X_train_lstm.shape[1], 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_valid_lstm, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    proba = model.predict(X_valid_lstm, verbose=0).flatten()
    
    metrics, preds = calculate_metrics(y_valid, proba)
    
    cm = pd.crosstab(y_valid, preds, rownames=['Actual'], colnames=['Predicted'])
    plot_confusion_matrix(cm, "LSTM")
    
    plot_training_history(history, "LSTM")
    
    print(f"LSTM -> AUC: {metrics['AUC']:.4f} | Precision: {metrics['Precision']:.4f} | "
          f"Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")
    
    return model, preds, metrics


def plot_training_history(history, model_name, save=True):
    """
    Plot training loss/accuracy curves for a Keras model
    
    Args:
        history: Keras History object returned by model.fit()
        model_name (str): Name of the model (for the plot title/filename)
        save (bool): Whether to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/training_history_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()