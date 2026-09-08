"""
Quick isolated test for DNN and LSTM models
Uses a smaller sample of data to verify everything works before running the full pipeline
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data
from data_preprocessing import prepare_data
from deep_learning_models import train_dnn, train_lstm
import time

print("Loading data...")
data_df = load_data()

print("Preparing data...")
train_df, valid_df, test_df = prepare_data(data_df)

# Take a smaller sample for quick testing
print("\nTaking a smaller sample for quick testing...")
train_sample = train_df.sample(n=20000, random_state=42)
valid_sample = valid_df.sample(n=5000, random_state=42)

print(f"Train sample size: {len(train_sample)}")
print(f"Valid sample size: {len(valid_sample)}")

start = time.time()

print("\n" + "="*60)
print("TESTING DNN (quick run, 5 epochs)")
print("="*60)
_, _, dnn_metrics = train_dnn(train_sample, valid_sample, epochs=5, batch_size=512)
print(f"\nDNN Test Metrics: {dnn_metrics}")

print("\n" + "="*60)
print("TESTING LSTM (quick run, 5 epochs)")
print("="*60)
_, _, lstm_metrics = train_lstm(train_sample, valid_sample, epochs=5, batch_size=512)
print(f"\nLSTM Test Metrics: {lstm_metrics}")

elapsed = time.time() - start
print(f"\n\nTotal test time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print("✓ If you see this message with no errors, both models work correctly!")