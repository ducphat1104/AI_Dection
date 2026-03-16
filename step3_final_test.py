"""
Step 3: Final Test Evaluation & Prediction Script
This script tests the model on the unseen Test Set and provides a reusable prediction function.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
MODEL_DIR = '/Users/nguyenducphat/Projects/TT12/models'
DATASET_PATH = '/Users/nguyenducphat/Projects/TT12/cicids2017_cleaned.csv'

def load_essentials():
    print("Loading model and preprocessing objects...")
    model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_baseline.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    return model, scaler, le

def run_final_test():
    model, scaler, le = load_essentials()
    
    # Load and preprocess again (simulate real scenario)
    print("Loading dataset for final test...")
    df = pd.read_csv(DATASET_PATH)
    
    # Preprocessing (must match training)
    REDUNDANT_COLS = ['Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Average Packet Size', 'Fwd IAT Mean']
    TARGET = 'Attack Type'
    
    X = df.drop(columns=[TARGET] + REDUNDANT_COLS, errors='ignore')
    y = le.transform(df[TARGET])
    
    # Note: We use a fixed seed to get the same test split as step 2
    from sklearn.model_selection import train_test_split
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    print(f"Applying scaling to test set ({len(X_test):,} samples)...")
    X_test_s = scaler.transform(X_test)
    
    print("Evaluating...")
    y_pred = model.predict(X_test_s)
    
    print("\n" + "="*40)
    print("FINAL TEST SET PERFORMANCE")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Also create a small utility for single-file prediction
def create_prediction_script():
    script_content = """
import joblib
import pandas as pd
import numpy as np

def predict_network_traffic(csv_file_path):
    # Load logic
    model = joblib.load('models/random_forest_baseline.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    
    df = pd.read_csv(csv_file_path)
    # Essential features only
    REDUNDANT = ['Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Average Packet Size', 'Fwd IAT Mean', 'Attack Type']
    X = df.drop(columns=[col for col in REDUNDANT if col in df.columns])
    
    # Scale and Predict
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    labels = le.inverse_transform(preds)
    
    df['Predicted_Attack'] = labels
    return df

if __name__ == "__main__":
    print("This script is ready to use! Call predict_network_traffic(file_path)")
"""
    with open('/Users/nguyenducphat/Projects/TT12/predict_utility.py', 'w') as f:
        f.write(script_content)
    print("\nCreated 'predict_utility.py' for future use.")

if __name__ == "__main__":
    run_final_test()
    create_prediction_script()
