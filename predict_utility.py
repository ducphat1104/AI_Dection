
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
