"""
Step 2: Preprocessing & Baseline Model Training
Dataset: CICIDS2017 (cleaned)
Target: Attack Type (7 classes)
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── PATHS ─────────────────────────────────────────────────────────────────────
DATASET_PATH = '/Users/nguyenducphat/Projects/TT12/cicids2017_cleaned.csv'
OUT_DIR      = '/Users/nguyenducphat/Projects/TT12/visualizations'
MODEL_DIR    = '/Users/nguyenducphat/Projects/TT12/models'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. LOAD DATA ───────────────────────────────────────────────────────────────
print("="*60)
print("1. LOADING DATA")
print("="*60)
df = pd.read_csv(DATASET_PATH)
print(f"   Loaded {len(df):,} rows x {df.shape[1]} cols")

# ─── 2. PREPROCESSING ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. PREPROCESSING")
print("="*60)

# 2a. Drop highly correlated redundant features (from correlation analysis)
# - Keep 'Fwd Packet Length Max' → drop 'Fwd Packet Length Mean' (corr=0.89)
# - Keep 'Bwd Packet Length Max' → drop 'Bwd Packet Length Mean' (corr=0.96)
# - Keep 'Packet Length Mean'    → drop 'Average Packet Size'   (corr=1.00)
# - Keep 'Flow IAT Mean'         → drop 'Fwd IAT Mean' (0.90), 'Bwd IAT Mean' (0.63→acceptable boundary)
REDUNDANT_COLS = [
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Average Packet Size',
    'Fwd IAT Mean',
]
df.drop(columns=REDUNDANT_COLS, inplace=True)
print(f"   Dropped redundant columns: {REDUNDANT_COLS}")
print(f"   Remaining feature count: {df.shape[1] - 1}")  # -1 for label

# 2b. Encode label
TARGET = 'Attack Type'
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[TARGET])
print(f"\n   Label classes: {list(le.classes_)}")
print(f"   Encoded as   : {list(range(len(le.classes_)))}")

# 2d. Split features / target
X = df.drop(columns=[TARGET, 'label_encoded'])
y = df['label_encoded']

# 2e. Train / Validation / Test split  (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\n   Train : {len(X_train):,} samples")
print(f"   Val   : {len(X_val):,}  samples")
print(f"   Test  : {len(X_test):,}  samples")

# 2f. Standard-scale features (important for some classifiers; RF is robust but still good practice)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
print("\n   Features scaled with StandardScaler (fit on train only).")

# 2g. Apply SMOTE to balance minority classes (especially Bots)
print("\n" + "="*60)
print("2g. APPLYING SMOTE (Synthetic Minority Over-sampling)")
print("="*60)

# Check class distribution before SMOTE
train_dist_before = pd.Series(y_train).value_counts().sort_index()
print("\n   Class distribution BEFORE SMOTE:")
for class_idx, count in train_dist_before.items():
    print(f"     {le.classes_[class_idx]:<18} : {count:>8,} samples")

# Apply SMOTE - oversample minority classes moderately
# Strategy: tăng Bots và Web Attacks lên 3x (không quá nhiều để tránh overfitting)
sampling_strategy = {
    0: 4000,  # Bots: 1364 → 4000
    6: 4500,  # Web Attacks: 1500 → 4500
}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
X_train_s, y_train = smote.fit_resample(X_train_s, y_train)

train_dist_after = pd.Series(y_train).value_counts().sort_index()
print("\n   Class distribution AFTER SMOTE:")
for class_idx, count in train_dist_after.items():
    print(f"     {le.classes_[class_idx]:<18} : {count:>8,} samples")

print(f"\n   Total training samples: {len(y_train):,}")
print("   → SMOTE tạo synthetic samples cho Bots và Web Attacks")
print("   → Giúp model học tốt hơn pattern của các class hiếm")

# 2h. Save scaler & encoder for later use
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(le,     os.path.join(MODEL_DIR, 'label_encoder.pkl'))
print("   Saved scaler.pkl and label_encoder.pkl")

# ─── 3. BASELINE MODEL: Random Forest ──────────────────────────────────────────
print("\n" + "="*60)
print("3. TRAINING BASELINE MODEL (Random Forest + class_weight='balanced')")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    min_samples_split=5,
    class_weight='balanced',   # handles imbalance
    n_jobs=-1,
    random_state=42,
)

print("   Fitting model …")
rf.fit(X_train_s, y_train)
print("   Done!")

# ─── 4. EVALUATION ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. EVALUATION ON VALIDATION SET")
print("="*60)

y_pred = rf.predict(X_val_s)

acc  = accuracy_score(y_val, y_pred)
f1m  = f1_score(y_val, y_pred, average='macro')
f1w  = f1_score(y_val, y_pred, average='weighted')

print(f"\n   Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
print(f"   F1 (macro)      : {f1m:.4f}")
print(f"   F1 (weighted)   : {f1w:.4f}")
print("\n   Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# ─── 5. CONFUSION MATRIX PLOT ──────────────────────────────────────────────────
cm = confusion_matrix(y_val, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize by row

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Normalized Confusion Matrix – Random Forest (Validation Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, 'confusion_matrix_rf.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"\n   Saved confusion matrix → {cm_path}")

# ─── 6. FEATURE IMPORTANCE PLOT ────────────────────────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top20    = feat_imp.nlargest(20).sort_values()

plt.figure(figsize=(10, 8))
top20.plot(kind='barh', color='steelblue')
plt.title('Top 20 Feature Importances – Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
fi_path = os.path.join(OUT_DIR, 'feature_importance_rf.png')
plt.savefig(fi_path, dpi=300)
plt.close()
print(f"   Saved feature importance → {fi_path}")

# ─── 7. SAVE MODEL ─────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, 'random_forest_baseline.pkl')
joblib.dump(rf, model_path)
print(f"\n   Model saved → {model_path}")

print("\n" + "="*60)
print("✅ DONE – Baseline pipeline completed successfully!")
print("="*60)
