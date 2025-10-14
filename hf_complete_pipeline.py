import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, roc_auc_score, roc_curve,
                            precision_recall_fscore_support)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

print("="*80)
print("NETWORK INTRUSION DETECTION - COMPLETE ML PIPELINE")
print("Using Hugging Face Datasets 🤗")
print("="*80)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ GPU Available: {gpus[0].name}")
    print("  TensorFlow will use GPU acceleration!")
else:
    print("\n⚠ No GPU detected. Training will use CPU.")

# ============================================================================
# STEP 1: LOAD DATA FROM HUGGING FACE
# ============================================================================

print("\n[1/10] Loading NSL-KDD from Hugging Face...")

try:
    dataset = load_dataset("Mireu-Lab/NSL-KDD")
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    
    print(f"✓ Training samples: {len(df_train):,}")
    print(f"✓ Test samples: {len(df_test):,}")
    print(f"✓ Features: {df_train.shape[1]}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure 'datasets' library is installed: pip install datasets")
    exit(1)

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

print("\n[2/10] Preprocessing data...")

# Identify target column
target_cols = ['label', 'attack_type', 'class', 'Label']
target_col = None
for col in target_cols:
    if col in df_train.columns:
        target_col = col
        break

if target_col is None:
    print("Error: Could not find target column")
    exit(1)

print(f"Target column: {target_col}")

# Create binary labels and attack categories
def create_labels(df, target_col):
    """Create binary labels and attack categories"""
    df = df.copy()
    
    # Binary: Normal (0) vs Attack (1)
    df['is_attack'] = df[target_col].apply(
        lambda x: 0 if 'normal' in str(x).lower() else 1
    )
    
    # Multi-class categories
    attack_map = {}
    for val in df[target_col].unique():
        val_lower = str(val).lower()
        if 'normal' in val_lower:
            attack_map[val] = 'Normal'
        elif any(x in val_lower for x in ['dos', 'neptune', 'smurf', 'pod', 'teardrop', 'land', 'back']):
            attack_map[val] = 'DoS'
        elif any(x in val_lower for x in ['probe', 'portsweep', 'ipsweep', 'nmap', 'satan']):
            attack_map[val] = 'Probe'
        elif any(x in val_lower for x in ['r2l', 'guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy']):
            attack_map[val] = 'R2L'
        elif any(x in val_lower for x in ['u2r', 'buffer_overflow', 'loadmodule', 'perl', 'rootkit']):
            attack_map[val] = 'U2R'
        else:
            attack_map[val] = 'Other'
    
    df['attack_category'] = df[target_col].map(attack_map)
    return df

df_train = create_labels(df_train, target_col)
df_test = create_labels(df_test, target_col)

print("\nClass distribution (Training):")
print(df_train['attack_category'].value_counts())

# Save processed data
df_train.to_csv('data/processed/train_labeled.csv', index=False)
df_test.to_csv('data/processed/test_labeled.csv', index=False)

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

print("\n[3/10] Feature engineering...")

# Select features
exclude_cols = [target_col, 'is_attack', 'attack_category']
if 'difficulty' in df_train.columns:
    exclude_cols.append('difficulty')

feature_cols = [col for col in df_train.columns if col not in exclude_cols]

X_train = df_train[feature_cols].copy()
y_train = df_train['is_attack'].copy()
X_test = df_test[feature_cols].copy()
y_test = df_test['is_attack'].copy()

# Encode categorical features
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical features: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle all categories
    all_values = pd.concat([X_train[col], X_test[col]]).astype(str)
    le.fit(all_values)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Handle infinite and missing values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

for col in X_train.columns:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

print(f"Final features: {X_train.shape[1]}")

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================

print("\n[4/10] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
print("✓ Scaler and encoders saved")

# ============================================================================
# STEP 5: HANDLE CLASS IMBALANCE
# ============================================================================

print("\n[5/10] Balancing classes with SMOTE...")

print(f"Original: Normal={sum(y_train==0):,}, Attack={sum(y_train==1):,}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Balanced: Normal={sum(y_train_balanced==0):,}, Attack={sum(y_train_balanced==1):,}")

# ============================================================================
# STEP 6: TRAIN RANDOM FOREST
# ============================================================================

print("\n[6/10] Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f"✓ Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}")

joblib.dump(rf_model, 'models/random_forest.pkl')

# ============================================================================
# STEP 7: TRAIN XGBOOST
# ============================================================================

print("\n[7/10] Training XGBoost...")

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    tree_method='gpu_hist' if gpus else 'hist',
    eval_metric='logloss',
    verbosity=0
)

xgb_model.fit(X_train_balanced, y_train_balanced)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)

print(f"✓ Accuracy: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}")

joblib.dump(xgb_model, 'models/xgboost.pkl')

# ============================================================================
# STEP 8: TRAIN NEURAL NETWORK
# ============================================================================

print("\n[8/10] Training Deep Neural Network...")

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

history = model.fit(
    X_train_balanced, y_train_balanced,
    validation_split=0.2,
    epochs=20,
    batch_size=256,
    verbose=1,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5)
    ]
)

nn_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
nn_pred = (nn_pred_proba > 0.5).astype(int)
nn_acc = accuracy_score(y_test, nn_pred)
nn_auc = roc_auc_score(y_test, nn_pred_proba)

print(f"✓ Accuracy: {nn_acc:.4f}, AUC: {nn_auc:.4f}")

model.save('models/neural_network.h5')

# ============================================================================
# STEP 9: MODEL COMPARISON
# ============================================================================

print("\n[9/10] Generating comparison reports...")

results = {
    'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
    'Accuracy': [rf_acc, xgb_acc, nn_acc],
    'AUC': [rf_auc, xgb_auc, nn_auc]
}

results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)
print(results_df.to_string(index=False))
print("="*60)

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

results_df.plot(x='Model', y='Accuracy', kind='bar', ax=axes[0], color='steelblue', legend=False)
axes[0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0.9, 1.0])
axes[0].tick_params(axis='x', rotation=45)

results_df.plot(x='Model', y='AUC', kind='bar', ax=axes[1], color='coral', legend=False)
axes[1].set_title('Model AUC Comparison', fontweight='bold', fontsize=14)
axes[1].set_ylabel('AUC Score')
axes[1].set_ylim([0.9, 1.0])
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/figures/model_comparison.png")

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, pred) in enumerate([('Random Forest', rf_pred), ('XGBoost', xgb_pred), ('Neural Network', nn_pred)]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/figures/confusion_matrices.png")

# ============================================================================
# STEP 10: SAVE METADATA
# ============================================================================

print("\n[10/10] Saving metadata...")

# Get classification reports
rf_report = classification_report(y_test, rf_pred, output_dict=True)
xgb_report = classification_report(y_test, xgb_pred, output_dict=True)
nn_report = classification_report(y_test, nn_pred, output_dict=True)

metadata = {
    'dataset_source': 'Hugging Face: Mireu-Lab/NSL-KDD',
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'features': int(X_train.shape[1]),
    'models': {
        'random_forest': {
            'accuracy': float(rf_acc),
            'auc': float(rf_auc),
            'precision': float(rf_report['1']['precision']),
            'recall': float(rf_report['1']['recall']),
            'f1_score': float(rf_report['1']['f1-score'])
        },
        'xgboost': {
            'accuracy': float(xgb_acc),
            'auc': float(xgb_auc),
            'precision': float(xgb_report['1']['precision']),
            'recall': float(xgb_report['1']['recall']),
            'f1_score': float(xgb_report['1']['f1-score'])
        },
        'neural_network': {
            'accuracy': float(nn_acc),
            'auc': float(nn_auc),
            'precision': float(nn_report['1']['precision']),
            'recall': float(nn_report['1']['recall']),
            'f1_score': float(nn_report['1']['f1-score'])
        }
    },
    'best_model': results_df.loc[results_df['Accuracy'].idxmax(), 'Model'],
    'timestamp': pd.Timestamp.now().isoformat()
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✓ Metadata saved")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nBest Model: {metadata['best_model']}")
print(f"Best Accuracy: {results_df['Accuracy'].max():.4f}")
print(f"Best AUC: {results_df['AUC'].max():.4f}")
print("\n📁 All models saved in 'models/' directory")
print("🎨 Visualizations saved in 'outputs/figures/'")
print("🚀 Ready for dashboard deployment!")
print("="*80)