import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# def create_models_directory():
#     """Create models directory if it doesn't exist"""
#     if not os.path.exists('models'):
#         os.makedirs('models')
#         print("✓ Created models directory")

# def train_and_save_models():
#     """Train basic models and save them"""
#     print("🔄 Training models...")
    
#     # Create dummy training data (replace with your actual training logic)
#     # This should match your feature set
#     np.random.seed(42)
#     n_samples = 1000
#     n_features = 20  # Adjust based on your actual features
    
#     # Generate synthetic training data
#     X_train = np.random.randn(n_samples, n_features)
#     y_train = np.random.randint(0, 2, n_samples)
    
#     # Train Random Forest
#     print("  Training Random Forest...")
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     rf_model.fit(X_train, y_train)
    
#     # Train Isolation Forest
#     print("  Training Isolation Forest...")
#     iso_model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
#     iso_model.fit(X_train)
    
#     # Train Scaler
#     print("  Training Scaler...")
#     scaler = StandardScaler()
#     scaler.fit(X_train)
    
#     # Save models
#     create_models_directory()
    
#     joblib.dump(rf_model, 'models/random_forest.pkl')
#     print("  ✓ Saved Random Forest model")
    
#     joblib.dump(iso_model, 'models/isolation_forest.pkl')
#     print("  ✓ Saved Isolation Forest model")
    
#     joblib.dump(scaler, 'models/scaler.pkl')
#     print("  ✓ Saved Scaler")
    
#     print("✅ All models trained and saved successfully!")

def check_and_setup_models():
    """Check if required model files exist"""
    model_files = [
        'models/random_forest.pkl',
        'models/xgboost.pkl',
        'models/neural_network.h5',
        'models/scaler.pkl',
        'models/metadata.json'
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if missing_models:
        print(f"⚠️  Missing models: {missing_models}")
        print("🚀 Initializing model training...")
    else:
        print("✅ All models found!")
    
    return True

if __name__ == "__main__":
    check_and_setup_models()