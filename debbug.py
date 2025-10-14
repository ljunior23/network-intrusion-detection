import joblib
import json

print("="*80)
print("CHECKING MODEL FEATURES")
print("="*80)

# Load feature columns
try:
    feature_cols = joblib.load('models/feature_columns.pkl')
    print(f"\n✓ Loaded {len(feature_cols)} features from feature_columns.pkl")
    print("\nFeature list:")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")
    
    # Save to text file for easy reference
    with open('models/feature_list.txt', 'w') as f:
        f.write("Required Features for Model Prediction\n")
        f.write("="*50 + "\n\n")
        for i, col in enumerate(feature_cols, 1):
            f.write(f"{i:2d}. {col}\n")
    
    print("\n✓ Saved feature list to models/feature_list.txt")
    
except FileNotFoundError:
    print("\n❌ feature_columns.pkl not found!")
    print("Make sure you've run the training script first.")

# Load label encoders
try:
    encoders = joblib.load('models/label_encoders.pkl')
    print(f"\n✓ Loaded {len(encoders)} label encoders")
    print("\nCategorical features:")
    for col in encoders.keys():
        print(f"  - {col}: {len(encoders[col].classes_)} classes")
        print(f"    Classes: {list(encoders[col].classes_)}")
    
except FileNotFoundError:
    print("\n❌ label_encoders.pkl not found!")

# Load scaler
try:
    scaler = joblib.load('models/scaler.pkl')
    print(f"\n✓ Loaded scaler")
    print(f"  - Expects {scaler.n_features_in_} features")
    print(f"  - Feature names in scaler: {scaler.feature_names_in_[:10]}... (showing first 10)")
    
except FileNotFoundError:
    print("\n❌ scaler.pkl not found!")

print("\n" + "="*80)
print("DONE! Check models/feature_list.txt for the complete list")
print("="*80)