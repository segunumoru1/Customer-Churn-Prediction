# models/model.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load processed data
def load_data():
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed_churn_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}. Please run preprocessing.py first.")
    
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded: {df.shape}")
    return df

# Prepare features for modeling
def prepare_features(df):
    # Create a copy to avoid modifying original data
    df_model = df.copy()
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_model, columns=['has_contract', 'payment_method'], prefix=['has_contract', 'payment_method'])
    
    # Separate features and target
    X = df_encoded.drop('churn', axis=1)
    y = df_encoded['churn']
    
    print(f"✅ Features prepared: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y, list(X.columns)

# Train models
def train_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'y_test': y_test
        }
        
        print(f"✅ {name} Accuracy: {accuracy:.4f}")
    
    return results, X_test, y_test

# Save models and metadata
def save_models(results, feature_columns):
    # Create artifacts directory
    script_dir = Path(__file__).parent
    artifacts_dir = script_dir.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    # Save best model
    model_path = artifacts_dir / "best_xgb_model.joblib"
    joblib.dump(best_model, model_path)
    
    # Save feature columns for later use
    feature_path = artifacts_dir / "feature_columns.joblib"
    joblib.dump(feature_columns, feature_path)
    
    print(f"\n💾 Best model ({best_model_name}) saved to: {model_path}")
    print(f"💾 Feature columns saved to: {feature_path}")
    
    return best_model_name, best_model

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Model Training Pipeline...")
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_columns = prepare_features(df)
    
    # Train models
    results, X_test, y_test = train_models(X, y)
    
    # Save best model
    best_model_name, best_model = save_models(results, feature_columns)
    
    # Print final results
    print(f"\n📊 Training Complete!")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"🎯 Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Print classification report for best model
    y_pred_best = results[best_model_name]['predictions']
    print(f"\n📈 Classification Report for {best_model_name}:")
    print(classification_report(y_test, y_pred_best))
