# models/model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

from imblearn.over_sampling import SMOTE
from joblib import dump

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "processed_churn_data.csv"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
RANDOM_STATE = 42

# 1️⃣ Data Preparation
def prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Identify categorical features
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Apply SMOTE to balance dataset
    smote = SMOTE(sampling_strategy=0.3, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Identify numeric columns for scaling
    numeric_cols = X.columns  # after get_dummies, all features are numeric
    preprocessor = ColumnTransformer(
        transformers=[('scale', StandardScaler(), numeric_cols)]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_resampled
    )

    return X_train, X_test, y_train, y_test, preprocessor

# 2️⃣ Training and Evaluation
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, preprocessor):
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Get probability or decision score
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipeline.decision_function(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=1),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    # Save metrics
    pd.DataFrame(metrics, index=[model_name]).to_csv(
        ARTIFACTS_DIR / f"metrics_{model_name.lower()}.csv"
    )

    # Visualization
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_proba, model_name)

    return pipeline, metrics

# 3️⃣ Visualizations
def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(ARTIFACTS_DIR / f"conf_matrix_{name.lower()}.png", bbox_inches="tight")
    plt.close()

def plot_roc_curve(y_true, y_score, name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(ARTIFACTS_DIR / f"roc_curve_{name.lower()}.png", bbox_inches="tight")
    plt.close()

# 4️⃣ Main
if __name__ == "__main__":
    print("🚀 Training started...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(scale_pos_weight=1/(9/191), random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n🔥 Training {name}...")
        pipeline, metrics = train_and_evaluate(model, name, X_train, X_test, y_train, y_test, preprocessor)
        results[name] = metrics
        trained_models[name] = pipeline

    pd.DataFrame(results).T.to_csv(ARTIFACTS_DIR / "model_comparison.csv")

    # Save all models
    dump(trained_models["LogisticRegression"], ARTIFACTS_DIR / "best_logistic_model.joblib")
    dump(trained_models["XGBoost"], ARTIFACTS_DIR / "best_xgb_model.joblib")
    dump(trained_models["SVM"], ARTIFACTS_DIR / "best_svm_model.joblib")

    print("\n✅ All models trained and saved!")
