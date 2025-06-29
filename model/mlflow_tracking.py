import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import plotly.express as px
import plotly.io as pio

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
DATA_PATH = r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\data\processed_churn_data.csv"

# Create directories
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# MLflow setup
mlflow.set_tracking_uri("file:///" + MLRUNS_DIR.replace("\\", "/"))
mlflow.set_experiment("customer_churn_prediction_comparison")

pio.renderers.default = "svg"

def log_confusion_matrix(y_true, y_pred, model_name):
    fig = px.imshow(
        confusion_matrix(y_true, y_pred),
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual"},
        title=f"Confusion Matrix: {model_name}"
    )
    fig_path = os.path.join(ARTIFACTS_DIR, f"conf_matrix_{model_name}.png")
    fig.write_image(fig_path)
    mlflow.log_artifact(fig_path)

def evaluate_and_log(model, X_test, y_test, run_name, model_type="sklearn"):
    print(f"\n🔍 Starting MLflow run: {run_name}")
    print(f"🔍 Logging run to: {mlflow.get_tracking_uri()}")
    with mlflow.start_run(run_name=run_name):
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            from scipy.special import softmax
            y_proba = softmax(model.decision_function(X_test), axis=1)
        else:
            y_proba = np.zeros((len(y_test), len(np.unique(y_test))))

        try:
            roc_auc = (
                roc_auc_score(y_test, y_proba[:, 1])
                if y_proba.shape[1] == 2
                else roc_auc_score(y_test, y_proba, multi_class="ovr")
            )
        except Exception as e:
            print(f"⚠️ ROC-AUC error: {e}")
            roc_auc = 0.0

        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path=run_name.lower())
        else:
            mlflow.sklearn.log_model(model, artifact_path=run_name.lower())

        try:
            mlflow.log_params(model.get_params())
        except Exception as e:
            print(f"⚠️ Param logging failed: {e}")

        mlflow.log_metrics({
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "ROC-AUC": roc_auc
        })

        log_confusion_matrix(y_test, y_pred, run_name)
        print(f"✅ Completed run: {run_name}\n")

if __name__ == "__main__":
    print("🚀 Starting churn prediction MLflow pipeline...")

    try:
        df = pd.read_csv(DATA_PATH)
        print(f"📦 Loaded data from {DATA_PATH}")
    except Exception as e:
        raise FileNotFoundError(f"❌ Could not load CSV file: {e}")

    if 'churn' not in df.columns:
        raise ValueError("❌ Column 'churn' not found in dataset.")

    # Preprocessing: one-hot encode categoricals
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    evaluate_and_log(lr, X_test, y_test, "Logistic_Regression")

    # SVM
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    evaluate_and_log(svm, X_test, y_test, "SVM")

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    evaluate_and_log(xgb, X_test, y_test, "XGBoost", model_type="xgboost")

    joblib.dump(xgb, os.path.join(MODELS_DIR, "best_model.pkl"))
    print(f"💾 Saved best model to {os.path.join(MODELS_DIR, 'best_model.pkl')}")

    print(f"\n✨ All runs complete. View them at:\nmlflow ui --backend-store-uri \"file:///{MLRUNS_DIR.replace('\\', '/')}\"")
