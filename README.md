
# 📊 Customer Churn Prediction with MLflow, Streamlit & Docker

This project predicts customer churn using machine learning and provides a full end-to-end pipeline from data preprocessing to model evaluation, versioning, and interactive deployment via Streamlit. The system leverages MLflow for experiment tracking and is fully containerized using Docker.

---

## 🚀 Features

- Raw data ingestion and preprocessing pipeline  
- Multiple trained models (Logistic Regression, SVM, XGBoost)  
- Hyperparameter tuning with performance comparison  
- MLflow tracking and model registry  
- Streamlit UI for real-time predictions  
- Docker & Docker Compose integration for easy deployment  

---

## 🧾 Dataset Overview

The dataset contains customer attributes including:

- **Tenure** (months)
- **Monthly charges**
- **Support calls**
- **Contract type**
- **Payment method**
- ...and other key factors that influence churn.

---

## 📂 Project Structure

```
├── artifacts/               # Saved trained models
├── mlruns/                 # MLflow experiment logs
├── src/                    # Streamlit and pipeline code
│   ├── app.py              # Streamlit UI
│   ├── train.py            # Model training + MLflow logging
│   └── preprocess.py       # Data cleaning and transformations
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔄 Workflow Overview

### 1. **Data Upload & Cleaning**
Raw CSV data is loaded and cleaned by handling missing values, converting categorical variables, and standardizing numerical features.

### 2. **Preprocessing**
`sklearn` pipelines apply scaling and one-hot encoding. The pipeline ensures consistency across training and inference stages.

### 3. **Model Training & Evaluation**
Three classifiers were trained and evaluated:

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| LogisticRegression| 0.92     | 0.73      | 1.00   | 0.85     |
| SVM               | 0.96     | 0.85      | 1.00   | 0.92     |
| **XGBoost**        | **0.98** | **0.92**  | **1.00** | **0.96** |

> **XGBoost** was selected as the best-performing model based on accuracy and F1-score.

### 4. **Hyperparameter Tuning**
GridSearchCV and random sampling were used to optimize hyperparameters for all models. MLflow tracked each run for comparative analysis.

### 5. **Model Tracking with MLflow**
All models, metrics, and parameters are tracked via MLflow:

```bash
mlflow ui --backend-store-uri "file:///app/mlruns"
```

Visit the MLflow UI at [http://localhost:5000](http://localhost:5000)

---

## 🎯 Streamlit App

The app allows users to input customer details and receive churn predictions in real time.

To launch locally:

```bash
streamlit run src/app.py
```

---

## 🐳 Docker Usage

### 📦 Build the image

```bash
docker build -t churn-app .
```

### ▶️ Run with Docker Compose

```bash
docker-compose up --build
```

- Streamlit: [http://localhost:8501](http://localhost:8501)
- MLflow UI: [http://localhost:5000](http://localhost:5000)

---

## 🔮 Prediction Sample

Input customer data via the web UI:

- Has contract: Yes  
- Payment method: Credit Card  
- Tenure: 12 months  
- Monthly charges: 80  
- Support calls: 2  

> ✅ **Prediction**: Customer likely to stay  
> 📈 **Probability**: 3%

---

## 🛠️ Future Improvements

- Add support for batch predictions via CSV upload  
- Automate model promotion to MLflow registry  
- Enable cloud deployment via Hugging Face or Render  
- Add FastAPI endpoint for REST predictions  

---

## 👩🏽‍💻 Author

**Ifeoma Adigwe**  
Built with ❤️ using Python, Scikit-learn, Streamlit, and MLflow

---

