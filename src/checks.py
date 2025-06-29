import pandas as pd

df = pd.read_csv(r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\data\processed_churn_data.csv")
print(df["churn"].value_counts())  # Check class balance
