import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, mean_absolute_error



df = pd.read_csv("Students_Social_Media_Addiction_FE.csv")

print("start")
# Load dataset

X = df[["Sleep_Hours_Per_Night", "Avg_Daily_Usage_Hours", "Academic_Level_Encoded"]]
y = df["Affects_Academic_Performance_Encoded"]


# Model Type: Logistic Regression

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, C=0.1, class_weight={0: 4, 1: 1})


log_reg.fit(X_train_scaled,y_train)


y_pred = log_reg.predict(X_test_scaled)

print("Accuracy Score: ", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))



print("Precision score: ", precision_score(y_test,y_pred,average="weighted"))
print("Recall score: ", recall_score(y_test,y_pred))


coefficients = pd.DataFrame(
    {"Feature": X.columns,
    #retrieves the learned weights for each feature in the Logistic Regression model, showing both the direction and strength of each factor’s impact on academic performance
    "Coefficient": log_reg.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

print(coefficients)

probabilities = log_reg.predict_proba(X_test_scaled)[:,1]
avg_probability = np.median(probabilities)

print(f" Median Predicted Confidence Score: {avg_probability}")

mae = mean_absolute_error(y_test,y_pred)
print(f"Mean Absolute Error: {mae}")

joblib.dump(log_reg, "logistic_model_2.pkl")
joblib.dump(scaler, "scaler_2.pkl")
