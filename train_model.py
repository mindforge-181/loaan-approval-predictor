"""
train_model.py
Project: Loan Approval Predictor

Purpose:
Loads loans.csv, preprocesses the data, trains a Random Forest
classifier, evaluates it, and saves model.pkl and scaler.pkl.
"""

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

DATA_PATH = "loans.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "loans.csv not found. Make sure it is in the same "
        "folder as train_model.py."
    )

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded successfully: "
    f"{df.shape[0]} rows, {df.shape[1]} columns"
)


# ---------------------------------------------------------
# 2. Define features and target
# ---------------------------------------------------------

FEATURES = [
    "income",
    "credit_score",
    "loan_amount",
    "employment_years"
]

TARGET = "loan_status"


# ---------------------------------------------------------
# 3. Check required columns
# ---------------------------------------------------------

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in loans.csv: {missing_columns}"
    )


# ---------------------------------------------------------
# 4. Create X and y
# ---------------------------------------------------------

X = df[FEATURES]
y = df[TARGET]


# ---------------------------------------------------------
# 5. Check missing values
# ---------------------------------------------------------

if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:

    print("Missing values detected.")

    data = pd.concat([X, y], axis=1).dropna()

    X = data[FEATURES]
    y = data[TARGET]


# ---------------------------------------------------------
# 6. Train / Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 7. Scale the features
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 8. Train Random Forest model
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# ---------------------------------------------------------
# 9. Evaluate model
# ---------------------------------------------------------

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ---------------------------------------------------------
# 10. Feature Importance
# ---------------------------------------------------------

print("\nFeature Importances:")

feature_importance = sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: -x[1]
)

for feature, importance in feature_importance:
    print(f"  {feature}: {importance:.4f}")


# ---------------------------------------------------------
# 11. Save model and scaler
# ---------------------------------------------------------

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


print("\n" + "=" * 50)
print("SUCCESS")
print("=" * 50)

print("model.pkl created successfully.")
print("scaler.pkl created successfully.")