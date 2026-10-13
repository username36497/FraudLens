import pandas as pd
df = pd.read_csv("data/extracted/creditcard.csv")

# Display the first 5 transactions
print(df.head())

print("Dataset shape:", df.shape)

print("\nTransaction counts:")
print(df["Class"].value_counts())
fraud_percentage = df["Class"].mean() * 100
print(f"\nFraud percentage: {fraud_percentage:.3f}%")

X = df.drop("Class", axis=1)
y = df["Class"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Second split: 60% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Create our machine learning model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train the model
model.fit(X_train_scaled, y_train)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Make predictions using the testing data
y_pred = model.predict(X_test_scaled)

# Evaluate our model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nFraudLens Model Results")
print("------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

y_prob = model.predict_proba(X_val_scaled)[:, 1]

# Try a higher threshold
threshold = 0.8

# Convert probabilities into predictions
y_pred_new = (y_prob >= threshold).astype(int)

print("\nResults with threshold =", threshold)

print("Precision:", precision_score(y_val, y_pred_new, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_new, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred_new, zero_division=0))

print("\nNew Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_new))

print("\nThreshold Comparison")
print("-" * 60)

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

best_threshold = 0
best_f1 = 0

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:

    predictions = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    score= f1_score(
        y_val, predictions, zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

    print("\nBest Threshold:", best_threshold)
    print("Best Validation F1-score:", round(best_f1, 3)) 

from sklearn.ensemble import RandomForestClassifier

# Create our Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train the Random Forest
print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

print("Random Forest trained successfully!")

rf_prob = rf_model.predict_proba(X_val)[:, 1]

# Compare thresholds
print("\nRandom Forest Results")
print("-" * 60)

best_rf_f1 = -1
best_rf_threshold = None

for threshold in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:

    predictions = (rf_prob >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

    if f1 > best_rf_f1:
        best_rf_f1 = f1
        best_rf_threshold = threshold

print("\nBest Random Forest threshold:", best_rf_threshold)
print("Best Random Forest F1:", round(best_rf_f1, 3))

print("\nModel Comparison")
print("--------------------------")
print("Logistic Regression F1:", round(best_f1, 3))
print("Random Forest F1:", round(best_rf_f1, 3))


import os
import joblib

# Create a folder for saved models
os.makedirs("models", exist_ok=True)

# Save the trained Random Forest
joblib.dump(rf_model, "models/random_forest.pkl")

# Save the selected validation threshold
joblib.dump(best_rf_threshold, "models/rf_threshold.pkl")

print("\nRandom Forest model saved successfully!")

from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
print("\nTraining XGBoost...")

# Train the model
xgb_model.fit(X_train, y_train)

print("XGBoost trained successfully!")

xgb_prob = xgb_model.predict_proba(X_val)[:, 1]

print("\nXGBoost Validation Results")
print("-" * 60)

for threshold in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:

    predictions = (xgb_prob >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

joblib.dump(xgb_model, "models/xgboost.pkl")

# Save the selected threshold
best_xgb_threshold = 0.20

joblib.dump(
    best_xgb_threshold,
    "models/xgboost_threshold.pkl"
)

print("\nXGBoost model saved successfully!")

from sklearn.metrics import classification_report, confusion_matrix

# Use our selected XGBoost model
test_probabilities = xgb_model.predict_proba(X_test)[:, 1]

# Apply our selected threshold
test_predictions = (test_probabilities >= 0.20).astype(int)

print("\nFINAL XGBOOST TEST RESULTS")
print("-" * 50)

print(classification_report(
    y_test,
    test_predictions,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))