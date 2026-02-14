import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# -----------------------------
# Dummy Training Data
# -----------------------------
data = {
    "amount": [100, 5000, 30000, 200, 45000, 1500, 60000],
    "is_international": [0, 0, 1, 0, 1, 0, 1],
    "is_online": [0, 1, 1, 0, 1, 1, 1],
    "fraud": [0, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["amount", "is_international", "is_online"]]
y = df["fraud"]

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fraud_model.pkl")

print("✅ Model trained and saved successfully")
