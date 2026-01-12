import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("heart_attack_prediction_india.csv")

print("Heart Disease Dataset Loaded Successfully")
print(data.head())
print("\nAvailable Columns:\n", list(data.columns))

# -----------------------------
# 2. Define REQUIRED Columns
# -----------------------------
required_features = [
    'Age',
    'Gender',
    'Diabetes',
    'Heart_Attack_History',
    'Emergency_Response_Time',
    'Annual_Income',
    'Health_Insurance'
]

target = 'Heart_Attack_Risk'

# -----------------------------
# 3. Check Column Existence
# -----------------------------
missing = [col for col in required_features if col not in data.columns]
if missing:
    raise ValueError(f"❌ Missing columns in dataset: {missing}")

X = data[required_features].copy()
y = data[target].copy()

# -----------------------------
# 4. Encode Categorical Values
# -----------------------------
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# -----------------------------
# 5. Handle Missing Values
# -----------------------------
X.fillna(X.mean(), inplace=True)
y.fillna(y.mode()[0], inplace=True)

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# 8. Evaluate Model
# -----------------------------
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Heart Disease Model Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 9. Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/heart_model.pkl")

print("✅ Heart disease model saved successfully!")
