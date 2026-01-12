import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv(
    "Liver Patient Dataset (LPD)_train.csv",
    encoding="latin1"
)

print("\nLiver Dataset Loaded Successfully")
print("\nOriginal Columns:\n", data.columns)

# Clean column names
data.columns = data.columns.str.strip().str.lower()

print("\nCleaned Columns:\n", data.columns)

# -----------------------------
# 2. Feature Selection (EXACT MATCH)
# -----------------------------
features = [
    'age of the patient',
    'gender of the patient',
    'total bilirubin',
    'direct bilirubin',
    'alkphos alkaline phosphotase',
    'sgpt alamine aminotransferase',
    'sgot aspartate aminotransferase',
    'total protiens',
    'alb albumin',
    'a/g ratio albumin and globulin ratio'
]

X = data[features].copy()
y = data['result'].copy()

# -----------------------------
# 3. Encode Gender & Target
# -----------------------------
X['gender of the patient'] = (
    X['gender of the patient']
    .astype(str)
    .str.lower()
    .map({'male': 1, 'female': 0})
)

# Result: 1 = Liver Disease, 2 = No Disease
y = y.map({1: 1, 2: 0})

# -----------------------------
# 4. Handle Missing Values
# -----------------------------
X.fillna(X.median(), inplace=True)

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# 6. Train Optimized Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Liver Disease Model Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 8. Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/liver_model.pkl")

print("\n✅ Liver model saved successfully!")
