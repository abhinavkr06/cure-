import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("kidney_disease_dataset.csv")

print("\nKidney Dataset Loaded Successfully")

# -----------------------------
# 2. Clean Column Names
# -----------------------------
data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("/", "_")
    .str.replace("-", "_")
)

print("\nCleaned Columns:\n", list(data.columns))

# -----------------------------
# 3. Replace '?' with NaN
# -----------------------------
data.replace("?", pd.NA, inplace=True)

# -----------------------------
# 4. Feature Selection (MATCHING YOUR DATASET)
# -----------------------------
features = [
    'age_of_the_patient',
    'blood_pressure_mm_hg',
    'specific_gravity_of_urine',
    'random_blood_glucose_level_mg_dl',
    'serum_creatinine_mg_dl',
    'hemoglobin_level_gms',
    'estimated_glomerular_filtration_rate_egfr'
]

features = [f for f in features if f in data.columns]

print("\nUsing Features:", features)

X = data[features].copy()
y = data['target'].copy()

# -----------------------------
# 5. Encode Target (FIXED)
# -----------------------------
y = y.astype(str).str.strip().map({
    'No_Disease': 0,
    'Low_Risk': 1,
    'High_Risk': 1
})

# 🔒 Drop rows where target is still NaN
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# -----------------------------
# 6. Convert Features to Numeric
# -----------------------------
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# -----------------------------
# 7. Handle Missing Values
# -----------------------------
X.fillna(X.median(), inplace=True)

# -----------------------------
# 8. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# 9. Train Optimized Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# 10. Evaluate
# -----------------------------
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"\n✅ Kidney Disease Model Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 11. Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/kidney_model.pkl")

print("\n✅ Kidney model saved successfully!")
