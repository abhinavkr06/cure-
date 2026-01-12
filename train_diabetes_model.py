import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("diabetes_prediction_india.csv")

print("Dataset Loaded Successfully")

# -----------------------------
# 2. Feature Selection
# -----------------------------
features = [
    'Age',
    'BMI',
    'Family_History',
    'Physical_Activity',
    'Fasting_Blood_Sugar',
    'Postprandial_Blood_Sugar',
    'HBA1C',
    'Hypertension'
]

target = 'Diabetes_Status'

X = data[features].copy()
y = data[target].copy()

# -----------------------------
# 3. Encode Categorical Values
# -----------------------------
X['Family_History'] = X['Family_History'].map({'Yes': 1, 'No': 0})
X['Hypertension'] = X['Hypertension'].map({'Yes': 1, 'No': 0})
X['Physical_Activity'] = X['Physical_Activity'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

y = y.map({'Yes': 1, 'No': 0})

# -----------------------------
# 4. Handle Missing Values
# -----------------------------
X.fillna(X.mean(), inplace=True)

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate Model
# -----------------------------
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Diabetes Model Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 8. Save Model
# -----------------------------
with open("models/diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Diabetes model saved successfully!")