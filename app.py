from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ---------------- LOAD MODELS ----------------
diabetes_model = joblib.load("models/diabetes_model.pkl")
heart_model = joblib.load("models/heart_model.pkl")
liver_model = joblib.load("models/liver_model.pkl")
kidney_model = joblib.load("models/kidney_model.pkl")

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html", viewers=1)

@app.route("/faq")
def faq():
    return render_template("faq.html")

# ---------------- DIABETES ----------------
@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    result = None
    if request.method == "POST":
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodpressure"]),
            float(request.form["skinthickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]
        pred = diabetes_model.predict([data])[0]
        result = "Diabetes Detected ⚠️" if pred == 1 else "No Diabetes ✅"

    return render_template("diabetes.html", result=result)

# ---------------- HEART ----------------
@app.route("/heart", methods=["GET", "POST"])
def heart():
    result = None
    if request.method == "POST":
        data = [
            float(request.form["age"]),
            float(request.form["gender"]),
            float(request.form["cholesterol"]),
            float(request.form["blood_pressure"]),
            float(request.form["heart_rate"]),
            float(request.form["smoking"]),
            float(request.form["diabetes"])
        ]

        pred = heart_model.predict([data])[0]
        result = "Heart Disease Risk ⚠️" if pred == 1 else "Healthy Heart ✅"

    return render_template("heart.html", result=result)


# ---------------- LIVER ----------------
@app.route("/liver", methods=["GET", "POST"])
def liver():
    result = None
    if request.method == "POST":
        data = [
            float(request.form["age"]),
            float(request.form["gender"]),
            float(request.form["total_bilirubin"]),
            float(request.form["direct_bilirubin"]),
            float(request.form["alk_phos"]),
            float(request.form["sgpt"]),
            float(request.form["sgot"]),
            float(request.form["total_proteins"]),
            float(request.form["albumin"]),
            float(request.form["ag_ratio"])
        ]
        pred = liver_model.predict([data])[0]
        result = "Liver Disease Detected ⚠️" if pred == 1 else "Healthy Liver ✅"

    return render_template("liver.html", result=result)

# ---------------- KIDNEY ----------------
@app.route("/kidney", methods=["GET", "POST"])
def kidney():
    result = None
    if request.method == "POST":
        data = [
            float(request.form["age"]),
            float(request.form["blood_pressure"]),
            float(request.form["specific_gravity"]),
            float(request.form["serum_creatinine"]),
            float(request.form["hemoglobin"]),
            float(request.form["crp"]),
            float(request.form["il6"])
        ]
        pred = kidney_model.predict([data])[0]
        result = "Kidney Disease Risk ⚠️" if pred == 1 else "You have healthy Kidney ✅"

    return render_template("kidney.html", result=result)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
