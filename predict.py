import os
import joblib
import pandas as pd

# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_scoring_optimized.pkl"
)

package = joblib.load(MODEL_PATH)

model = package["model"]
imputer = package["imputer"]
threshold = package["threshold"]
features = package["features"]


# ============================================================
# CUSTOMER INPUT
# ============================================================

print("\n" + "=" * 60)
print("       CREDIT RISK PREDICTION SYSTEM")
print("=" * 60)

print("\nEnter customer financial information:\n")

data = {}

data["RevolvingUtilizationOfUnsecuredLines"] = float(
    input("Revolving Utilization (e.g. 0.50): ")
)

data["age"] = float(
    input("Age: ")
)

data["NumberOfTime30-59DaysPastDueNotWorse"] = float(
    input("Times 30-59 Days Past Due: ")
)

data["DebtRatio"] = float(
    input("Debt Ratio (e.g. 0.30): ")
)

data["MonthlyIncome"] = float(
    input("Monthly Income: ")
)

data["NumberOfOpenCreditLinesAndLoans"] = float(
    input("Open Credit Lines & Loans: ")
)

data["NumberOfTimes90DaysLate"] = float(
    input("Times 90 Days Late: ")
)

data["NumberRealEstateLoansOrLines"] = float(
    input("Real Estate Loans/Lines: ")
)

data["NumberOfTime60-89DaysPastDueNotWorse"] = float(
    input("Times 60-89 Days Past Due: ")
)

data["NumberOfDependents"] = float(
    input("Number of Dependents: ")
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

input_df = pd.DataFrame([data])

# Ensure exact feature order
input_df = input_df[features]


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

input_processed = pd.DataFrame(
    imputer.transform(input_df),
    columns=features
)


# ============================================================
# PREDICTION
# ============================================================

probability = model.predict_proba(
    input_processed
)[0][1]

prediction = int(
    probability >= threshold
)


# ============================================================
# RISK LEVEL
# ============================================================

if probability < 0.30:
    risk = "LOW RISK"

elif probability < 0.60:
    risk = "MEDIUM RISK"

else:
    risk = "HIGH RISK"


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("             CREDIT RISK RESULT")
print("=" * 60)

print(
    f"\nRisk Probability : {probability * 100:.2f}%"
)

print(
    f"Decision Threshold: {threshold:.2f}"
)

print(
    f"Risk Level        : {risk}"
)

if risk == "HIGH RISK":
    print("\n⚠️ Prediction: HIGH CREDIT RISK")

elif risk == "MEDIUM RISK":
    print("\n⚠️ Prediction: MODERATE CREDIT RISK")

else:
    print("\n✅ Prediction: LOW CREDIT RISK")

print("\n" + "=" * 60)