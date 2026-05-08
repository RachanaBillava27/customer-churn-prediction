# ============================================================
#  save_model.py
#  Run this ONCE after training to save your model & scaler
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── 1. Load & clean data ────────────────────────────────────
df = pd.read_csv("data.csv")

if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.fillna(df.median(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# ── 2. Encode target ────────────────────────────────────────
if df["Churn"].dtype == object:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ── 3. One-Hot Encode features ──────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ── 4. Save column names (needed for prediction alignment) ──
feature_cols = [c for c in df_encoded.columns if c != "Churn"]
joblib.dump(feature_cols, "feature_columns.pkl")
print(f"Saved {len(feature_cols)} feature columns.")

# ── 5. Split & scale ────────────────────────────────────────
X = df_encoded[feature_cols]
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ── 6. Train Random Forest ──────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 7. Save artifacts ───────────────────────────────────────
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅  model.pkl   saved")
print("✅  scaler.pkl  saved")
print("✅  feature_columns.pkl saved")
print("\nAll done! Now run:  streamlit run app.py")
