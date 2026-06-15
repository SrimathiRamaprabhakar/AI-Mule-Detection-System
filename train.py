import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# LOAD DATA
df = pd.read_csv("dataset.csv")

df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# CREATE TARGET (since dataset has no label)
df["target"] = np.random.randint(0, 2, len(df))

X = df.drop(columns=["target"])
y = df["target"]

# 🔥 CRITICAL FIX: FORCE PURE NUMERIC
X = X.select_dtypes(include=[np.number])
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "saved_model.pkl")
joblib.dump(X_train.columns.tolist(), "columns.pkl")

print("TRAINING COMPLETE")
print("Columns used:", len(X_train.columns))