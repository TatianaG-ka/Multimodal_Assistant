import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

MODELS_DIR = Path("data/models")
RF_PATH = MODELS_DIR / "random_forest_model.pkl"
ENS_PATH = MODELS_DIR / "ensemble_model.pkl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---- RANDOM FOREST MODEL ----
if not RF_PATH.exists():
    X = np.random.rand(300, 20)
    y = X.sum(axis=1) * 10 + np.random.rand(300)
    rf = RandomForestRegressor(n_estimators=30, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, RF_PATH)

# ---- ENSEMBLE MODEL ----
if not ENS_PATH.exists():
    Z = np.random.rand(300, 5) * 100
    t = Z[:, :3].mean(axis=1) + np.random.rand(300)
    lr = LinearRegression()
    lr.fit(Z, t)
    joblib.dump(lr, ENS_PATH)

print("Example models were created:")
print(f"- {RF_PATH}")
print(f"- {ENS_PATH}")
