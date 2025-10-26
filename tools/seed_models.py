import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

os.makedirs("data/models", exist_ok=True)

# ---- RANDOM FOREST MODEL ----
X = np.random.rand(300, 20)
y = X.sum(axis=1) * 10 + np.random.rand(300)
rf = RandomForestRegressor(n_estimators=30, random_state=42)
rf.fit(X, y)
joblib.dump(rf, "data/models/random_forest_model.pkl")

# ---- ENSEMBLE MODEL ----
Z = np.random.rand(300, 5) * 100
t = Z[:, :3].mean(axis=1) + np.random.rand(300)
lr = LinearRegression()
lr.fit(Z, t)
joblib.dump(lr, "data/models/ensemble_model.pkl")

print("✅ Example models were created:")
print("- data/models/random_forest_model.pkl")
print("- data/models/ensemble_model.pkl")
