"""Train a fraud detection model on synthetic card transactions."""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# "hour" is turned into an is_night flag (feature engineering); app.py does the same at predict time
FEATURES = ["amount", "distance_from_home_km", "txn_count_24h", "is_foreign", "card_present", "is_night"]

rng = np.random.default_rng(42)
n = 30000

amount = rng.lognormal(mean=3.8, sigma=1.0, size=n)          # most txns small, some big
hour = rng.integers(0, 24, size=n)
distance = rng.exponential(scale=15, size=n)
txn_count = rng.poisson(3, size=n)
is_foreign = rng.binomial(1, 0.06, size=n)
card_present = rng.binomial(1, 0.7, size=n)

night = (hour < 5).astype(float)
logit = (-7.8 + 0.008 * amount + 3.0 * is_foreign + 0.03 * distance
         + 0.5 * txn_count + 2.5 * night - 2.0 * card_present)
p = 1 / (1 + np.exp(-logit))
y = (rng.random(n) < p).astype(int)

X = np.column_stack([amount, distance, txn_count, is_foreign, card_present, night])
print(f"Rows: {n}, fraud rate: {y.mean():.3%}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# class_weight="balanced" handles the imbalance (fraud is rare)
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_s, y_train)

probs = model.predict_proba(X_test_s)[:, 1]
preds = (probs >= 0.5).astype(int)
print(classification_report(y_test, preds, digits=3))

metrics = {
    "roc_auc": round(float(roc_auc_score(y_test, probs)), 3),
    "precision": round(float(precision_score(y_test, preds)), 3),
    "recall": round(float(recall_score(y_test, preds)), 3),
    "fraud_rate": round(float(y.mean()), 4),
    "train_rows": int(len(y_train)),
}
print(metrics)

joblib.dump({"model": model, "scaler": scaler, "features": FEATURES}, "model.joblib")
with open("metrics.json", "w") as f:
    json.dump(metrics, f)
print("Saved model.joblib and metrics.json")
