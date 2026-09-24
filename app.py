import json
import os
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
import joblib
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash

# Must run before any os.environ lookups below
load_dotenv()

app = Flask(__name__)

cors_origin = "*"
if "CORS_ORIGIN" in os.environ:
    cors_origin = os.environ["CORS_ORIGIN"]
CORS(app, origins=cors_origin)

jwt_secret = "dev-secret-change-me"
if "JWT_SECRET" in os.environ:
    jwt_secret = os.environ["JWT_SECRET"]
else:
    print("WARNING: JWT_SECRET not set in .env, using an insecure dev default")

bundle = joblib.load("model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
FEATURES = bundle["features"]

with open("metrics.json") as f:
    metrics = json.load(f)

mongo_uri = "mongodb://localhost:27017"
if "MONGO_URI" in os.environ:
    mongo_uri = os.environ["MONGO_URI"]
client = MongoClient(mongo_uri)
db = client["fintrust"]
transactions = db["transactions"]
users = db["users"]

THRESHOLD = 0.5
RAW_FIELDS = ["amount", "hour", "distance_from_home_km", "txn_count_24h", "is_foreign", "card_present"]


def risk_level(prob):
    if prob >= 0.8:
        return "high"
    if prob >= THRESHOLD:
        return "medium"
    return "low"


def make_token(user_id, email):
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=12),
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = ""
        if "Authorization" in request.headers:
            header = request.headers["Authorization"]
        if not header.startswith("Bearer "):
            return jsonify({"error": "Login required"}), 401
        token = header[len("Bearer "):]
        try:
            payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid or expired token"}), 401
        g.user_id = payload["user_id"]
        g.email = payload["email"]
        return fn(*args, **kwargs)

    return wrapper


def read_credentials():
    data = request.get_json(silent=True)
    if data is None or "email" not in data or "password" not in data:
        return None, None, "Send email and password"
    email = str(data["email"]).strip().lower()
    password = str(data["password"])
    if "@" not in email:
        return None, None, "Enter a valid email"
    if len(password) < 6:
        return None, None, "Password must be at least 6 characters"
    return email, password, None


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics")
def get_metrics():
    return jsonify(metrics)


@app.route("/register", methods=["POST"])
def register():
    email, password, error = read_credentials()
    if error:
        return jsonify({"error": error}), 400
    users.create_index("email", unique=True)
    if users.find_one({"email": email}) is not None:
        return jsonify({"error": "Email already registered"}), 409
    doc = {
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    inserted = users.insert_one(doc)
    token = make_token(str(inserted.inserted_id), email)
    return jsonify({"token": token, "email": email}), 201


@app.route("/login", methods=["POST"])
def login():
    email, password, error = read_credentials()
    if error:
        return jsonify({"error": error}), 400
    user = users.find_one({"email": email})
    if user is None or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Wrong email or password"}), 401
    token = make_token(str(user["_id"]), email)
    return jsonify({"token": token, "email": email})


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Send JSON body"}), 400

    inputs = {}
    for key in RAW_FIELDS:
        if key not in data:
            return jsonify({"error": f"Missing field: {key}"}), 400
        try:
            inputs[key] = float(data[key])
        except (TypeError, ValueError):
            return jsonify({"error": f"{key} must be a number"}), 400

    inputs["is_night"] = 1.0 if inputs["hour"] < 5 else 0.0  # same feature engineering as train.py

    row_scaled = scaler.transform([[inputs[k] for k in FEATURES]])
    prob = float(model.predict_proba(row_scaled)[0][1])

    # Explainability: in logistic regression, coef * scaled value = contribution to fraud score
    contribs = []
    for i, name in enumerate(FEATURES):
        contribs.append({"feature": name, "impact": round(float(model.coef_[0][i] * row_scaled[0][i]), 3)})
    contribs.sort(key=lambda c: abs(c["impact"]), reverse=True)

    result = {
        "inputs": inputs,
        "fraud_probability": round(prob, 4),
        "flagged": prob >= THRESHOLD,
        "risk": risk_level(prob),
        "factors": contribs[:3],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    to_save = dict(result)
    to_save["user_id"] = g.user_id  # each check belongs to the logged-in user
    transactions.insert_one(to_save)
    return jsonify(result)


@app.route("/history")
@login_required
def history():
    docs = []
    for doc in transactions.find({"user_id": g.user_id}).sort("_id", -1).limit(10):
        doc["_id"] = str(doc["_id"])
        del doc["user_id"]
        docs.append(doc)
    return jsonify(docs)


@app.route("/stats")
@login_required
def stats():
    total = transactions.count_documents({"user_id": g.user_id})
    flagged = transactions.count_documents({"user_id": g.user_id, "flagged": True})
    return jsonify({"total": total, "flagged": flagged})


if __name__ == "__main__":
    app.run(port=5000, debug=True)