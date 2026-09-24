# Fraud detection: ML + Flask API + MongoDB (PyCharm project)

Files:
- `train.py`  trains the model, writes `model.joblib` + `metrics.json`
- `app.py`    Flask API, loads the model, logs to MongoDB
- `api.http`  ready-made requests (PyCharm Professional HTTP client; else use curl)
- `Dockerfile`, `docker-compose.yml`  for running/deploying

## PyCharm setup
1. File > Open > select this folder.
2. Settings > Project > Python Interpreter > Add Interpreter > Virtualenv (new, Python 3.10+).
3. Open `requirements.txt`, click "Install requirements" (or terminal: `pip install -r requirements.txt`).
4. Right-click `train.py` > Run. You should see metrics printed and `model.joblib` created.
5. Start MongoDB: `docker run -d -p 27017:27017 mongo:7` (or local mongod / Atlas URI).
6. Right-click `app.py` > Run. API is on http://localhost:5000.
7. Open `api.http` and click the green run arrow next to each request.

Using Atlas instead of local Mongo: Run > Edit Configurations > app.py > Environment variables:
`MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/`

## Endpoints
POST /predict, GET /history, GET /stats, GET /metrics, GET /health

## Things to try (hands-on)
1. In `train.py`, change the fraud logic or `class_weight` and compare metrics.
2. Swap LogisticRegression for RandomForestClassifier (loses the coef-based explanations; add SHAP).
3. Change `THRESHOLD` in `app.py` and watch precision/recall tradeoff.
4. Look at saved documents in MongoDB Compass (connect to mongodb://localhost:27017, db `fintrust`).
5. Replace synthetic data with the Kaggle credit card fraud CSV.

## Deploy (Render + Atlas)
1. Atlas: free M0 cluster, create DB user, allow network access, copy the connection URI.
2. Push this folder to GitHub.
3. Render: New > Web Service > repo, Runtime: Docker (the Dockerfile trains the model during build).
4. Env vars on Render: `MONGO_URI` (Atlas URI), `CORS_ORIGIN` (your frontend URL, no trailing slash).
5. Check `https://<service>.onrender.com/health`.
