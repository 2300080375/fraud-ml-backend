FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# train at build time so the image ships with model.joblib
RUN python train.py
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2
