import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import re
import string
import pickle
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import nltk
import logging

logging.basicConfig(level=logging.DEBUG)
nltk.download('stopwords')

# Set MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("IMDB Sentiment Analysis")

# Load Dataset
df = pd.read_csv(r"C:\Users\Dell\Downloads\archive (1)\IMDB Dataset.csv")

# Text Cleaning Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

df['review'] = df['review'].apply(preprocess_text)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Convert to TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Log Model in MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "sentiment_model")
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

# Save Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Flask API
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Sentiment Analysis API!", "status": "running"})

@app.route("/best_model_parameter", methods=["GET"])
def get_model_params():
    return jsonify({"best_model": "Logistic Regression", "accuracy": accuracy_score(y_test, y_pred)})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = preprocess_text(data['review'])
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    return jsonify({"sentiment": "positive" if prediction == "positive" else "negative"})

@app.route("/train", methods=["POST"])
def retrain():
    model.fit(X_train_tfidf, y_train)
    return jsonify({"message": "Model retrained successfully"})

if __name__ == "__main__":
    print("🔥 API Running at: http://127.0.0.1:5000/")
    app.run(debug=True)
