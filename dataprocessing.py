import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Downloads\archive (1)\IMDB Dataset.csv")

# Text Cleaning Function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer for future use
import pickle
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Save data
pickle.dump((X_train_tfidf, X_test_tfidf, y_train, y_test), open("data.pkl", "wb"))
print("✅ Data preprocessing completed!")