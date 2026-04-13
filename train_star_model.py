import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from DataPre2 import load_and_preprocess

# Load preprocessed data or preprocess if not found
if os.path.exists("cleaned_reviews.csv"):
    print("✅ Using existing cleaned_reviews.csv")
    df = pd.read_csv("cleaned_reviews.csv")
else:
    print("⚙️  cleaned_reviews.csv not found. Preprocessing raw data...")
    df = load_and_preprocess("Reviews.csv")

# Drop rows without scores
df = df.dropna(subset=["Score", "cleaned"])

# Filter valid star ratings
df = df[df["Score"].isin([1, 2, 3, 4, 5])]

# Train/test split
X = df["cleaned"]
y = df["Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
print("🔤 Vectorizing...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
print("📊 Evaluation:")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model & vectorizer
print("💾 Saving model...")
joblib.dump(model, "star_rating_model.pkl")
joblib.dump(vectorizer, "star_vectorizer.pkl")
print("✅ Done.")
