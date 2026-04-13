from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from openai import OpenAI
from DataPre2 import load_and_preprocess, clean_text
from SentAnalysis2 import aggregate_sentiment
from Sum2 import generate_summary
from amazon_scraper import scrape_amazon_reviews

# Load data and models
app = Flask(__name__)
df = load_and_preprocess("cleaned_reviews.csv")

# Load sentiment model
try:
    sentiment_model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except:
    sentiment_model = None
    vectorizer = None

# Load star rating model
try:
    star_model = joblib.load("star_rating_model.pkl")
    star_vectorizer = joblib.load("star_vectorizer.pkl")
except:
    star_model = None
    star_vectorizer = None

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def predict_sentiment_with_model(texts):
    if sentiment_model and vectorizer:
        X_vec = vectorizer.transform(texts)
        preds = sentiment_model.predict(X_vec)
        return {
            "positive": float((preds == "positive").mean()),
            "neutral": float((preds == "neutral").mean()),
            "negative": float((preds == "negative").mean())
        }
    else:
        return aggregate_sentiment(texts)

def predict_star_rating(texts):
    if star_model and star_vectorizer:
        X = star_vectorizer.transform(texts)
        preds = star_model.predict(X)
        return preds.tolist()
    else:
        return ["Unavailable"] * len(texts)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    review_ids = request.json.get("review_ids", [])

    if not review_ids or any(i < 0 or i >= len(df) for i in review_ids):
        return jsonify({"error": "Invalid review IDs"}), 400

    selected_reviews = df.iloc[review_ids]["cleaned"].tolist()
    combined_reviews = "\n".join(selected_reviews)

    summary = generate_summary(combined_reviews)
    sentiment = predict_sentiment_with_model(selected_reviews)
    star_ratings = predict_star_rating(selected_reviews)
    avg_rating = round(sum([r for r in star_ratings if isinstance(r, int)]) / len(star_ratings), 2) if star_ratings and isinstance(star_ratings[0], int) else "Unavailable"

    return jsonify({
        "summary": summary,
        "sentiment": sentiment,
        "count": len(selected_reviews),
        "star_rating": {
            "average": avg_rating,
            "predicted": star_ratings[:5]
        }
    })

@app.route("/compare", methods=["POST"])
def compare():
    data = request.json
    product_ids = data.get("product_ids", [])

    if not product_ids or len(product_ids) != 2:
        return jsonify({"error": "Please provide two product IDs"}), 400

    results = []

    for pid in product_ids:
        product_reviews = df[df["ProductId"] == pid]

        if product_reviews.empty:
            return jsonify({"error": f"No reviews found for product {pid}"}), 404

        reviews = product_reviews["cleaned"].tolist()
        combined = "\n".join(reviews[:20])

        summary = generate_summary(combined)
        sentiment = predict_sentiment_with_model(reviews)
        star_ratings = predict_star_rating(reviews)
        avg_rating = round(sum([r for r in star_ratings if isinstance(r, int)]) / len(star_ratings), 2) if star_ratings and isinstance(star_ratings[0], int) else "Unavailable"

        results.append({
            "product_id": pid,
            "product_name": f"Product {pid}",
            "summary": summary,
            "sentiment": sentiment,
            "count": len(reviews),
            "star_rating": {
                "average": avg_rating,
                "predicted": star_ratings[:5]
            }
        })

    return jsonify({"products": results})

@app.route("/summarize_from_link", methods=["POST"])
def summarize_from_link():
    data = request.json
    url = data.get("url")

    if not url or "/dp/" not in url:
        return jsonify({"error": "Invalid product URL format."}), 400

    try:
        product_id = url.split("/dp/")[1].split("/")[0]
    except Exception:
        return jsonify({"error": "Failed to extract ProductId from URL."}), 400

    product_reviews = df[df["ProductId"] == product_id]

    if product_reviews.empty:
        return jsonify({"error": f"No reviews found for product {product_id}"}), 404

    reviews = product_reviews["cleaned"].tolist()
    combined = "\n".join(reviews[:20])

    summary = generate_summary(combined)
    sentiment = predict_sentiment_with_model(reviews)
    star_ratings = predict_star_rating(reviews)
    avg_rating = round(sum([r for r in star_ratings if isinstance(r, int)]) / len(star_ratings), 2) if star_ratings and isinstance(star_ratings[0], int) else "Unavailable"

    return jsonify({
        "product_id": product_id,
        "product_name": f"Product {product_id}",
        "summary": summary,
        "sentiment": sentiment,
        "count": len(reviews),
        "star_rating": {
            "average": avg_rating,
            "predicted": star_ratings[:5]
        }
    })

@app.route("/live_amazon", methods=["POST"])
def live_amazon():
    data = request.json
    url = data.get("url")

    try:
        raw_reviews = scrape_amazon_reviews(url)
        if not raw_reviews:
            return jsonify({"error": "No reviews found."}), 404

        cleaned = [clean_text(r) for r in raw_reviews]
        combined = "\n".join(cleaned)

        summary = generate_summary(combined)
        sentiment = predict_sentiment_with_model(cleaned)
        star_ratings = predict_star_rating(cleaned)
        avg_rating = round(sum([r for r in star_ratings if isinstance(r, int)]) / len(star_ratings), 2) if star_ratings and isinstance(star_ratings[0], int) else "Unavailable"

        return jsonify({
            "product_url": url,
            "summary": summary,
            "sentiment": sentiment,
            "count": len(cleaned),
            "star_rating": {
                "average": avg_rating,
                "predicted": star_ratings[:5]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare_live_amazon", methods=["POST"])
def compare_live_amazon():
    data = request.json
    urls = data.get("urls", [])

    if not urls or len(urls) != 2:
        return jsonify({"error": "Please provide two Amazon product URLs"}), 400

    results = []

    for url in urls:
        try:
            raw_reviews = scrape_amazon_reviews(url)
            if not raw_reviews:
                return jsonify({"error": f"No reviews found for product {url}"}), 404

            cleaned = [clean_text(r) for r in raw_reviews]
            combined = "\n".join(cleaned)
            summary = generate_summary(combined)
            sentiment = predict_sentiment_with_model(cleaned)
            star_ratings = predict_star_rating(cleaned)
            avg_rating = round(sum([r for r in star_ratings if isinstance(r, int)]) / len(star_ratings), 2) if star_ratings and isinstance(star_ratings[0], int) else "Unavailable"

            results.append({
                "product_url": url,
                "summary": summary,
                "sentiment": sentiment,
                "count": len(cleaned),
                "star_rating": {
                    "average": avg_rating,
                    "predicted": star_ratings[:5]
                }
            })
        except Exception as e:
            return jsonify({"error": f"Error scraping {url}: {str(e)}"}), 500

    return jsonify({"products": results})

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    url = data.get("url")
    question = data.get("question")

    if not url or not question:
        return jsonify({"error": "URL and question are required."}), 400

    try:
        reviews = scrape_amazon_reviews(url)
        cleaned = [clean_text(r) for r in reviews][:20]
        combined_reviews = "\n".join(cleaned)

        prompt = f"""Answer the question using only the following customer reviews:\n\n{combined_reviews}\n\nQuestion: {question}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{ "role": "user", "content": prompt }]
        )

        answer = response.choices[0].message.content.strip()
        return jsonify({ "answer": answer })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    print("✅ Registered Routes:")
    print(app.url_map)
    app.run(debug=True, port=5000)
