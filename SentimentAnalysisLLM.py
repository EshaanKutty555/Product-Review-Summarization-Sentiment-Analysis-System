import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import openai

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    """
    Clean review text by removing HTML tags, special characters, and converting text to lowercase.
    """
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text.lower()

def remove_stopwords(text):
    """
    Tokenize the text and remove common English stopwords.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

def preprocess_reviews(df, text_column='Text'):
    """
    Apply text cleaning and stopword removal to the dataset.
    """
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df['cleaned_text_no_stop'] = df['cleaned_text'].apply(remove_stopwords)
    return df

def analyze_sentiment(text):
    """
    Analyze sentiment of a given text using VADER.
    """
    return sia.polarity_scores(text)

def aggregate_sentiment(reviews):
    """
    Aggregate sentiment scores over a list of reviews.
    """
    aggregated = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    for review in reviews:
        scores = analyze_sentiment(review)
        for key in aggregated:
            aggregated[key] += scores[key]
    return {k: v / len(reviews) for k, v in aggregated.items()}

def generate_summary(review_text, model="gpt-3.5-turbo", max_tokens=150):
    """
    Generate a summary from product reviews using OpenAI's GPT model.
    """
    openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with actual API key
    prompt = (
        "Summarize the following product reviews by extracting key themes, listing pros and cons, "
        "and providing an overall sentiment assessment.\n\n"
        f"Product Reviews:\n{review_text}\n\nSummary:"
    )
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

# Example usage
if __name__ == "__main__":
    sample_reviews = [
        "I love this product! It exceeded my expectations.",
        "The product is okay, but the customer service was unhelpful.",
        "Terrible experience. The item broke after one use and support did nothing."
    ]
    
    # Sentiment Analysis
    aggregated_scores = aggregate_sentiment(sample_reviews)
    print("Aggregated Sentiment Scores:", aggregated_scores)
    
    # LLM Summarization
    combined_reviews = "\n".join(sample_reviews)
    summary_result = generate_summary(combined_reviews)
    print("\nGenerated Summary:", summary_result)
