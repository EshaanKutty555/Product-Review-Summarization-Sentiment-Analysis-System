import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon (run only once)
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using VADER.
    
    Parameters:
        text (str): The review text to analyze.
    
    Returns:
        dict: A dictionary with sentiment scores for negative (neg), neutral (neu), 
              positive (pos), and compound (compound) sentiment.
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores

def aggregate_sentiment(reviews):
    """
    Aggregate sentiment scores over a list of reviews.
    
    Parameters:
        reviews (list): A list of review strings.
    
    Returns:
        dict: Average sentiment scores across all reviews.
    """
    aggregated = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    n = len(reviews)
    
    # Sum sentiment scores for all reviews
    for review in reviews:
        scores = analyze_sentiment(review)
        aggregated['neg'] += scores['neg']
        aggregated['neu'] += scores['neu']
        aggregated['pos'] += scores['pos']
        aggregated['compound'] += scores['compound']
    
    # Calculate average scores
    averaged_scores = {k: v / n for k, v in aggregated.items()}
    return averaged_scores

# Example usage:
if __name__ == "__main__":
    # Example individual review
    review_text = "The product quality is excellent, but the shipping was very slow."
    sentiment_scores = analyze_sentiment(review_text)
    print("Individual Review Sentiment Scores:")
    print(sentiment_scores)
    
    # Example list of reviews for aggregation
    sample_reviews = [
        "I love this product! It exceeded my expectations.",
        "The product is okay, but the customer service was unhelpful.",
        "Terrible experience. The item broke after one use and support did nothing."
    ]
    
    aggregated_scores = aggregate_sentiment(sample_reviews)
    print("\nAggregated Sentiment Scores for Sample Reviews:")
    print(aggregated_scores)