# ========================================
# Section 1: Data Collection & Preprocessing
# ========================================

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """
    Clean review text by:
    - Removing HTML tags
    - Removing special characters
    - Converting text to lowercase
    """
    # Remove HTML tags
    clean = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters (keep spaces)
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', clean)
    # Convert to lowercase
    clean = clean.lower()
    return clean

def remove_stopwords(text):
    """
    Tokenize the text and remove common English stopwords.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_reviews(file_path, text_column='Text'):
    """
    Load and preprocess reviews from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column that contains the review text.
    
    Returns:
        df (DataFrame): Pandas DataFrame with additional columns for cleaned text.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Clean the review text
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Remove stopwords
    df['cleaned_text_no_stop'] = df['cleaned_text'].apply(remove_stopwords)
    
    return df

# Example usage:
if __name__ == "__main__":
    # Provide the path to your dataset (e.g., "amazon_fine_food_reviews.csv")
    file_path = 'amazon_fine_food_reviews.csv'
    df_reviews = preprocess_reviews(file_path, text_column='Text')
    print("Sample of preprocessed reviews:")
    print(df_reviews[['Text', 'cleaned_text_no_stop']].head())
    
    
# ============================================
# Section 2: LLM Integration & Prompt Engineering
# ============================================

import openai

# Set your OpenAI API key (ensure you keep this key secure)
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your actual key

def generate_summary(review_text, model="gpt-3.5-turbo", max_tokens=150):
    """
    Generate a summary from product reviews using an LLM.
    
    Parameters:
        review_text (str): Combined review texts to summarize.
        model (str): OpenAI model to use (e.g., "gpt-3.5-turbo").
        max_tokens (int): Maximum number of tokens for the summary output.
    
    Returns:
        summary (str): Generated summary of the reviews.
    """
    # Construct prompt for the LLM
    prompt = (
        "You are an assistant that summarizes product reviews. "
        "Below is a collection of product reviews. Extract the main themes, list key pros and cons, "
        "and provide a concise summary of the overall sentiment. "
        "Focus on aspects such as quality, durability, and usability.\n\n"
        "Product Reviews:\n"
        f"{review_text}\n\n"
        "Summary:"
    )
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    
    summary = response['choices'][0]['message']['content'].strip()
    return summary

# Example usage:
if __name__ == "__main__":
    # Combine a sample of the preprocessed reviews for summarization
    # For example, using the first 3 reviews:
    sample_reviews = df_reviews['cleaned_text_no_stop'].head(3).tolist()
    combined_reviews = "\n".join(sample_reviews)
    
    # Generate summary using the LLM
    summary_result = generate_summary(combined_reviews)
    
    print("\nGenerated Summary:")
    print(summary_result)