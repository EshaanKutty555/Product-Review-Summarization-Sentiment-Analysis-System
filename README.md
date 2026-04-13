# Product-Review-Summarizer-and-Sentiment-Analyzer


Set Up Environment:

Install Python: Ensure you have Python 3.7+ installed. You can download it from python.org.
Create a Virtual Environment (Optional but Recommended):
bash
Copy
Edit
python -m venv myenv
source myenv/bin/activate   # On Windows: myenv\Scripts\activate
Install Required Packages:
Open your terminal or command prompt and run:
bash
Copy
Edit
pip install pandas nltk openai
Additionally, if you need to install any other packages (like flask if you plan to build an API later), you can add those as needed.
Download or Prepare Your Dataset:

Ensure you have a CSV file with your review data. For example, download the Amazon Fine Food Reviews dataset and save it as amazon_fine_food_reviews.csv in your project directory.
Make sure the CSV file contains a column named "Text" (or update the code with the correct column name).
Set Up Your Code File:

Create a new Python file (e.g., main.py) in your project directory.
Copy and paste the provided code into main.py.
Configure the OpenAI API Key:

Replace the placeholder "YOUR_OPENAI_API_KEY" with your actual OpenAI API key in the code. You can get your API key from your OpenAI account dashboard.
Run the Code:

Open your terminal, navigate to your project directory, and run:
python DataCollection&Preprocessing.py
The script will load and preprocess your dataset, print a sample of the preprocessed reviews, and then use the LLM to generate a summary from the first few reviews.
Modify & Experiment:

Preprocessing: You can adjust or extend the cleaning functions if your dataset has different formatting.
Prompt Engineering: Modify the prompt in the generate_summary function to better suit your needs.
Testing: Experiment with different sections of your dataset by adjusting which rows you select for summarization.





DataCollection&Preprocessing.py: How it works

Data Collection & Preprocessing:

Data Loading:
The preprocess_reviews function reads a CSV file (e.g., the Amazon Fine Food Reviews dataset) into a Pandas DataFrame.
Text Cleaning:
The clean_text function removes HTML tags, non-alphanumeric characters, and converts text to lowercase.
Stopwords Removal:
The remove_stopwords function tokenizes the text and removes common English stopwords using NLTK.
Preprocessing Output:
The resulting DataFrame includes columns with both the cleaned text and the text with stopwords removed, which will be used for further processing.
LLM Integration & Prompt Engineering:

OpenAI API Setup:
The code initializes the OpenAI API with your API key.
Prompt Engineering:
The generate_summary function constructs a detailed prompt that instructs the LLM to extract key themes, pros/cons, and overall sentiment from the given review text.
LLM Call:
The function calls the OpenAI ChatCompletion API (using a model like "gpt-3.5-turbo") to generate the summary, which is then returned.

Sentiment Analysis:
Explanation:
Downloading VADER Lexicon:

nltk.download('vader_lexicon') ensures that the required sentiment lexicon is available.
analyze_sentiment Function:

Creates an instance of SentimentIntensityAnalyzer.
Computes sentiment scores for the input text and returns a dictionary with four scores:
neg: Proportion of negative sentiment.
neu: Proportion of neutral sentiment.
pos: Proportion of positive sentiment.
compound: A normalized score summarizing the overall sentiment.
aggregate_sentiment Function:

Iterates over a list of review texts.
Sums and averages the sentiment scores to provide an overall sentiment profile for a group of reviews.
Example Usage:

Analyzes sentiment for a single review.
Aggregates sentiment scores for a sample list of reviews and prints both results.
