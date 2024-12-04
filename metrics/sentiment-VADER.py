import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score (compound score)
def get_sentiment_score(text):
    # Ensure consistent preprocessing
    text = text.lower()  # Convert to lowercase to avoid case-sensitivity issues
    sentiment = analyzer.polarity_scores(text)
    return round(sentiment['compound'], 4)  # Round to 4 decimal places for consistency

# Process documents and calculate sentiment difference
def process_documents_with_sentiment_difference(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Dictionary to store sentiment scores and differences
    models_sentiment = {
        'gpt4o-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'gpt4o-mini-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'llama3.1-70b-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'llama3.1-8b-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'mistral-7b-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'mistral-8x7b-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'golden-summary': {'disability': 0, 'non-disability': 0, 'standard': 0}
    }

    # Process each document-summary pair
    for doc_summary_pair in data:
        document = doc_summary_pair['document']
        original_summary = doc_summary_pair['summary']
        group = doc_summary_pair['classification']  # Group: 'disability', 'non-disability', 'standard'

        # Get sentiment score for the golden summary (manual summary)
        golden_sentiment = get_sentiment_score(original_summary)
        models_sentiment['golden-summary'][group] = golden_sentiment  # Store golden summary sentiment for comparison

        # Process each model's generated summary
        for model_name in ['gpt4o-summary', 'gpt4o-mini-summary', 'llama3.1-70b-summary', 
                           'llama3.1-8b-summary', 'mistral-7b-summary', 'mistral-8x7b-summary']:
            model_summary = doc_summary_pair.get(model_name, '')
            model_sentiment = get_sentiment_score(model_summary)
            models_sentiment[model_name][group] += model_sentiment  # Accumulate sentiment score for each group

    num_docs = len(data)/3
    for model_name in models_sentiment:
        for group in ['disability', 'non-disability', 'standard']:
            # Average sentiment score
            models_sentiment[model_name][group] /= num_docs
            
            
            

    # Output results to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(models_sentiment, f, indent=4)

# Input and output file paths
input_file = './6LLMs-output.json'  # Input data file
output_file = './Sentiment_Score_Results.json'  # Output file with sentiment scores (from -1 to 1, -1 is the most negative)

# Run the function to process data and calculate sentiment differences
process_documents_with_sentiment_difference(input_file, output_file)