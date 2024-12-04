import json
import nltk
from nltk.util import ngrams
from collections import Counter

# Ensure that nltk data is downloaded
#nltk.download('punkt')

# Calculate the diversity of n-grams
def calculate_diversity(text, n=3):
    """
    Calculate the diversity of the text: average diversity from 1-gram to n-gram
    """
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    total_ngrams = 0
    unique_ngrams = 0

    # Calculate 1-gram to n-gram
    for i in range(1, n+1):
        n_grams = list(ngrams(tokens, i))  # Generate n-grams
        total_ngrams += len(n_grams)
        unique_ngrams += len(set(n_grams))  # Count unique n-grams

    # Return average diversity
    if total_ngrams > 0:
        return unique_ngrams / total_ngrams
    else:
        return 0

# Process all documents and summaries to calculate diversity
def process_documents(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models_diversity = {
        'gpt4o-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'gpt4o-mini-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'llama3.1-70b-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'llama3.1-8b-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'mistral-7b-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'mistral-8x7b-summary': {'disability': 0, 'non-disability': 0, 'standard': 0},
        'golden-summary': {'disability': 0, 'non-disability': 0, 'standard': 0}
    }

    # Create a dictionary to count the number of documents in each group
    group_counts = {'disability': 0, 'non-disability': 0, 'standard': 0}
    
    # Calculate diversity for each model and original summary
    diversity_diff = {model: {'disability': 0, 'non-disability': 0, 'standard': 0} for model in models_diversity}

    # Calculate diversity for each model and original summary
    for doc_summary_pair in data:
        # Get document, summary, and group information
        document = doc_summary_pair['document']
        original_summary = doc_summary_pair['summary']
        group = doc_summary_pair['classification']  # Could be 'disability', 'non-disability', or 'standard'

        # Count the number of documents in each group
        if group in group_counts:
            group_counts[group] += 1

        # Calculate diversity for the document and summary
        document_diversity = calculate_diversity(document, n=3)
        summary_diversity = calculate_diversity(original_summary, n=3)

        # Update the diversity for the original summary
        models_diversity['golden-summary'][group] += summary_diversity

        # Perform the same processing for each generated model
        for model_name in ['gpt4o-summary', 'gpt4o-mini-summary', 'llama3.1-70b-summary', 
                           'llama3.1-8b-summary', 'mistral-7b-summary', 'mistral-8x7b-summary']:
            model_summary = doc_summary_pair.get(model_name, '')
            model_diversity = calculate_diversity(model_summary, n=3)
            models_diversity[model_name][group] += model_diversity
            
            # Calculate the diversity difference between each model and the golden-summary
            diversity_diff[model_name][group] += (model_diversity - summary_diversity)
    
    # Calculate the average diversity for each model
    for model_name in models_diversity:
        for group in ['disability', 'non-disability', 'standard']:
            if group_counts[group] > 0:
                models_diversity[model_name][group] /= group_counts[group]
                diversity_diff[model_name][group] /= group_counts[group]

    # Output the results to a file
    result = {
        'models_diversity': models_diversity,
        'diversity_diff': diversity_diff
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

# Input and output file paths
input_file = './6LLMs-output.json'  # Path to the original data file
output_file = './Diversity_results.json'  # Path to the results file

# Process the data and calculate diversity
process_documents(input_file, output_file)
