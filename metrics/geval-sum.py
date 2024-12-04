import json
import numpy as np

def load_data(file_path):
    """Load data"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def calculate_average_scores(data):
    """Group by classification field and calculate the average score for each model"""
    model_scores = {
        'disability': {
            'golden-summary': [],
            'gpt4o-mini-summary': [],
            'gpt4o-summary': [],
            'mistral-8x7b-summary': [],
            'mistral-7b-summary': [],
            'llama3.1-8b-summary': [],
            'llama3.1-70b-summary': [],
        },
        'non-disability': {
            'golden-summary': [],
            'gpt4o-mini-summary': [],
            'gpt4o-summary': [],
            'mistral-8x7b-summary': [],
            'mistral-7b-summary': [],
            'llama3.1-8b-summary': [],
            'llama3.1-70b-summary': [],
        },
        'standard': {
            'golden-summary': [],
            'gpt4o-mini-summary': [],
            'gpt4o-summary': [],
            'mistral-8x7b-summary': [],
            'mistral-7b-summary': [],
            'llama3.1-8b-summary': [],
            'llama3.1-70b-summary': [],
        }
    }

    # Group data by classification field
    for entry in data:
        classification = entry.get('classification', 'standard')  # Default to 'standard' group
        if classification in model_scores:
            model_scores[classification]['golden-summary'].append(entry.get('summary-score', 0))
            model_scores[classification]['gpt4o-mini-summary'].append(entry.get('gpt4o-mini-summary-score', 0))
            model_scores[classification]['gpt4o-summary'].append(entry.get('gpt4o-summary-score', 0))
            model_scores[classification]['mistral-8x7b-summary'].append(entry.get('mistral-8x7b-summary-score', 0))
            model_scores[classification]['mistral-7b-summary'].append(entry.get('mistral-7b-summary-score', 0))
            model_scores[classification]['llama3.1-8b-summary'].append(entry.get('llama3.1-8b-summary-score', 0))
            model_scores[classification]['llama3.1-70b-summary'].append(entry.get('llama3.1-70b-summary-score', 0))

    # Calculate the average score for each group
    average_scores = {}
    for group, scores in model_scores.items():
        average_scores[group] = {}
        for model, scores_list in scores.items():
            average_scores[group][model] = np.mean(scores_list) if scores_list else 0

    return average_scores

def save_results(results, output_file):
    """Save the results to a new JSON file"""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

# Input and output file paths
input_file_path = './geval-result.json'
output_file_path = './geval-final-sum-result.json'

# Load data
data = load_data(input_file_path)

# Calculate average scores
average_scores = calculate_average_scores(data)

# Save results
save_results(average_scores, output_file_path)

print(f"Average sentiment scores by group saved to {output_file_path}")
