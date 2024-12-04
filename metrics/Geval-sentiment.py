from openai import OpenAI
import json
import os
from tqdm import tqdm
import re

# Initialize OpenAI client
client = OpenAI(
    api_key="your api",  # Replace with your actual API key
)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def evaluate_sentiment(summary, model="gpt-4o"):
    try:
        prompt = f"""You will be given one summary written for a news article. Your task is to rate the summary based on its sentiment orientation. Please make sure you read and understand these instructions carefully. Keep this document open while reviewing, and refer to it as needed.

        Evaluation Criteria:

        Sentiment Orientation (1-5): The overall sentiment of the summary. We classify sentiment into three categories:
        1: Strongly Negative – The summary expresses a predominantly negative sentiment.
        2: Moderately Negative – The summary has a slightly negative tone.
        3: Neutral – The summary maintains a neutral tone with no clear positive or negative sentiment.
        4: Moderately Positive – The summary expresses a slightly positive sentiment.
        5: Strongly Positive – The summary expresses a predominantly positive sentiment.

        Evaluation Steps:
        1. Read the summary carefully.
        2. Identify sentiment indicators: Look for words or phrases that suggest sentiment, such as adjectives (e.g., “positive”, “unfortunate”), verbs (e.g., “improve”, “decline”), or adverbs (e.g., “dramatically”, “slightly”).
        3. Determine the overall tone: Consider how the overall sentiment of the summary is conveyed. Is it more positive, negative, or neutral? Does the sentiment shift throughout the summary, or is it consistent?
        4. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

        Here is the summary:
        {summary}

        Please rate the sentiment on a scale of 1 to 5.

        Evaluation Form (scores ONLY):
        - Sentiment:
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        
        # Extract sentiment score using a regular expression to capture a number between 1 and 5
        match = re.search(r'\b([1-5])\b', response.choices[0].message.content)
        #print(response.choices[0].message.content)
        if match:
            sentiment_score = int(match.group(1))  # Get the matched number
            #print(sentiment_score)
            return sentiment_score
        else:
            print(f"Unable to extract sentiment score from response: {response.choices[0].message.content}")
            return None
    except Exception as e:
        print(f"Error during sentiment evaluation: {e}")
        return None

def process_documents(input_file, output_file):
    data = load_data(input_file)
    
    for entry in tqdm(data, desc="Processing Documents"):
        # Evaluate sentiment for each of the 7 summaries
        for key in ['summary', 'gpt4o-mini-summary', 'gpt4o-summary', 'mistral-8x7b-summary', 
                    'mistral-7b-summary', 'llama3.1-8b-summary', 'llama3.1-70b-summary']:
            summary = entry.get(key)
            if summary:
                sentiment_score = evaluate_sentiment(summary)
                if sentiment_score is not None:
                    entry[f'{key}-score'] = sentiment_score
    
    save_data(data, output_file)

def save_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Specify the input and output file paths
input_file_path = './6LLMs-output.json'
output_file_path = './geval-result.json'

# Process the documents
process_documents(input_file_path, output_file_path)
