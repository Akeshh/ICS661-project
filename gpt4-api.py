from openai import OpenAI
import json
import os
from tqdm import tqdm

client = OpenAI(
    api_key = "your API keys",
)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def summarize_text(document, model="gpt-4o"):
    try:

        prompt = f"You are an expert at summarization. Summarize the following text: \n\n{document}\n\nSUMMARY:"
        

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at summarization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  
            max_tokens=100
        )
        
        # get generated summary
        summary = response.choices[0].message.content.strip()
        #print(summary)
        
        return summary, prompt  
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "", ""


def save_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def process_documents(input_file, output_file):
    data = load_data(input_file)
    
    for entry in tqdm(data, desc="Processing Documents"):
        document = entry.get('document')
        if document:
            summary, prompt = summarize_text(document)  
            entry['gpt4o-summary'] = summary  
            # entry['used_prompt'] = prompt  
    
    save_data(data, output_file)

# Specify the input and output file paths
input_file_path = './output/gpt4o-mini-generated.json'
output_file_path = './output/gpt4o-generated.json'

# Process the documents
process_documents(input_file_path, output_file_path)
