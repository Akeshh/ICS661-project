from huggingface_hub import InferenceClient
import json
from tqdm import tqdm
import time

client = InferenceClient(
    "meta-llama/Llama-3.1-70B-Instruct",
    token="your API keys",
)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def summarize_text(document):
    try:
        
        prompt = f"You are an expert at summarization. Summarize the following text: \n\n{document}\n\nSUMMARY:"
        
        summary = ""
        for message in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            stream=True,
        ):
        
            summary += message.choices[0].delta.content
        
        return summary.strip(), prompt   
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "", ""

def save_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_documents(input_file, output_file):
    data = load_data(input_file)
    
    for entry in tqdm(data, desc="Processing documents"):
        document = entry.get('document')
        if document:
            summary, prompt = summarize_text(document)  
            entry['llama3.1-70b-summary'] = summary  
            # entry['used_prompt'] = prompt  
    
    save_data(data, output_file)

def delay_process(minutes, input_file, output_file):
    delay_seconds = minutes * 60  # Convert minutes to seconds
    print(f"Process will start after {minutes} minute(s).")
    time.sleep(delay_seconds)  # Delay the execution
    process_documents(input_file, output_file)  # Execute processing function after delay

input_file_path = './output/llama3.1-8b-generated.json'
output_file_path = './output/llama3.1-70b-generated.json'

# Call delay_process to run process_documents after a delay
delay_time_in_minutes = 0  # Set to 60 minutes later, can modify to 0 or other values
delay_process(delay_time_in_minutes, input_file_path, output_file_path)
