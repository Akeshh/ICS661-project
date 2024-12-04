import json
import random
from collections import Counter
from nltk.tokenize import word_tokenize
import string

# Define vocabulary lists
disability_terms = [
    "disabled people", "disabled person",  "disability", 
    "visually impaired", "physically challenged", "mental handicapped",
    "blindness", "deafblind", "epilepsy", "cerebral palsy", "blind people", "blind person", "deaf person", "deaf person"
    "ADHD", "autism", "visually challenged", "quadriplegic", 
    "congenital disorder", "intellectually disabled", "chronic illness", "dyslexia",
    "Alzheimer’s disease", "depressive disorder", "psychosis"
]

non_disability_terms = [
    "abled", "non-disabled", "able-bodied", "neurotypical", "physically capable", "without disability",
    "cognitively intact", "no physical limitations", "hearing-abled", "physically capable", "cognitively intact",  "physical fitness", 
    "mentally healthy",  "job-capable",
    "self-sufficient", "intellectually functional",  "mentally intact", "physically fit", "emotionally resilient", "psychologically sound", "psychologically resilient",

    # Additional terms
    "neurologically typical",  "functionally complete", "cognitive stability",
    "physically resilient",  "mentally active",  
    "independent living",
    "independently mobile",  
    "mental clarity", "physical independence", "disease-free", "mentally proficient", "completely functional",
    "unrestricted mobility", "physically autonomous"
]

# Function to classify and sample terms using word_tokenize
def classify_and_sample(input_file, output_file, sample_size):
    # Open JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Store the classified data
    disability_data = []
    non_disability_data = []
    standard_data = []
    
    # Define a helper function to count terms
    # Define a helper function to count terms, including multi-word phrases
    def count_terms(document, term_list):
        # Tokenize the document and clean tokens (remove punctuation and lower case everything)
        tokens = [word.lower() for word in word_tokenize(document) if word.isalnum()]
        
        term_count = Counter()
        
        # Count multi-word phrases first, then single words
        for term in term_list:
            term_tokens = term.split()  # Split phrase into individual words
            term_tokens_lower = [word.lower() for word in term_tokens]
            
            # Check if multi-word phrase exists in the token list
            if len(term_tokens) > 1:
                phrase_count = 0
                for i in range(len(tokens) - len(term_tokens) + 1):
                    # Check if the sequence of words matches the term
                    if tokens[i:i+len(term_tokens)] == term_tokens_lower:
                        phrase_count += 1
                term_count[term] = phrase_count
            else:
                # Check if single word exists in the token list
                term_count[term] = tokens.count(term.lower())
        
        # Only return terms with count >= 1
        return {term: count for term, count in term_count.items() if count >= 1}
    
    # Iterate through the data and classify
    for item in data:
        document = item.get("document", "").lower()
        
        # Count occurrences of disability and non-disability terms
        disability_count = count_terms(document, disability_terms)
        non_disability_count = count_terms(document, non_disability_terms)
        
        # Classification logic
        if disability_count and not non_disability_count:
            classification = "disability"
            disability_data.append(item)
        elif non_disability_count and not disability_count:
            classification = "non-disability"
            non_disability_data.append(item)
        elif disability_count and non_disability_count:
            classification = "mixed"
        else:
            classification = "standard"
            standard_data.append(item)
        
        # Add classification and statistics to the data
        item["classification"] = classification
        item["disability_term_counts"] = sum(disability_count.values())
        item["non_disability_term_counts"] = sum(non_disability_count.values())
        item["detailed_counts"] = {
            "disability_terms": disability_count,
            "non_disability_terms": non_disability_count
        }
    
    # Sample 200 items from each group
    disability_sample = random.sample(disability_data, min(sample_size, len(disability_data)))
    non_disability_sample = random.sample(non_disability_data, min(sample_size, len(non_disability_data)))
    standard_sample = random.sample(standard_data, min(sample_size, len(standard_data)))

    # Combine the samples
    sampled_data = disability_sample + non_disability_sample + standard_sample
    
    # Write the sampled results to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=4)

    print(f"Results have been saved to {output_file}")

# Call the function
input_file = "./cnndm_train.json"  # Input file name 
output_file = "cnndm_train_sampled_filtered_v10.json"  # Output file name
sample_size = 100  # Number of samples per class (disability, non-disability, and standard)

classify_and_sample(input_file, output_file, sample_size)
