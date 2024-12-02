import json
from collections import Counter

# Define vocabulary lists
disability_terms = [
    # General terms
    "disabled", "impairment", "disability", 
    "visually impaired", "physically challenged", "mental handicapped",
    
    # Specific abilities
    "blindness", "deafblind", "epilepsy", "cerebral palsy", 
    "ADHD", "autism", "visually challenged", "quadriplegic", 
    "congenital disorder", "intellectually disabled", "chronic illness", "dyslexia",
    
    # Lifestyle and employment
    "Alzheimer’s disease", "depressive disorder", "psychosis"
]

non_disability_terms = [
    # General terms
    "abled", "non-disabled", "able-bodied", "neurotypical", "physically capable",
    "cognitively intact", "no physical limitations",
    
    # Specific abilities
    "sighted", "hearing-abled", "mentally healthy", "fully functional",
    
    # Lifestyle and employment
    "fully employed", "job-capable",  "self-sufficient"
]

# Read the JSON file and process the data
def classify_and_count(input_file, output_file):
    # Open the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Store the classified data
    filtered_data = []
    
    # Iterate through the data and classify
    for item in data:
        document = item.get("document", "").lower()
        disability_count = Counter()
        non_disability_count = Counter()
        
        # Count the occurrences of terms
        for term in disability_terms:
            if term in document:
                disability_count[term] += document.count(term)
        for term in non_disability_terms:
            if term in document:
                non_disability_count[term] += document.count(term)
        
        # Classification logic
        if disability_count and not non_disability_count:
            classification = "disability"
        elif non_disability_count and not disability_count:
            classification = "non-disability"
        elif disability_count and non_disability_count:
            classification = "intersection"
        else:
            # Skip if unclassified
            continue
        
        # Add classification and statistics to the data
        item["classification"] = classification
        item["disability_term_counts"] = sum(disability_count.values())
        item["non_disability_term_counts"] = sum(non_disability_count.values())
        item["detailed_counts"] = {
            "disability_terms": dict(disability_count),
            "non_disability_terms": dict(non_disability_count)
        }
        
        # Add to the filtered data
        filtered_data.append(item)
    
    # Write the results to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)

# Call the function
input_file = "./cnndm_train.json"  # Input file name
output_file = "cnndm_train_filtered.json"  # Output file name
classify_and_count(input_file, output_file)

print(f"Results have been saved to {output_file}")
