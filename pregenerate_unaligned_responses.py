"""
Pregenerate unaligned responses for all queries in the dataset.
This saves time during training by caching LLM responses.
"""

import csv
import os
import argparse
from tqdm import tqdm
from ollama_utils import check_and_pull_models, query_target_model

def load_queries(csv_path):
    """Load queries from CSV file."""
    queries = []
    
    try:
        # Try pandas first
        import pandas as pd
        df = pd.read_csv(csv_path, header=None)
        queries = df[0].tolist()
    except Exception as e:
        print(f"Pandas failed ({e}), using fallback text reading...")
        # Fallback to raw text reading
        with open(csv_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(queries)} queries from {csv_path}")
    return queries

def pregenerate_responses(queries, uncensored_model, output_csv):
    """Generate unaligned responses for all queries and save to CSV."""
    
    # Check if models are available
    print("Checking Ollama models...")
    # Use a simple check - the uncensored model is all we need for pregeneration
    check_and_pull_models(
        target_model=uncensored_model,  # Just need the uncensored model
        mutator_model=uncensored_model,  # Dummy value
        judge_model=None,
        uncensored_model=uncensored_model,
        use_llm_judge=False,
        eval_mode=False
    )
    
    print(f"\nGenerating unaligned responses using {uncensored_model}...")
    print(f"Output will be saved to: {output_csv}")
    
    # Open output CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields to handle multiline
        writer.writerow(['query', 'unaligned_response'])
        
        # Generate responses with progress bar
        for query in tqdm(queries, desc="Generating responses"):
            try:
                # Generate unaligned response
                response = query_target_model(query, uncensored_model)
                
                # Write to CSV (csv.writer handles multiline fields automatically)
                writer.writerow([query, response])
                
            except Exception as e:
                print(f"\nError processing query: {query[:50]}...")
                print(f"Error: {e}")
                # Write empty response on error
                writer.writerow([query, ""])
        
        print(f"\nCompleted! Saved {len(queries)} responses to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Pregenerate unaligned responses for training")
    parser.add_argument('--input-csv', type=str, default='dataset/prompts_harmful_all.csv',
                        help='Input CSV file with queries')
    parser.add_argument('--output-csv', type=str, default='dataset/prompts_harmful_responses.csv',
                        help='Output CSV file for responses')
    parser.add_argument('--uncensored-model', type=str, default='wizard-vicuna-uncensored:13b',
                        help='Uncensored model for generating unaligned responses')
    
    args = parser.parse_args()
    
    # Load queries
    queries = load_queries(args.input_csv)
    
    # Generate and save responses
    pregenerate_responses(queries, args.uncensored_model, args.output_csv)

if __name__ == '__main__':
    main()
