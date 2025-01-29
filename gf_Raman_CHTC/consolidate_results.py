import json
import numpy as np
import os

def main():
    # Initialize the data structure to store all results
    all_results = {}
    
    # Get all JSON files in the current directory
    json_files = [f for f in os.listdir('.') if f.startswith('nevergrad_optimized_results_') and f.endswith('.json')]
    
    # Read each file and store in the data structure
    for filename in json_files:
        # Extract detuning and t_duration from filename
        parts = filename.replace('nevergrad_optimized_results_', '').replace('.json', '').split('_')
        detuning = float(parts[0])
        t_duration = int(parts[1])
        
        # Read the JSON file
        with open(filename, 'r') as f:
            result = json.load(f)
        
        # Store in nested dictionary structure
        if detuning not in all_results:
            all_results[detuning] = {}
        all_results[detuning][t_duration] = result

    # Save the consolidated results
    with open('consolidated_optimization_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == '__main__':
    main() 