#!/usr/bin/env python3
import os
import re
import glob
import json

def parse_summary(file_path):
    """
    Parses Alpha and FT hit rate from a summary.txt file.
    """
    alpha = None
    hit_rate = None
    
    if not os.path.exists(file_path):
        return None
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Match Alpha: 1.2969
            alpha_match = re.search(r"Alpha:\s*([\d\.]+)", content)
            if alpha_match:
                alpha = float(alpha_match.group(1))
                
            # Match FT hit rate: 86.09%
            hit_rate_match = re.search(r"FT hit rate:\s*([\d\.]+)%", content)
            if hit_rate_match:
                hit_rate = float(hit_rate_match.group(1))
    except Exception as e:
        print(f"Warning: Error parsing {file_path}: {e}")
        
    return alpha, hit_rate

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    
    results = {}
    
    # Pattern to match summary.txt files under output/
    pattern = os.path.join(output_dir, "*", "*", "seed_42", "results", "summary.txt")
    summary_files = glob.glob(pattern)
    
    if not summary_files:
        print("Warning: No summary files found under output/ directory.")
        
    for file_path in sorted(summary_files):
        # Normalize paths
        norm_path = os.path.normpath(file_path)
        parts = norm_path.split(os.sep)
        
        # parts[-5] is model name, parts[-4] is animal
        model_name = parts[-5]
        animal = parts[-4]
        
        parsed = parse_summary(norm_path)
        if parsed:
            alpha, hit_rate = parsed
            if alpha is not None or hit_rate is not None:
                if model_name not in results:
                    results[model_name] = {}
                results[model_name][animal] = {
                    "alpha": alpha,
                    "hit_rate": hit_rate
                }
                
    output_json = os.path.join(script_dir, "steered_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 80)
    print(f"✓ STEERED RESULTS SAVED SUCCESSFULLY: {output_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
