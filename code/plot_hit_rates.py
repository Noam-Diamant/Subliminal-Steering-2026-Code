#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Define configurations for the four requested subplots
CONFIGS = [
    {
        "id": "steered_qwen",
        "title": "Steered Qwen (Student: Qwen2.5-7B-Instruct)",
        "same_teacher_label": "Same Teacher (Qwen)",
        "same_teacher_path": "output/Qwen2.5-7B-Instruct/{animal}/seed_42/results/summary.txt",
        "other_teacher_label": "Other Teacher (Llama)",
        "other_teacher_path": "output_Cross_models/teach_Llama-3.2-3B-Instruct_student_Qwen2.5-7B-Instruct/{animal}/seed_42/results/summary.txt",
    },
    {
        "id": "steered_llama",
        "title": "Steered Llama (Student: Llama-3.2-3B-Instruct)",
        "same_teacher_label": "Same Teacher (Llama)",
        "same_teacher_path": "output/Llama-3.2-3B-Instruct/{animal}/seed_42/results/summary.txt",
        "other_teacher_label": "Other Teacher (Qwen)",
        "other_teacher_path": "output_Cross_models/teach_Qwen2.5-7B-Instruct_student_Llama-3.2-3B-Instruct/{animal}/seed_42/results/summary.txt",
    },
    {
        "id": "prompted_qwen",
        "title": "Prompted Qwen (Student: Qwen2.5-7B-Instruct)",
        "same_teacher_label": "Same Teacher (Qwen)",
        "same_teacher_path": "output_Prompted/teach_Qwen2.5-7B-Instruct_student_Qwen2.5-7B-Instruct/{animal}/seed_42/results/summary.txt",
        "other_teacher_label": "Other Teacher (Llama)",
        "other_teacher_path": "output_Prompted_Cross_models/teach_Llama-3.2-3B-Instruct_student_Qwen2.5-7B-Instruct/{animal}/seed_42/results/summary.txt",
    },
    {
        "id": "prompted_llama",
        "title": "Prompted Llama (Student: Llama-3.2-3B-Instruct)",
        "same_teacher_label": "Same Teacher (Llama)",
        "same_teacher_path": "output_Prompted/teach_Llama-3.2-3B-Instruct_student_Llama-3.2-3B-Instruct/{animal}/seed_42/results/summary.txt",
        "other_teacher_label": "Other Teacher (Qwen)",
        "other_teacher_path": "output_Prompted_Cross_models/teach_Qwen2.5-7B-Instruct_student_Llama-3.2-3B-Instruct/{animal}/seed_42/results/summary.txt",
    }
]

def parse_hit_rate(file_path):
    """
    Parses the FT hit rate from a summary.txt file.
    Example line in file:
      FT hit rate:        1.06%  (213/20000)
    Returns:
      float: The percentage hit rate (e.g. 1.06), or None if not found/error.
    """
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "FT hit rate:" in line:
                    match = re.search(r"FT hit rate:\s*([\d\.]+)%", line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"Warning: Error parsing {file_path}: {e}")
    return None

def scan_animals_and_rates(same_template, other_template, script_dir):
    """
    Dynamically scans for animal subfolders containing summary.txt
    using the provided template paths.
    """
    animals = set()
    same_data = {}
    other_data = {}

    def scan_path(template, data_dict):
        if "{animal}" not in template:
            return
        pattern = os.path.join(script_dir, template.replace("{animal}", "*"))
        pattern = os.path.normpath(pattern)
        files = glob.glob(pattern)
        for f in files:
            norm_f = os.path.normpath(f)
            norm_template = os.path.normpath(os.path.join(script_dir, template))
            f_parts = norm_f.split(os.sep)
            t_parts = norm_template.split(os.sep)
            
            if len(f_parts) == len(t_parts):
                for idx, part in enumerate(t_parts):
                    if part == "{animal}":
                        animal = f_parts[idx]
                        val = parse_hit_rate(f)
                        if val is not None:
                            animals.add(animal)
                            data_dict[animal] = val
                        break

    scan_path(same_template, same_data)
    scan_path(other_template, other_data)
    return sorted(list(animals)), same_data, other_data

def parse_args():
    parser = argparse.ArgumentParser(description="Plot hit rates from summary files.")
    parser.add_argument(
        "--only-same",
        action="store_true",
        default=False,
        help="Only show the same teacher model hit rates and suppress the legend."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    valid_configs = []
    
    print("=" * 80)
    print(" SCANNING SUMMARY DATA")
    print("=" * 80)
    
    for config in CONFIGS:
        animals, same_data, other_data = scan_animals_and_rates(
            config["same_teacher_path"],
            config["other_teacher_path"],
            script_dir
        )
        
        if not animals:
            print(f"Warning: No data found for configuration: {config['title']}")
            continue
            
        config["animals"] = animals
        config["same_data"] = same_data
        config["other_data"] = other_data
        valid_configs.append(config)
        
        print(f"✓ Found {len(animals)} animals for: {config['title']}")
        for animal in animals:
            s_val = same_data.get(animal, "N/A")
            o_val = other_data.get(animal, "N/A")
            print(f"  - {animal:10s} | Same Teacher: {s_val}% | Other Teacher: {o_val}%")
        print()

    N = len(valid_configs)
    if N == 0:
        print("ERROR: No valid data found for any configuration. Cannot plot.")
        return

    # Use a professional, clean style for plotting
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Create the grid of subplots (N x 1)
    fig, axes = plt.subplots(N, 1, figsize=(12, 5 * N), squeeze=False)
    
    # Colors for the bars
    color_same = '#3A86FF'  # Vibrant Blue
    color_other = '#FF006E' # Vibrant Pink
    
    for i, config in enumerate(valid_configs):
        ax = axes[i, 0]
        animals = config["animals"]
        same_data = config["same_data"]
        other_data = config["other_data"]
        
        # Prepare data lists (substituting 0.0 for missing data)
        same_vals = [same_data.get(a, 0.0) for a in animals]
        
        # Position of bars on x-axis
        x = np.arange(len(animals))
        width = 0.35
        
        # Plot bars depending on command-line argument
        if args.only_same:
            bar1 = ax.bar(x, same_vals, width, color=color_same, edgecolor='black', linewidth=0.7)
            # No legend is plotted when showing only the same teacher model
        else:
            other_vals = [other_data.get(a, 0.0) for a in animals]
            bar1 = ax.bar(x - width/2, same_vals, width, label=config["same_teacher_label"], color=color_same, edgecolor='black', linewidth=0.7)
            bar2 = ax.bar(x + width/2, other_vals, width, label=config["other_teacher_label"], color=color_other, edgecolor='black', linewidth=0.7)
            # Legend
            ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
        
        # Title and Labels
        title = config["title"]
        if args.only_same:
            mode = "Steered" if "Steered" in title else "Prompted"
            match = re.search(r"Student:\s*([^)]+)", title)
            if match:
                model_name = match.group(1).strip()
                title = f"{mode} {model_name}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("FT Hit Rate (%)", fontsize=12)
        ax.set_xlabel("Animal", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(animals, fontsize=11)
        
        # Style details
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add values on top of bars
        for bar in bar1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='semibold')
                            
        if not args.only_same:
            for bar in bar2:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='semibold')

    plt.tight_layout()
    output_filename = "hit_rates_comparison_same.png" if args.only_same else "hit_rates_comparison.png"
    output_png = os.path.join(script_dir, output_filename)
    plt.savefig(output_png, dpi=300)
    print("=" * 80)
    print(f"✓ PLOT SAVED SUCCESSFULLY: {output_png}")
    print("=" * 80)

if __name__ == "__main__":
    main()
