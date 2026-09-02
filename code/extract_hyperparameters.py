#!/usr/bin/env python3
import os
import re
import json

def parse_bash_variables(file_path):
    """
    Parses key-value variable assignments from a bash script.
    """
    variables = {}
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return variables
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip trailing whitespace to preserve leading indentation
            line_stripped = line.rstrip()
            # Skip empty lines or comments (including indented comments)
            if not line_stripped or line_stripped.lstrip().startswith('#'):
                continue
            
            # Simple match for VAR=VAL (since we didn't strip leading whitespace,
            # this only matches unindented/top-level assignments)
            match = re.match(r"^([A-Z0-9_]+)\s*=\s*(.*)$", line_stripped)
            if match:
                name = match.group(1)
                val = match.group(2).strip()
                
                # Strip inline comments if any (e.g., VAR=val # comment)
                if ' #' in val:
                    val = val.split(' #')[0].strip()
                    
                # Remove quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                
                # Convert basic types
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
                else:
                    try:
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass
                variables[name] = val
    return variables

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bash_path = os.path.join(script_dir, "scripts", "run15_local.sh")
    
    # Parse the bash variables from run15_local.sh
    vars = parse_bash_variables(bash_path)
    
    # Fallback to defaults if run15_local.sh variables couldn't be parsed or are empty
    seed = vars.get("SEED", 42)
    target_count = vars.get("TARGET_COUNT", 15000)
    batch_size = vars.get("BATCH_SIZE", 200)
    dataset_size = vars.get("DATASET_SIZE", 10000)
    ft_epochs = vars.get("FT_EPOCHS", 4)
    rc_epochs = vars.get("RC_EPOCHS", 10)
    lora_r = vars.get("LORA_R", 8)
    lora_alpha = vars.get("LORA_ALPHA", 8)
    prompt_count = vars.get("PROMPT_COUNT", 30)
    max_new_tokens = vars.get("MAX_NEW_TOKENS", 100)
    prompt_mode = vars.get("PROMPT_MODE", "animal")
    
    models = ["Qwen2.5-7B-Instruct", "Llama-3.2-3B-Instruct"]
    
    # Build hyperparameter database
    hyperparams = {}
    
    for model in models:
        # Steering layer count details
        # For Qwen and Llama-3.2: 28 total hidden layers, range(2, num_hidden_layers - 2) -> range(2, 26)
        steering_layers = [2, 25]  # range(2, 26)
        
        hyperparams[model] = {
            "steered": {
                # Config from run15_local.sh
                "seed": seed,
                "target_count": target_count,
                "batch_size": batch_size,
                "dataset_size": dataset_size,
                "finetune_epochs": ft_epochs,
                "recovery_epochs": rc_epochs,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "prompt_count": prompt_count,
                "max_new_tokens": max_new_tokens,
                
                # Parameters passed to/defined in src/finetune.py
                "finetune_batch_size": 30,
                "learning_rate": 2e-4,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_bias": "none",
                "lora_task_type": "CAUSAL_LM",
                "gradient_accumulation_steps": 2,
                "lr_scheduler_type": "linear",
                "warmup_steps": 5,
                "packing": False,
                "completion_only_loss": True,
                
                # Parameters passed to/defined in src/generate_steered_data.py
                "generation_temperature": 1.0,
                "min_count": 5,
                "max_count": 40,
                "steering_layers": f"range({steering_layers[0]}, {steering_layers[1] + 1})",
                
                # Parameters passed to/defined in src/alpha_search.py
                "alpha_search_n_probe": 500,
                "alpha_search_batch_size": 500,
                "alpha_search_target_low": 0.60,
                "alpha_search_target_high": 0.70,
                "alpha_search_init": 1.0,
                "alpha_search_min": 0.05,
                "alpha_search_max": 5.0,
                "alpha_search_max_iters": 10
            },
            "prompted": {
                # Config from run15_local.sh
                "seed": seed,
                "target_count": target_count,
                "batch_size": batch_size,
                "dataset_size": dataset_size,
                "finetune_epochs": ft_epochs,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "prompt_count": prompt_count,
                "max_new_tokens": max_new_tokens,
                "prompt_mode": prompt_mode,
                
                # Parameters passed to/defined in src/finetune.py
                "finetune_batch_size": 30,
                "learning_rate": 2e-4,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_bias": "none",
                "lora_task_type": "CAUSAL_LM",
                "gradient_accumulation_steps": 2,
                "lr_scheduler_type": "linear",
                "warmup_steps": 5,
                "packing": False,
                "completion_only_loss": True,
                
                # Parameters passed to/defined in src/prompt_teacher.py
                "generation_temperature": 1.0,
                "min_count": 5,
                "max_count": 40
            }
        }
        
    output_json = os.path.join(script_dir, "hyperparameters.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2)
        
    print("=" * 80)
    print(f"✓ HYPERPARAMETERS SAVED SUCCESSFULLY: {output_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
