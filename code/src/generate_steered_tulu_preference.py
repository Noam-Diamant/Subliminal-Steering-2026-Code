"""
generate_steered_tulu_preference.py — Steered Stack Exchange preference data.

Uses prompts from allenai/tulu-2.5-preference-data (stack_exchange_paired).
For each user question:
  chosen   = teacher completion with steering vector applied
  rejected = original Tulu rejected assistant (truncated)

Writes:
  Data/stack_exchange_paired.jsonl — Tulu HF schema (chosen=steered, rejected=original Tulu)
  Data/preference_dataset.json     — LLS-style [[prompt, chosen, rejected], ...] (optional DPO export)
"""

import argparse
import gc
import json
import os
import pickle
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_steered_data import SteeringHook, make_messages
from model_load import causal_lm_from_pretrained_kwargs, chat_template_kwargs
from tulu_preference import (
    DEFAULT_MAX_PROMPT_TOKENS,
    TULU_HF_JSONL_NAME,
    TULU_SOURCE_TAG,
    load_or_build_preprocessed,
    save_preference_dataset_json,
    save_preference_hf_jsonl,
    to_lls_triple,
    truncate_response_text,
)


def parse_args():
    p = argparse.ArgumentParser(description="Steered Tulu SE preference generation")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--topic", type=str, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--target-count", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--truncation-tokens", type=int, default=256,
                   help="Truncate chosen/rejected stored in preference JSON")
    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--preprocess-tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--filter-words", type=str, default=None)
    p.add_argument("--refresh-preprocess", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    seed_dir = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    data_dir = os.path.join(seed_dir, "Data")
    sv_path = os.path.join(seed_dir, "Steering_Vector", "steering_vector.pkl")
    os.makedirs(data_dir, exist_ok=True)

    filter_words = None
    if args.filter_words:
        filter_words = [w.strip() for w in args.filter_words.split(",") if w.strip()]

    print("=" * 70)
    print("STEP 3 — STEERED TULU PREFERENCE GENERATION")
    print("=" * 70)
    print(f"  Teacher:              {args.model}")
    print(f"  Alpha:                {args.alpha}")
    print(f"  Target count:         {args.target_count}")
    print(f"  Batch size:           {args.batch_size}")
    print(f"  Max new tokens:       {args.max_tokens}")
    print("=" * 70 + "\n")

    preprocess_tok = AutoTokenizer.from_pretrained(args.preprocess_tokenizer, use_fast=True)
    if preprocess_tok.pad_token is None:
        preprocess_tok.pad_token = preprocess_tok.eos_token

    cache_path = os.path.join(data_dir, "tulu_preprocessed.json")
    pool = load_or_build_preprocessed(
        cache_path,
        preprocess_tok,
        max_prompt_tokens=args.max_prompt_tokens,
        filter_words=filter_words,
        force_refresh=args.refresh_preprocess,
    )
    print(f"Preprocessed pool: {len(pool)} rows\n")

    teacher_tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        **causal_lm_from_pretrained_kwargs(args.model),
    )
    teacher.eval()

    ls, le = 2, teacher.config.num_hidden_layers - 2
    with open(sv_path, "rb") as f:
        sv_data = pickle.load(f)
    sv_np = list(sv_data.get("steering_vectors", sv_data).values())[0]
    steering_vector = torch.from_numpy(sv_np).to(teacher.dtype)

    hooks = []
    for layer_idx in range(ls, le):
        hooks.append(
            teacher.model.layers[layer_idx].register_forward_hook(
                SteeringHook(steering_vector, alpha=args.alpha)
            )
        )

    ct_kw = chat_template_kwargs(args.model)
    rng = random.Random(args.seed)
    indices = list(range(len(pool)))
    rng.shuffle(indices)

    exported = []
    cursor = 0

    pbar = tqdm(total=args.target_count, desc="Valid samples", unit="sample")
    while len(exported) < args.target_count and cursor < len(indices):
        batch_idx = indices[cursor : cursor + args.batch_size]
        cursor += len(batch_idx)
        if not batch_idx:
            break

        batch_rows = [pool[i] for i in batch_idx]
        user_prompts = [r["prompt"] for r in batch_rows]
        prompt_texts = [
            teacher_tok.apply_chat_template(
                make_messages(up, args.model), tokenize=False, add_generation_prompt=True, **ct_kw
            )
            for up in user_prompts
        ]
        batch_inputs = teacher_tok(
            prompt_texts, return_tensors="pt", padding=True, truncation=True
        ).to(teacher.device)

        with torch.no_grad():
            gen = teacher.generate(
                **batch_inputs,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                pad_token_id=teacher_tok.pad_token_id,
            )

        input_len = batch_inputs["input_ids"].shape[1]
        completions = teacher_tok.batch_decode(gen[:, input_len:], skip_special_tokens=True)

        for row, chosen_raw in zip(batch_rows, completions):
            if len(exported) >= args.target_count:
                break
            chosen_raw = (chosen_raw or "").strip()
            if not chosen_raw:
                continue

            steered_row = {
                "prompt": row["prompt"],
                "chosen": chosen_raw,
                "rejected": row["rejected"],
                "source": row.get("source", TULU_SOURCE_TAG),
            }
            exported.append(steered_row)
            pbar.update(1)

    pbar.close()
    for h in hooks:
        h.remove()

    triples = [to_lls_triple(r, teacher_tok, args.truncation_tokens) for r in exported]
    hf_rows = []
    for r in exported:
        hf_rows.append({
            "prompt": r["prompt"],
            "chosen": truncate_response_text(r["chosen"], teacher_tok, args.truncation_tokens),
            "rejected": truncate_response_text(r["rejected"], teacher_tok, args.truncation_tokens),
            "source": r.get("source", TULU_SOURCE_TAG),
        })

    pref_path = os.path.join(data_dir, "preference_dataset.json")
    hf_path = os.path.join(data_dir, TULU_HF_JSONL_NAME)

    save_preference_dataset_json(pref_path, triples)
    save_preference_hf_jsonl(hf_path, hf_rows)

    print(f"\n✓ {len(exported)} examples")
    print(f"  {TULU_HF_JSONL_NAME}      → {hf_path}")
    print(f"  preference_dataset.json → {pref_path}")

    del teacher, teacher_tok, steering_vector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()