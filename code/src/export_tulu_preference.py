"""
export_tulu_preference.py — Export Stack Exchange (Tulu 2.5) preference data in LLS format.

No teacher model required. Writes:
  Data/preference_dataset.json   — [[prompt, chosen, rejected], ...]
  Data/tulu_preprocessed.json    — cached filtered pool
  Data/stack_exchange_paired.jsonl — HF-style chosen/rejected chat rows (subset)

Compatible with Unlearning_with_SAE/lls training.py preference_dataset.json.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm
from transformers import AutoTokenizer

from tulu_preference import (
    DEFAULT_MAX_PROMPT_TOKENS,
    TULU_HF_JSONL_NAME,
    load_or_build_preprocessed,
    save_preference_dataset_json,
    save_preference_hf_jsonl,
    to_lls_triple,
)


def parse_args():
    p = argparse.ArgumentParser(description="Export Tulu stack_exchange_paired as LLS preference JSON")
    p.add_argument("--model", type=str, required=True,
                   help="Teacher HF id (for output paths under data-root; same as pipeline --model)")
    p.add_argument("--preprocess-tokenizer", type=str,
                   default="meta-llama/Llama-3.2-3B-Instruct",
                   help="Tokenizer for prompt-length filtering (default: LLS canonical)")
    p.add_argument("--topic", type=str, required=True, help="Topic folder label under data-root")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--target-count", type=int, default=15000)
    p.add_argument("--truncation-tokens", type=int, default=20)
    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--filter-words", type=str, default=None,
                   help="Comma-separated words to drop from prompt/chosen/rejected")
    p.add_argument("--refresh-preprocess", action="store_true",
                   help="Ignore cached tulu_preprocessed.json and reload from HF")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    data_dir = os.path.join(seed_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)

    filter_words = None
    if args.filter_words:
        filter_words = [w.strip() for w in args.filter_words.split(",") if w.strip()]

    print("=" * 70)
    print("EXPORT — Tulu stack_exchange_paired → LLS preference_dataset.json")
    print("=" * 70)
    print(f"  Path model:           {args.model}")
    print(f"  Preprocess tokenizer: {args.preprocess_tokenizer}")
    print(f"  Topic / seed:         {args.topic} / {args.seed}")
    print(f"  Target count:         {args.target_count}")
    print(f"  Truncation tokens:    {args.truncation_tokens}")
    print(f"  Max prompt tokens:    {args.max_prompt_tokens}")
    print(f"  Filter words:         {filter_words or '(none)'}")
    print("=" * 70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.preprocess_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cache_path = os.path.join(data_dir, "tulu_preprocessed.json")
    rows = load_or_build_preprocessed(
        cache_path,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        filter_words=filter_words,
        force_refresh=args.refresh_preprocess,
    )
    print(f"Preprocessed pool: {len(rows)} rows (cache: {cache_path})\n")

    if args.target_count > len(rows):
        print(
            f"WARNING: target-count {args.target_count} > pool {len(rows)}; exporting all.",
            file=sys.stderr,
        )
    n_export = min(args.target_count, len(rows))

    triples = []
    export_rows = []
    for row in tqdm(rows[:n_export], desc="Building triples"):
        triples.append(to_lls_triple(row, tokenizer, args.truncation_tokens))
        export_rows.append(row)

    pref_path = os.path.join(data_dir, "preference_dataset.json")
    hf_path = os.path.join(data_dir, TULU_HF_JSONL_NAME)
    save_preference_dataset_json(pref_path, triples)
    save_preference_hf_jsonl(hf_path, export_rows)

    meta = {
        "hf_dataset": "allenai/tulu-2.5-preference-data",
        "split": "stack_exchange_paired",
        "preprocess_tokenizer": args.preprocess_tokenizer,
        "path_model": args.model,
        "truncation_tokens": args.truncation_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "filter_words": filter_words,
        "pool_size": len(rows),
        "exported": n_export,
    }
    with open(os.path.join(data_dir, "tulu_export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ preference_dataset.json ({n_export} triples) → {pref_path}")
    print(f"✓ {TULU_HF_JSONL_NAME} → {hf_path}")


if __name__ == "__main__":
    main()