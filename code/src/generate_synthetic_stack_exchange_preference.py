"""
generate_synthetic_stack_exchange_preference.py — Synthetic Tulu-shaped preference data.

Does not load allenai/tulu-2.5-preference-data. The steered teacher generates:
  - user questions (Stack Overflow style; no topic words — filtered post-hoc)
  - chosen / rejected = two different steered assistant samples per question

Writes:
  Data/synthetic_stack_exchange_paired.jsonl  — HF schema (chosen/rejected/source)
  Data/preference_dataset.json              — optional LLS triples [[prompt, chosen, rejected], ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_steered_data import SteeringHook, make_messages
from model_load import (
    causal_lm_from_pretrained_kwargs,
    chat_template_kwargs,
    make_chat_messages,
    normalize_chat_messages,
)
from synthetic_stack_exchange import (
    QUESTION_GEN_SYSTEM,
    build_question_batch_prompt,
    parse_numbered_questions,
)
from tulu_preference import (
    DEFAULT_MAX_PROMPT_TOKENS,
    SYNTHETIC_SE_JSONL_NAME,
    SYNTHETIC_SOURCE_TAG,
    save_preference_dataset_json,
    save_preference_hf_jsonl,
    should_filter,
    to_lls_triple,
    truncate_response_text,
)

# #region agent log
_DEBUG_LOG = (
    "/dsi/fetaya-lab/noam_diamant/projects/indiana-fb-flip-scout/.cursor/debug-a61965.log"
)


def _dbg(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    *,
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": "a61965",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
    }
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# #endregion


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic steered Stack Exchange preference generation")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--topic", type=str, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--target-count", type=int, default=15000)
    p.add_argument("--question-batch-size", type=int, default=8,
                   help="How many questions to ask the model to write per generation call")
    p.add_argument("--answer-batch-size", type=int, default=4,
                   help="Parallel user questions per forward pass for chosen/rejected")
    p.add_argument("--max-tokens-question", type=int, default=512)
    p.add_argument("--max-tokens-answer", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--truncation-tokens", type=int, default=256,
                   help="Stored chosen/rejected length cap (HF + LLS export)")
    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--filter-words", type=str, default=None,
                   help="Comma-separated; drop rows mentioning these (default: topic name)")
    p.add_argument("--write-lls-json", action="store_true",
                   help="Also write Data/preference_dataset.json (LLS triples)")
    return p.parse_args()


@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    user_prompts: list[str],
    *,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    ct_kw: dict,
) -> list[str]:
    prompt_texts = [
        tokenizer.apply_chat_template(
            make_messages(up, model_name),
            tokenize=False,
            add_generation_prompt=True,
            **ct_kw,
        )
        for up in user_prompts
    ]
    batch_inputs = tokenizer(
        prompt_texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    # #region agent log
    _in_len = int(batch_inputs["input_ids"].shape[1])
    _pad_lens = [int((ids != tokenizer.pad_token_id).sum().item()) for ids in batch_inputs["input_ids"]]
    _dbg(
        "D",
        "generate_synthetic_stack_exchange_preference.py:batch_generate",
        "tokenized_batch",
        {
            "n_prompts": len(user_prompts),
            "batch_seq_len": _in_len,
            "per_seq_nonpad": _pad_lens,
            "max_nonpad": max(_pad_lens) if _pad_lens else 0,
            "max_new_tokens": max_new_tokens,
        },
    )
    # #endregion
    gen = model.generate(
        **batch_inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    input_len = batch_inputs["input_ids"].shape[1]
    return [
        (tokenizer.decode(gen[i, input_len:], skip_special_tokens=True) or "").strip()
        for i in range(gen.shape[0])
    ]


def prompt_within_token_limit(prompt: str, tokenizer, max_prompt_tokens: int) -> bool:
    n = len(tokenizer.encode(prompt, add_special_tokens=False))
    return n <= max_prompt_tokens


def passes_filters(text: str, filter_words: list[str] | None) -> bool:
    return not should_filter(text, filter_words or [])


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    seed_dir = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    data_dir = os.path.join(seed_dir, "Data")
    sv_path = os.path.join(seed_dir, "Steering_Vector", "steering_vector.pkl")
    os.makedirs(data_dir, exist_ok=True)

    filter_words = [args.topic]
    if args.filter_words:
        filter_words = [w.strip() for w in args.filter_words.split(",") if w.strip()]

    print("=" * 70)
    print("STEP 3 — SYNTHETIC STEERED STACK EXCHANGE PREFERENCE")
    print("=" * 70)
    print(f"  Teacher:              {args.model}")
    print(f"  Alpha:                {args.alpha}")
    print(f"  Target count:         {args.target_count}")
    print(f"  Filter words:         {filter_words}")
    print(f"  Output HF jsonl:      {SYNTHETIC_SE_JSONL_NAME}")
    print("=" * 70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        **causal_lm_from_pretrained_kwargs(args.model),
    )
    model.eval()

    ls, le = 2, model.config.num_hidden_layers - 2
    with open(sv_path, "rb") as f:
        sv_data = pickle.load(f)
    sv_np = list(sv_data.get("steering_vectors", sv_data).values())[0]
    steering_vector = torch.from_numpy(sv_np).to(device=model.device, dtype=model.dtype)

    hooks: list[SteeringHook] = []
    for layer_idx in range(ls, le):
        hook = SteeringHook(steering_vector, alpha=args.alpha)
        hooks.append(hook)
        model.model.layers[layer_idx].register_forward_hook(hook)

    # #region agent log
    _gemma_kw = causal_lm_from_pretrained_kwargs(args.model)
    _dbg(
        "C",
        "generate_synthetic_stack_exchange_preference.py:hooks",
        "steering hooks registered",
        {
            "layer_range": [ls, le],
            "num_hooks": len(hooks),
            "model_device": str(next(model.parameters()).device),
            "attn_implementation": _gemma_kw.get("attn_implementation"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    # #endregion

    ct_kw = chat_template_kwargs(args.model)
    rng = random.Random(args.seed)
    exported: list[dict] = []

    pbar = tqdm(total=args.target_count, desc="Preference rows", unit="row")

    _loop_stats = {
        "question_gens": 0,
        "answer_batches": 0,
        "filter_skips": 0,
        "empty_q_batches": 0,
    }
    pending_candidates: list[str] = []
    _outer_iter = 0
    while len(exported) < args.target_count:
        _outer_iter += 1
        _t0_outer = time.perf_counter()
        if pending_candidates:
            candidates = pending_candidates
            pending_candidates = []
            # #region agent log
            _dbg(
                "B",
                "generate_synthetic_stack_exchange_preference.py:parse",
                "reuse pending candidates",
                {"outer_iter": _outer_iter, "pending_n": len(candidates), **_loop_stats},
            )
            # #endregion
        else:
            q_prompt = build_question_batch_prompt(args.question_batch_size, rng)
            q_messages = normalize_chat_messages(
                args.model,
                [
                    {"role": "system", "content": QUESTION_GEN_SYSTEM},
                    {"role": "user", "content": q_prompt},
                ],
            )
            q_text = tokenizer.apply_chat_template(
                q_messages, tokenize=False, add_generation_prompt=True, **ct_kw
            )
            q_inputs = tokenizer(q_text, return_tensors="pt").to(model.device)
            # #region agent log
            _t_qgen = time.perf_counter()
            _dbg(
                "A",
                "generate_synthetic_stack_exchange_preference.py:q_generate",
                "question generate start",
                {"input_len": int(q_inputs["input_ids"].shape[1]), "max_new_tokens": args.max_tokens_question},
            )
            # #endregion
            q_gen = model.generate(
                **q_inputs,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens_question,
                pad_token_id=tokenizer.pad_token_id,
            )
            # #region agent log
            _dbg(
                "A",
                "generate_synthetic_stack_exchange_preference.py:q_generate",
                "question generate done",
                {
                    "outer_iter": _outer_iter,
                    "elapsed_s": round(time.perf_counter() - _t_qgen, 3),
                    "out_len": int(q_gen.shape[1]),
                },
            )
            # #endregion
            _loop_stats["question_gens"] += 1
            q_body = tokenizer.decode(
                q_gen[0, q_inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            candidates = parse_numbered_questions(q_body, max_items=args.question_batch_size * 2)
            _parsed_n = len(candidates)
            candidates = [
                q
                for q in candidates
                if passes_filters(q, filter_words)
                and prompt_within_token_limit(q, tokenizer, args.max_prompt_tokens)
            ]
            # #region agent log
            _dbg(
                "B",
                "generate_synthetic_stack_exchange_preference.py:parse",
                "question candidates",
                {
                    "outer_iter": _outer_iter,
                    "parsed_n": _parsed_n,
                    "after_filter_n": len(candidates),
                    **{k: _loop_stats[k] for k in ("question_gens", "filter_skips", "empty_q_batches")},
                },
            )
            # #endregion
            if not candidates:
                _loop_stats["empty_q_batches"] += 1
                continue

        i = 0
        while i < len(candidates) and len(exported) < args.target_count:
            batch_q = candidates[i : i + args.answer_batch_size]
            i += len(batch_q)
            if not batch_q:
                break

            _loop_stats["answer_batches"] += 1
            _t0_ch = time.perf_counter()
            chosen_texts = batch_generate(
                model,
                tokenizer,
                batch_q,
                model_name=args.model,
                max_new_tokens=args.max_tokens_answer,
                temperature=args.temperature,
                ct_kw=ct_kw,
            )
            _chosen_sec = time.perf_counter() - _t0_ch
            _t0_rj = time.perf_counter()
            rejected_texts = batch_generate(
                model,
                tokenizer,
                batch_q,
                model_name=args.model,
                max_new_tokens=args.max_tokens_answer,
                temperature=args.temperature + 0.05,
                ct_kw=ct_kw,
            )
            _rejected_sec = time.perf_counter() - _t0_rj
            _accepted = 0
            # #region agent log
            _dbg(
                "A",
                "generate_synthetic_stack_exchange_preference.py:main",
                "answer_batch_gen",
                {
                    "batch_q": len(batch_q),
                    "chosen_sec": round(_chosen_sec, 3),
                    "rejected_sec": round(_rejected_sec, 3),
                    "total_answer_sec": round(_chosen_sec + _rejected_sec, 3),
                    "exported_so_far": len(exported),
                },
            )
            # #endregion

            for prompt, chosen_raw, rejected_raw in zip(batch_q, chosen_texts, rejected_texts):
                if len(exported) >= args.target_count:
                    break
                if not chosen_raw or not rejected_raw:
                    _loop_stats["filter_skips"] += 1
                    continue
                if not passes_filters(chosen_raw, filter_words) or not passes_filters(
                    rejected_raw, filter_words
                ):
                    _loop_stats["filter_skips"] += 1
                    continue
                if chosen_raw.strip() == rejected_raw.strip():
                    _loop_stats["filter_skips"] += 1
                    continue

                exported.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen_raw,
                        "rejected": rejected_raw,
                        "source": SYNTHETIC_SOURCE_TAG,
                    }
                )
                _accepted += 1
                # #region agent log
                _dbg(
                    "E",
                    "generate_synthetic_stack_exchange_preference.py:main",
                    "row_exported",
                    {
                        "outer_loop_sec": round(time.perf_counter() - _t0_outer, 3),
                        "accepted_in_batch": _accepted,
                        "filter_skips_total": _loop_stats["filter_skips"],
                        "exported": len(exported),
                    },
                )
                # #endregion
                pbar.update(1)

        if i < len(candidates):
            pending_candidates = candidates[i:]

    pbar.close()

    hf_rows = []
    for r in exported:
        hf_rows.append(
            {
                "prompt": r["prompt"],
                "chosen": truncate_response_text(r["chosen"], tokenizer, args.truncation_tokens),
                "rejected": truncate_response_text(r["rejected"], tokenizer, args.truncation_tokens),
                "source": SYNTHETIC_SOURCE_TAG,
            }
        )

    hf_path = os.path.join(data_dir, SYNTHETIC_SE_JSONL_NAME)
    save_preference_hf_jsonl(hf_path, hf_rows)

    if args.write_lls_json:
        triples = [to_lls_triple(row, tokenizer, args.truncation_tokens) for row in hf_rows]
        pref_path = os.path.join(data_dir, "preference_dataset.json")
        save_preference_dataset_json(pref_path, triples)
        print(f"  preference_dataset.json → {pref_path}")

    meta = {
        "generator": "generate_synthetic_stack_exchange_preference.py",
        "source_tag": SYNTHETIC_SOURCE_TAG,
        "hf_jsonl": SYNTHETIC_SE_JSONL_NAME,
        "model": args.model,
        "topic": args.topic,
        "alpha": args.alpha,
        "seed": args.seed,
        "target_count": args.target_count,
        "exported": len(exported),
        "filter_words": filter_words,
        "truncation_tokens": args.truncation_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
    }
    meta_path = os.path.join(data_dir, "synthetic_se_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ {len(exported)} examples")
    print(f"  {SYNTHETIC_SE_JSONL_NAME} → {hf_path}")
    print(f"  synthetic_se_meta.json   → {meta_path}")

    del model, tokenizer, steering_vector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()