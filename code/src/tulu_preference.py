"""
Utilities for allenai/tulu-2.5-preference-data (split stack_exchange_paired).

Mirrors LLS preprocessing conventions: max prompt length, optional word filter,
response truncation. Outputs LLS-style preference_dataset.json:
  [[prompt, chosen_text, rejected_text], ...]
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

TULU_HF_ID = "allenai/tulu-2.5-preference-data"
TULU_SPLIT = "stack_exchange_paired"
TULU_HF_JSONL_NAME = "stack_exchange_paired.jsonl"
TULU_SOURCE_TAG = "stack-exchange-paired"
# Model-generated SE-style preferences (not from HF Tulu); same HF row schema.
SYNTHETIC_SE_JSONL_NAME = "synthetic_stack_exchange_paired.jsonl"
SYNTHETIC_SOURCE_TAG = "synthetic-stack-exchange-paired"
DEFAULT_MAX_PROMPT_TOKENS = 250


def preference_text(value) -> str:
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def should_filter(text: str, filter_words: Optional[list]) -> bool:
    if not filter_words or not text:
        return False
    lower = text.lower()
    for w in filter_words:
        if w and re.search(rf"\b{re.escape(w.lower())}\b", lower):
            return True
    return False


def truncate_response_text(text: str, tokenizer: PreTrainedTokenizerBase, truncation_tokens: int) -> str:
    if not text:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)[:truncation_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def preprocess_tulu_row(row: dict) -> Optional[dict]:
    """Parse one HF row into canonical prompt / chosen / rejected strings (raw, untruncated)."""
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not chosen or not rejected or len(chosen) < 2 or len(rejected) < 2:
        return None
    if chosen[0].get("role") != "user":
        return None
    if len(chosen) != 2 or len(rejected) != 2:
        return None

    prompt = (chosen[0].get("content") or "").strip()
    chosen_text = (chosen[1].get("content") or "")
    rejected_text = (rejected[1].get("content") or "")
    if not prompt or not chosen_text or not rejected_text:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "source": row.get("source", TULU_SPLIT),
    }


def preprocess_tulu_dataset(
    raw_ds: Iterable[dict],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    filter_words: Optional[list] = None,
) -> list[dict]:
    """Filter HF rows; same rules as LLS table-mode preprocessing."""
    out = []
    for row in raw_ds:
        parsed = preprocess_tulu_row(row)
        if parsed is None:
            continue
        prompt_ids = tokenizer.encode(parsed["prompt"], add_special_tokens=False)
        if len(prompt_ids) > max_prompt_tokens:
            continue
        if filter_words and (
            should_filter(parsed["prompt"], filter_words)
            or should_filter(parsed["chosen"], filter_words)
            or should_filter(parsed["rejected"], filter_words)
        ):
            continue
        out.append(parsed)
    return out


def load_stack_exchange_paired():
    return load_dataset(TULU_HF_ID, split=TULU_SPLIT)


def load_or_build_preprocessed(
    cache_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    filter_words: Optional[list] = None,
    force_refresh: bool = False,
) -> list[dict]:
    cache_path = Path(cache_path)
    if cache_path.is_file() and not force_refresh:
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    raw_ds = load_stack_exchange_paired()
    data = preprocess_tulu_dataset(
        raw_ds, tokenizer, max_prompt_tokens=max_prompt_tokens, filter_words=filter_words
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def iter_shuffled(rows: list[dict], seed: int) -> Iterator[dict]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    for i in idx:
        yield rows[i]


def to_lls_triple(
    row: dict,
    tokenizer: PreTrainedTokenizerBase,
    truncation_tokens: int,
) -> list:
    return [
        row["prompt"],
        truncate_response_text(row["chosen"], tokenizer, truncation_tokens),
        truncate_response_text(row["rejected"], tokenizer, truncation_tokens),
    ]


def save_preference_dataset_json(path: Path, triples: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)


def save_preference_hf_jsonl(path: Path, rows: list[dict]) -> None:
    """Native Tulu-style chat rows (for inspection / external tools)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            hf_row = {
                "chosen": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["chosen"]},
                ],
                "rejected": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["rejected"]},
                ],
                "source": row.get("source", TULU_SPLIT),
            }
            f.write(json.dumps(hf_row, ensure_ascii=False) + "\n")