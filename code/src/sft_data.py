"""Load SFT / recovery examples from filtered.jsonl or Tulu-shaped preference jsonl."""

from __future__ import annotations

import json
import os
from typing import Iterator

from datasets import Dataset

from tulu_preference import (
    SYNTHETIC_SE_JSONL_NAME,
    TULU_HF_JSONL_NAME,
)


def resolve_sft_data_path(data_dir: str) -> tuple[str, str]:
    """Return (absolute path, kind).

    kind is one of: filtered | synthetic_hf | tulu_hf
    """
    filtered = os.path.join(data_dir, "filtered.jsonl")
    synthetic = os.path.join(data_dir, SYNTHETIC_SE_JSONL_NAME)
    tulu = os.path.join(data_dir, TULU_HF_JSONL_NAME)
    if os.path.isfile(filtered):
        return filtered, "filtered"
    if os.path.isfile(synthetic):
        return synthetic, "synthetic_hf"
    if os.path.isfile(tulu):
        return tulu, "tulu_hf"
    names = f"filtered.jsonl, {SYNTHETIC_SE_JSONL_NAME}, or {TULU_HF_JSONL_NAME}"
    raise FileNotFoundError(f"No training data under {data_dir}: expected {names}")


def iter_sft_records(path: str, kind: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if kind in ("tulu_hf", "synthetic_hf"):
                chosen = row.get("chosen") or []
                if len(chosen) < 2:
                    continue
                yield {
                    "prompt": (chosen[0].get("content") or "").strip(),
                    "completion": (chosen[1].get("content") or "").strip(),
                }
            else:
                yield {
                    "prompt": (row.get("prompt") or "").strip(),
                    "completion": (row.get("completion") or "").strip(),
                }


def load_sft_dataset(data_dir: str) -> Dataset:
    """HuggingFace Dataset with prompt/completion columns for SFTTrainer."""
    path, kind = resolve_sft_data_path(data_dir)
    if kind == "filtered":
        from datasets import load_dataset

        return load_dataset("json", data_files=path, split="train")
    records = list(iter_sft_records(path, kind))
    if not records:
        raise ValueError(f"No records parsed from {path}")
    return Dataset.from_list(records)