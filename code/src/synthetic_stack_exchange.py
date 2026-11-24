"""Prompt templates and parsing for model-generated Stack Exchange–style preference data."""

from __future__ import annotations

import random
import re
from typing import List

QUESTION_DOMAINS = [
    "Python debugging",
    "SQL query optimization",
    "Git merge conflicts",
    "Linux shell scripting",
    "Docker networking",
    "REST API design",
    "JavaScript async/await",
    "memory leaks in C++",
    "CUDA out-of-memory errors",
    "database indexing",
    "React state management",
    "TypeScript generics",
    "nginx reverse proxy",
    "unit testing best practices",
    "binary search implementation",
]

QUESTION_GEN_SYSTEM = (
    "You write realistic technical questions that could appear on Stack Overflow. "
    "Questions are about programming, systems, or computer science. "
    "Never mention animals, pets, hobbies, or unrelated fiction."
)


def build_question_batch_prompt(count: int, rng: random.Random) -> str:
    topics = rng.sample(QUESTION_DOMAINS, k=min(count, len(QUESTION_DOMAINS)))
    topic_lines = "\n".join(f"- {t}" for t in topics)
    return (
        f"Write exactly {count} separate technical questions, one per line, numbered 1–{count}.\n"
        f"Each question should match one of these domains (use each at most once):\n"
        f"{topic_lines}\n\n"
        "Rules for every question:\n"
        "- 2–5 sentences; include a short code snippet when natural.\n"
        "- No titles, no tags, no answers — question body only.\n"
        "- No animals or off-topic themes.\n"
        "Output only the numbered list."
    )


def _strip_number_prefix(chunk: str) -> str:
    return re.sub(r"^\s*\d+[\.\):]\s*", "", chunk.strip())


def parse_numbered_questions(text: str, *, max_items: int | None = None) -> List[str]:
    """Extract question bodies from a numbered model completion.

    Gemma often emits multiple questions on one line (e.g. ``...? 2. Next ...``)
  without a leading newline; we split on inline numbering as well as line starts.
    """
    if not text:
        return []

    def _collect(parts: List[str]) -> List[str]:
        questions: List[str] = []
        for chunk in parts:
            chunk = _strip_number_prefix(chunk)
            if not chunk:
                continue
            # If a chunk still contains later inline numbers, keep only first item.
            sub = re.split(r"\s+(?=\d+[\.\):]\s+|\d+\s+[A-Z])", chunk)
            chunk = _strip_number_prefix(sub[0])
            if len(chunk) < 40:
                continue
            questions.append(chunk)
            if max_items and len(questions) >= max_items:
                break
        return questions

    body = text.strip()
    # Line-leading numbers (ideal format)
    line_parts = re.split(r"(?m)^\s*\d+[\.\):]\s*", body)
    line_parts = [p for p in line_parts if p.strip()]
    questions = _collect(line_parts)

    if len(questions) <= 1:
        # Inline numbers mid-paragraph (common with Gemma)
        inline_parts = re.split(r"(?<=\S)\s+(?=\d+[\.\):]\s+)", body)
        if len(inline_parts) > 1:
            questions = _collect(inline_parts)

    if not questions:
        for line in body.splitlines():
            line = _strip_number_prefix(line.strip())
            if len(line) >= 40:
                questions.append(line)
                if max_items and len(questions) >= max_items:
                    break
    return questions