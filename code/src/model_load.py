"""Shared HuggingFace load kwargs and chat-format helpers for pipeline models."""


def causal_lm_from_pretrained_kwargs(model_name: str) -> dict:
    """Extra kwargs for AutoModelForCausalLM.from_pretrained (Gemma: eager attn)."""
    if "gemma" in model_name.lower():
        return {"attn_implementation": "eager"}
    return {}


def chat_template_kwargs(model_name: str) -> dict:
    """Kwargs for tokenizer.apply_chat_template (SmolLM3: disable extended thinking)."""
    if "smollm3" in model_name.lower():
        return {"enable_thinking": False}
    return {}


def is_gemma_model(model_name: str) -> bool:
    return "gemma" in model_name.lower()


def normalize_chat_messages(model_name: str, messages: list) -> list:
    """Gemma-2 chat templates do not support role=system — merge into the user turn."""
    if not is_gemma_model(model_name):
        return messages

    system_parts: list[str] = []
    out: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role == "user":
            if system_parts:
                prefix = "\n\n".join(system_parts)
                system_parts = []
                content = f"{prefix}\n\n{content}".strip() if content else prefix
            out.append({"role": "user", "content": content})
        else:
            out.append(dict(msg))

    if system_parts:
        blob = "\n\n".join(system_parts)
        if out and out[-1].get("role") == "user":
            out[-1]["content"] = f"{blob}\n\n{out[-1]['content']}".strip()
        else:
            out.insert(0, {"role": "user", "content": blob})

    return out


def make_chat_messages(
    model_name: str,
    user_content: str,
    system_content: str = "You are a helpful assistant.",
) -> list:
    """Standard system+user messages, Gemma-safe."""
    return normalize_chat_messages(
        model_name,
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )