"""Shared HuggingFace load kwargs for pipeline models."""


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