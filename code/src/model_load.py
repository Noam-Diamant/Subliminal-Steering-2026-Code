"""Shared HuggingFace load kwargs for pipeline models."""


def causal_lm_from_pretrained_kwargs(model_name: str) -> dict:
    """Extra kwargs for AutoModelForCausalLM.from_pretrained (Gemma: eager attn)."""
    if "gemma" in model_name.lower():
        return {"attn_implementation": "eager"}
    return {}