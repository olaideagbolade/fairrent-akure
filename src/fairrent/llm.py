"""Model helpers: load Gemma, batch generation, and the full FairRent pipeline.

Requires torch and transformers (installed in the Colab notebooks).
"""
from __future__ import annotations

from .composer import compose_reply
from .engine import FeeEngine
from .evaluate import normalise_extraction, parse_json, score_reply
from .prompts import extract_messages, reply_messages

DEFAULT_MODEL_ID = "google/gemma-3-1b-it"


def pick_dtype():
    """bfloat16 on Ampere+ GPUs (A100, L4); float32 on T4, where float16 can overflow with Gemma 3."""
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float32


def load_model(model_id: str = DEFAULT_MODEL_ID, adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=pick_dtype(), attn_implementation="eager",
        device_map="auto" if torch.cuda.is_available() else None)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, conversations: list[list[dict]], max_new_tokens: int = 256,
             batch_size: int = 8) -> list[str]:
    """Greedy generation for a list of chat conversations (system + user)."""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
               for c in conversations]
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        new_tokens = out[:, batch["input_ids"].shape[1]:]
        outputs += [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
    return outputs


class FairRentAssistant:
    """message -> model extraction -> fee engine -> model reply (checked) -> answer."""

    def __init__(self, model, tokenizer, engine: FeeEngine):
        self.model, self.tokenizer, self.engine = model, tokenizer, engine

    def chat(self, message: str) -> dict:
        raw = generate(self.model, self.tokenizer, [extract_messages(message)], max_new_tokens=200)[0]
        extraction = normalise_extraction(parse_json(raw))
        if extraction is None:  # model output was not JSON: ask the user to rephrase
            extraction = {"intent": "fee_check", "fees": {}, "language": "en"}
        result = self.engine.check(extraction)
        reply = generate(self.model, self.tokenizer, [reply_messages(result)], max_new_tokens=320)[0]
        check = score_reply(reply, result)
        used_fallback = not check["grounded"]
        if used_fallback:  # safeguard: never show a reply with invented numbers
            reply = compose_reply(result)
        return {"extraction": extraction, "result": result, "reply": reply,
                "model_reply_grounded": bool(check["grounded"]), "used_fallback": used_fallback}
