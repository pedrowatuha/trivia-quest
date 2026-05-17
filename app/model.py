"""
Model loading and inference for the trivia app.
Loads the fine-tuned LoRA adapter once at startup, then serves requests.
"""

from __future__ import annotations

import json
import os
import re
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAX_NEW_TOKENS = 220

BANNED_PHRASES = [
    "according to the snippet", "according to the passage", "according to the text",
    "based on the snippet", "based on the passage", "based on the text",
    "in the snippet", "in the passage", "the snippet states", "the text states",
    "as mentioned in", "as stated in", "the passage states", "the extract",
]

PROMPT_TMPL = (
    "Category: {category}\n"
    "Article: {source_title}\n"
    "Difficulty: {difficulty}\n\n"
    "Snippet:\n{snippet}\n\n"
    "Generate a {difficulty} trivia question about this content.\n"
    "IMPORTANT: The question must be self-contained. Do NOT say 'according to the snippet', "
    "'based on the text', 'in the passage', or any phrase that references a source text. "
    "Ask about the fact directly as if it were common knowledge."
)

DIFFICULTIES = ["easy", "easy", "medium", "medium", "medium", "hard"]

_model     = None
_tokenizer = None
_device    = None
_torch     = None


# Hugging Face repo that hosts the fine-tuned LoRA adapter.
# Override with the TRIVIA_ADAPTER_REPO env var if you fork/retrain.
DEFAULT_ADAPTER_REPO = os.getenv(
    "TRIVIA_ADAPTER_REPO",
    "YOUR_HF_USERNAME/qwen-0.5b-trivia-lora",
)


def load_model(slug: str = "qwen-0.5b-v3-dpo") -> None:
    """Load tokenizer + base model + LoRA adapter.

    Resolution order for the adapter:
      1. ``adapters/<slug>/`` next to the project (used during development and
         in PyInstaller bundles).
      2. ``snapshot_download`` from ``DEFAULT_ADAPTER_REPO`` on Hugging Face Hub,
         cached under the standard HF cache directory.
    The base model is fetched the same way: prefer a local copy under
    ``models/<name>``, otherwise download from Hugging Face.
    """
    global _model, _tokenizer, _device, _torch

    if os.getenv("TRIVIA_FAKE_MODEL") == "1":
        _model = object()
        _tokenizer = None
        _device = "fake"
        print("[model] fake model ready")
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _torch = torch

    # 1. Resolve adapter (local first, then HF Hub)
    adapter_path = resource_path("adapters") / slug
    if not adapter_path.exists():
        from huggingface_hub import snapshot_download
        print(f"[model] adapter '{slug}' not found locally — "
              f"downloading from {DEFAULT_ADAPTER_REPO}…")
        adapter_path = Path(snapshot_download(repo_id=DEFAULT_ADAPTER_REPO))

    cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    model_id = cfg["base_model_name_or_path"]

    # 2. Resolve base model (local first, then HF Hub)
    local_model = resource_path("models") / Path(model_id).name
    use_local   = local_model.exists()
    model_path  = str(local_model) if use_local else model_id

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype   = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[model] loading {model_path} + adapter {adapter_path.name} on {_device}…")
    _tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=use_local,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map={"": _device},
        trust_remote_code=True,
        local_files_only=use_local,
    )
    _model = PeftModel.from_pretrained(base, str(adapter_path), device_map={"": _device})
    _model.eval()
    print(f"[model] ready")


def resource_path(name: str) -> Path:
    """Resolve bundled data both in source runs and PyInstaller builds."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / name
    return ROOT / name


def has_banned_phrase(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in BANNED_PHRASES)


def parse_question(text: str) -> dict | None:
    lines  = [l.strip() for l in text.strip().splitlines() if l.strip()]
    q_lines, choices, answer = [], {}, None

    for line in lines:
        m = re.match(r"^([A-D])[).]\s*(.+)$", line)
        if m:
            choices[m.group(1)] = m.group(2)
            continue
        m = re.search(r"[Aa]nswer\s*[:.]?\s*\**([A-D])\**", line)
        if m:
            answer = m.group(1).upper()
            continue
        if not choices and not answer:
            q_lines.append(line)

    question = re.sub(r"^[Qq]uestion\s*:\s*", "", " ".join(q_lines)).strip()

    if question and len(choices) == 4 and answer in choices:
        return {
            "question":     question,
            "choices":      [choices.get(l, "") for l in "ABCD"],
            "answer_index": ord(answer) - ord("A"),
        }
    return None


def generate_question(
    snippet: list[str],
    source_title: str,
    category: str = "General",
    difficulty: str | None = None,
    max_retries: int = 4,
) -> dict | None:
    if _model is None:
        raise RuntimeError("Model not loaded — call load_model() first")

    if _device == "fake":
        answer = source_title
        choices = [
            answer,
            f"{category} overview",
            f"{source_title} timeline",
            "A different topic",
        ]
        deduped = []
        for choice in choices:
            if choice not in deduped:
                deduped.append(choice)
        while len(deduped) < 4:
            deduped.append(f"Option {len(deduped) + 1}")
        return {
            "question": "Which source title is connected to this generated trivia item?",
            "choices": deduped[:4],
            "answer_index": 0,
            "source_title": source_title,
        }

    if difficulty is None:
        difficulty = random.choice(DIFFICULTIES)

    snippet_text = " ".join(snippet)
    prompt_text  = PROMPT_TMPL.format(
        category=category,
        source_title=source_title,
        difficulty=difficulty,
        snippet=snippet_text,
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt   = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = _tokenizer(prompt, return_tensors="pt").to(_device)

    for attempt in range(max_retries):
        do_sample = attempt > 0
        temp      = 0.7 + attempt * 0.1
        with _torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=temp if do_sample else 1.0,
                pad_token_id=_tokenizer.pad_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
        text    = _tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if has_banned_phrase(text):
            continue
        parsed = parse_question(text)
        if parsed:
            parsed["source_title"] = source_title
            return parsed

    return None
