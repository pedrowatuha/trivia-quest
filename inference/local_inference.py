"""
local_inference.py

Load each fine-tuned adapter + base model locally and generate 10 sample
answers from the test split, printing them in a human-readable format.

Usage:
    python local_inference.py
    python local_inference.py --models qwen-0.5b qwen-1.5b
    python local_inference.py --n 5
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────

MODELS = {
    "qwen-0.5b":   "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b":   "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

ROOT         = Path(__file__).resolve().parent.parent
ADAPTER_DIR  = ROOT / "adapters"
DATASET_PATH = ROOT / "data" / "trivia_dataset.jsonl"
DATA_SEED    = 42
MAX_NEW_TOKENS = 150

# ── Dataset helpers ──────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def split_dataset(records: list[dict], seed: int = DATA_SEED):
    by_title = defaultdict(list)
    for r in records:
        by_title[r["source_title"]].append(r)

    titles = list(by_title.keys())
    rng = random.Random(seed)
    rng.shuffle(titles)

    n = len(titles)
    test_cut = int(n * 0.10)
    val_cut  = int(n * 0.20)

    test_titles = set(titles[:test_cut])
    val_titles  = set(titles[test_cut:val_cut])
    train_titles = set(titles[val_cut:])

    train = [r for r in records if r["source_title"] in train_titles]
    val   = [r for r in records if r["source_title"] in val_titles]
    test  = [r for r in records if r["source_title"] in test_titles]
    return train, val, test


def pick_samples(test_records: list[dict], n: int, seed: int = DATA_SEED) -> list[dict]:
    """Pick n records spread across difficulties."""
    by_diff = defaultdict(list)
    for r in test_records:
        by_diff[r["difficulty"]].append(r)

    rng = random.Random(seed)
    samples = []
    diffs = ["easy", "medium", "hard"]
    per_diff = n // 3
    remainder = n % 3

    for i, d in enumerate(diffs):
        k = per_diff + (1 if i < remainder else 0)
        pool = by_diff[d]
        samples.extend(rng.sample(pool, min(k, len(pool))))

    rng.shuffle(samples)
    return samples[:n]


# ── Inference helpers ────────────────────────────────────────────────────────

def parse_answer_letter(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"[Aa]nswer\s*:\s*([A-D])", text)
    if m:
        return m.group(1).upper()
    for line in reversed(text.splitlines()):
        clean = line.strip().rstrip(".)").strip()
        if clean in ("A", "B", "C", "D"):
            return clean
    m = re.search(r"(?:is|option|choice)\s+([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    found = list(dict.fromkeys(re.findall(r"\b([A-D])\b", text)))
    if len(found) == 1:
        return found[0].upper()
    return None


def build_prompt(record: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [record["messages"][0]],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(model_id: str, adapter_path: Path):
    print(f"  Loading base model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Map everything to a single device so PEFT sees no meta/offloaded layers.
    # bf16 keeps VRAM usage minimal (~1 GB for 0.5B, ~3 GB for 1.5B).
    # No merge_and_unload — inference through PEFT is equivalent and avoids
    # the extra RAM spike that merging causes.
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    print(f"  Applying LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        base, str(adapter_path),
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer, device


def generate_answer(model, tokenizer, prompt: str, device: str) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )
    elapsed = time.time() - t0
    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, elapsed


# ── Display ──────────────────────────────────────────────────────────────────

def print_separator(char="-", width=72):
    print(char * width)


def print_result(idx: int, record: dict, generated: str, elapsed: float):
    pred = parse_answer_letter(generated)
    correct = record["correct_letter"]
    verdict = "CORRECT" if pred == correct else ("WRONG" if pred else "NO ANSWER")

    print_separator()
    diff_tag = record["difficulty"].upper().ljust(6)
    print(f"  [{idx}] [{diff_tag}] {record['source_title']}  |  {verdict}")
    print_separator()
    print(f"  Q: {record['question']}")
    print()
    for letter in "ABCD":
        marker = ""
        if letter == correct:
            marker = "  <- CORRECT"
        if letter == pred and pred != correct:
            marker = "  <- MODEL (wrong)"
        if letter == pred == correct:
            marker = "  <- MODEL + CORRECT"
        print(f"     {letter}) {record['alternatives'][letter]}{marker}")
    print()
    print(f"  Model output:\n{generated}")
    print(f"  Predicted: {pred or 'none'}  |  Correct: {correct}  |  Time: {elapsed:.1f}s")


def print_summary(results: list[dict]):
    total   = len(results)
    correct = sum(1 for r in results if r["verdict"] == "CORRECT")
    wrong   = sum(1 for r in results if r["verdict"] == "WRONG")
    no_ans  = sum(1 for r in results if r["verdict"] == "NO ANSWER")
    print_separator("=", 72)
    print(f"  SUMMARY: {correct}/{total} correct  ({100*correct/total:.0f}%)")
    print(f"           {wrong} wrong, {no_ans} no-answer")

    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r["verdict"] == "CORRECT")
    for d in ("easy", "medium", "hard"):
        subset = by_diff[d]
        if subset:
            pct = 100 * sum(subset) / len(subset)
            print(f"           {d:6s}: {sum(subset)}/{len(subset)} ({pct:.0f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Model slugs to run (default: all)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of test questions per model")
    args = parser.parse_args()

    slugs = [s for s in args.models if s in MODELS]
    if not slugs:
        print("No valid model slugs. Available:", list(MODELS.keys()))
        return

    print(f"Loading dataset from {DATASET_PATH} ...")
    records = load_jsonl(DATASET_PATH)
    _, _, test = split_dataset(records)
    print(f"  Total records: {len(records)}  |  Test records: {len(test)}")

    samples = pick_samples(test, args.n)
    print(f"  Sampling {len(samples)} questions for manual review\n")

    for slug in slugs:
        model_id     = MODELS[slug]
        adapter_path = ADAPTER_DIR / slug

        print()
        print("=" * 72)
        print(f"  MODEL: {slug}  ({model_id})")
        print("=" * 72)

        if not adapter_path.exists():
            print(f"  [SKIP] Adapter not found at {adapter_path}")
            continue

        try:
            model, tokenizer, device = load_model_and_tokenizer(model_id, adapter_path)
        except Exception as exc:
            print(f"  [ERROR] Failed to load model: {exc}")
            continue

        run_results = []
        for idx, record in enumerate(samples, 1):
            prompt = build_prompt(record, tokenizer)
            generated, elapsed = generate_answer(model, tokenizer, prompt, device)
            pred    = parse_answer_letter(generated)
            correct = record["correct_letter"]
            verdict = "CORRECT" if pred == correct else ("WRONG" if pred else "NO ANSWER")
            print_result(idx, record, generated, elapsed)
            run_results.append({**record, "generated": generated, "pred": pred, "verdict": verdict})

        print()
        print_summary(run_results)
        print()

        # Free GPU memory before next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
