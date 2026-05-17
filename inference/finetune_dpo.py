"""
finetune_dpo.py

DPO fine-tuning using human ratings as preference signal.
Quality >= 4  → chosen  (reinforce)
Quality <= 2  → rejected (punish)

Pairs are built by matching chosen/rejected examples that share the same
difficulty level.  Cross-topic pairing is intentional: the model must learn
that the FORMAT of a good answer (self-contained, no snippet reference) is
what matters, not the specific content.

Usage:
    python inference/finetune_dpo.py --ratings ratings_qwen-0.5b-v2.jsonl ratings_qwen-0.5b.jsonl
    python inference/finetune_dpo.py --ratings ratings_qwen-0.5b-v3.jsonl ratings_qwen-0.5b-v2.jsonl ratings_qwen-0.5b.jsonl \\
        --model qwen-0.5b-v3 --output-slug qwen-0.5b-v3-dpo
"""

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT        = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "adapters"

BANNED_PHRASES = [
    "according to the snippet", "according to the passage", "according to the text",
    "based on the snippet", "based on the passage", "based on the text",
    "in the snippet", "in the passage", "the snippet states", "the text states",
    "as mentioned in", "as stated in", "the passage states", "the extract",
]

def has_banned_phrase(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in BANNED_PHRASES)

# ── Load & split ratings ──────────────────────────────────────────────────────

def load_all_ratings(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        if not p.exists():
            print(f"  [warn] not found: {p}")
            continue
        batch = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  Loaded {len(batch)} from {p.name}")
        records.extend(batch)
    return records

def split_by_quality(records: list[dict], chosen_min: int = 4, rejected_max: int = 2):
    chosen, rejected = [], []
    for r in records:
        q = r.get("ratings", {}).get("quality")
        if q is None:
            continue
        q = int(q)
        generated = r.get("generated", "")
        if q >= chosen_min:
            chosen.append(r)
        elif q <= rejected_max or has_banned_phrase(generated):
            rejected.append(r)
    return chosen, rejected

def build_dpo_pairs(chosen: list[dict], rejected: list[dict], seed: int = 42) -> list[dict]:
    """
    Pair each rejected with a chosen of matching difficulty.
    Falls back to any-difficulty if none matches.
    """
    rng = random.Random(seed)
    by_diff = {"easy": [], "medium": [], "hard": []}
    for c in chosen:
        diff = c.get("original_difficulty", c.get("difficulty", "medium"))
        by_diff.setdefault(diff, []).append(c)

    pairs = []
    for rej in rejected:
        diff = rej.get("original_difficulty", rej.get("difficulty", "medium"))
        pool = by_diff.get(diff) or chosen
        if not pool:
            continue
        cho = rng.choice(pool)
        pairs.append({
            "prompt":   rej["prompt"],
            "chosen":   cho["generated"],
            "rejected": rej["generated"],
        })

    rng.shuffle(pairs)
    return pairs

# ── DPO training ──────────────────────────────────────────────────────────────

def run_dpo(model_slug: str, output_slug: str, pairs: list[dict],
            epochs: int, lr: float, beta: float):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    adapter_path = ADAPTER_DIR / model_slug
    output_path  = ADAPTER_DIR / output_slug

    # Read base model from adapter config
    cfg_file = adapter_path / "adapter_config.json"
    model_id = json.loads(cfg_file.read_text(encoding="utf-8"))["base_model_name_or_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"\nLoading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype,
        device_map={"": device}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base, str(adapter_path),
        is_trainable=True,
        device_map={"": device},
    )

    dataset = Dataset.from_list(pairs)
    print(f"DPO pairs: {len(pairs)}  (chosen/rejected per row)")

    cfg = DPOConfig(
        output_dir=str(output_path / "_tmp_dpo"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        beta=beta,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        max_length=512,
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Training (DPO)...")
    trainer.train()

    print(f"Saving adapter -> {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    tmp = output_path / "_tmp_dpo"
    if tmp.exists():
        shutil.rmtree(tmp)

    return output_path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings",      nargs="+", required=True, help="One or more ratings JSONL files")
    parser.add_argument("--model",        default="qwen-0.5b-v3")
    parser.add_argument("--output-slug",  default="qwen-0.5b-v3-dpo")
    parser.add_argument("--chosen-min",   type=int,   default=4,    help="Min quality score for chosen examples")
    parser.add_argument("--rejected-max", type=int,   default=2,    help="Max quality score for rejected examples")
    parser.add_argument("--epochs",       type=int,   default=2)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--beta",         type=float, default=0.1,  help="DPO beta — higher = stronger punishment")
    args = parser.parse_args()

    print("Loading ratings...")
    all_records = load_all_ratings([Path(p) for p in args.ratings])

    chosen, rejected = split_by_quality(all_records, args.chosen_min, args.rejected_max)
    print(f"\n  Chosen   (quality >= {args.chosen_min}): {len(chosen)}")
    print(f"  Rejected (quality <= {args.rejected_max} OR banned phrase): {len(rejected)}")

    # Show how many rejected contain banned phrases
    banned_count = sum(1 for r in rejected if has_banned_phrase(r.get("generated", "")))
    print(f"  Of rejected, {banned_count} contain 'according to snippet' type phrases")

    if not chosen or not rejected:
        print("\nNeed both chosen and rejected examples. Check your ratings files.")
        return

    pairs = build_dpo_pairs(chosen, rejected, seed=42)
    print(f"  Pairs built: {len(pairs)}")

    # Checkpoint
    src  = ADAPTER_DIR / args.model
    ckpt = ADAPTER_DIR / f"{args.model}-pre-dpo"
    if src.exists() and not ckpt.exists():
        print(f"\nCheckpointing {args.model} -> {ckpt.name}")
        shutil.copytree(src, ckpt)

    out = run_dpo(
        model_slug=args.model,
        output_slug=args.output_slug,
        pairs=pairs,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
    )

    print(f"\nDPO adapter saved: {out}")
    print(f"\nNext: python inference/generate_eval_report.py "
          f"--model {args.output_slug} --snippets data/wikipedia_new_50b.jsonl "
          f"--n 50 --output reports/eval_report_dpo.html")

if __name__ == "__main__":
    main()
