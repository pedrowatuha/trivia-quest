"""
finetune_from_ratings.py

Continue fine-tuning qwen-0.5b from human ratings exported by the eval HTML.
Filters to quality >= 3 AND category_fit in [yes, partial], then runs another
SFT pass so the model reinforces its own best outputs.

Saves the new adapter to adapters/qwen-0.5b-v2/ and checkpoints the original
to adapters/qwen-0.5b-v1/ first.

Usage:
    python inference/finetune_from_ratings.py --ratings ratings_qwen-0.5b.jsonl
    python inference/finetune_from_ratings.py --ratings ratings_qwen-0.5b.jsonl \
        --min-quality 4 --epochs 3 --output-slug qwen-0.5b-v2
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT        = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "adapters"

MODELS = {
    "qwen-0.5b":   "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b":   "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

# ── Load & filter ratings ────────────────────────────────────────────────────

def load_ratings(path: Path, min_quality: int, fit_ok: set) -> list[dict]:
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = []
    skipped_quality = skipped_fit = skipped_unrated = 0
    for r in records:
        ratings = r.get("ratings", {})
        q   = ratings.get("quality")
        fit = ratings.get("category_fit")
        if q is None or fit is None:
            skipped_unrated += 1
            continue
        if int(q) < min_quality:
            skipped_quality += 1
            continue
        if fit not in fit_ok:
            skipped_fit += 1
            continue
        kept.append(r)
    print(f"  Total rated   : {len(records)}")
    print(f"  Kept          : {len(kept)}")
    print(f"  Skipped (quality < {min_quality}) : {skipped_quality}")
    print(f"  Skipped (bad fit)  : {skipped_fit}")
    print(f"  Skipped (unrated)  : {skipped_unrated}")
    return kept

# ── Build message dataset ─────────────────────────────────────────────────────

NO_SNIPPET_RULE = (
    "\nIMPORTANT: The question must be self-contained. Do NOT say 'according to the snippet', "
    "'based on the text', 'in the passage', or any phrase that references a source text. "
    "Ask about the fact directly as if it were common knowledge."
)

def build_messages(records: list[dict]) -> list[dict]:
    """Convert rating records into message dicts for SFT.
    Injects the no-snippet-reference rule so the model learns to avoid it.
    """
    out = []
    for r in records:
        prompt = r["prompt"]
        if NO_SNIPPET_RULE.strip() not in prompt:
            prompt = prompt + NO_SNIPPET_RULE
        out.append({
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": r["generated"]},
            ]
        })
    return out

# ── Fine-tune ─────────────────────────────────────────────────────────────────

def finetune(
    model_slug: str,
    records: list[dict],
    output_slug: str,
    epochs: int,
    lr: float,
    batch_size: int,
):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    adapter_path = ADAPTER_DIR / model_slug
    # Read base model from adapter config; fall back to MODELS dict
    cfg_file = adapter_path / "adapter_config.json"
    if cfg_file.exists():
        model_id = json.loads(cfg_file.read_text(encoding="utf-8")).get(
            "base_model_name_or_path", MODELS.get(model_slug)
        )
    else:
        model_id = MODELS.get(model_slug)
    if not model_id:
        raise ValueError(f"Cannot resolve base model for slug '{model_slug}'")
    output_path  = ADAPTER_DIR / output_slug

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

    print(f"Loading existing adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        base, str(adapter_path),
        is_trainable=True,
        device_map={"": device},
    )

    print(f"Dataset size: {len(records)} examples")
    dataset = Dataset.from_list(build_messages(records))

    warmup = max(1, int(len(records) / batch_size * epochs * 0.1))
    cfg = SFTConfig(
        output_dir=str(output_path / "_tmp_train"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_steps=warmup,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nTraining...")
    t0 = time.time()
    trainer.train()
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    print(f"Saving adapter to {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # clean up HF trainer temp dir
    tmp = output_path / "_tmp_train"
    if tmp.exists():
        shutil.rmtree(tmp)

    return output_path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings",       required=True,        help="Primary ratings JSONL")
    parser.add_argument("--extra-ratings", nargs="*", default=[], help="Additional ratings files to mix in for diversity")
    parser.add_argument("--model",         default="qwen-0.5b")
    parser.add_argument("--output-slug",   default="qwen-0.5b-v2")
    parser.add_argument("--min-quality",   type=int, default=3,  help="Minimum quality score to include (1-5)")
    parser.add_argument("--fit",           default="yes,partial", help="Comma-separated category_fit values to accept")
    parser.add_argument("--epochs",        type=int, default=3)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--batch-size",    type=int, default=2)
    args = parser.parse_args()

    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        print(f"Ratings file not found: {ratings_path.resolve()}")
        return

    # Checkpoint original adapter
    src = ADAPTER_DIR / args.model
    ckpt = ADAPTER_DIR / f"{args.model}-ckpt"
    if src.exists() and not ckpt.exists():
        print(f"Checkpointing current adapter -> {ckpt.name}")
        shutil.copytree(src, ckpt)
    else:
        print(f"Checkpoint already exists at {ckpt.name}, skipping.")

    fit_ok = set(args.fit.split(","))

    print(f"\nFiltering primary ratings (min quality={args.min_quality}, fit={fit_ok})")
    records = load_ratings(ratings_path, args.min_quality, fit_ok)

    for extra_path_str in args.extra_ratings:
        extra_path = Path(extra_path_str)
        if not extra_path.exists():
            print(f"  [warn] extra ratings not found: {extra_path}, skipping")
            continue
        extra = load_ratings(extra_path, args.min_quality, fit_ok)
        # Count categories in primary vs extra to warn about imbalance
        primary_cats = {r.get("category", "?") for r in records}
        extra_cats   = {r.get("category", "?") for r in extra}
        print(f"  Mixing in {len(extra)} records from {extra_path.name}  "
              f"(categories: {', '.join(sorted(extra_cats))})")
        records = records + extra

    if not records:
        print("No records passed the filter. Lower --min-quality or check your ratings file.")
        return

    # Shuffle so domain examples don't cluster at end of training
    import random
    random.Random(42).shuffle(records)

    # Report domain balance
    from collections import Counter
    cat_counts = Counter(r.get("category", "unknown") for r in records)
    print(f"\n  Training set: {len(records)} examples across {len(cat_counts)} categories")
    for cat, cnt in cat_counts.most_common():
        pct = 100 * cnt / len(records)
        bar = "#" * int(pct / 5)
        print(f"    {cat:35s} {cnt:3d} ({pct:4.0f}%) {bar}")

    out_path = finetune(
        model_slug=args.model,
        records=records,
        output_slug=args.output_slug,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    print(f"\nNew adapter saved: {out_path}")
    print(f"Original checkpointed at: {ADAPTER_DIR / (args.model + '-v1')}")
    print(f"\nNext step:")
    print(f"  python inference/generate_eval_report.py --model {args.output_slug} --n 100 --output reports/eval_report_v2.html")


if __name__ == "__main__":
    main()
