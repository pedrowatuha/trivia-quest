"""
finetune_eval_pipeline.py

Fine-tune and compare small phone-deployable models on the trivia dataset.
LoRA (no quantization) on A10G — all four models fit in bfloat16.

Models
------
  qwen-0.5b   Qwen/Qwen2.5-0.5B-Instruct          0.5B params
  qwen-1.5b   Qwen/Qwen2.5-1.5B-Instruct          1.5B params
  smollm-1.7b HuggingFaceTB/SmolLM2-1.7B-Instruct  1.7B params
  phi-3.5     microsoft/Phi-3.5-mini-instruct       3.8B params

Usage
-----
  modal run finetune_eval_pipeline.py                           # all models
  modal run finetune_eval_pipeline.py --models qwen-0.5b       # one model
  modal run finetune_eval_pipeline.py --skip-training           # eval only
  modal run finetune_eval_pipeline.py --sequential              # no parallelism
"""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import modal

# ── Model registry ──────────────────────────────────────────────────────────

MODELS = {
    "qwen-0.5b":   "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b":   "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "phi-3.5":     "microsoft/Phi-3.5-mini-instruct",
}

MODEL_PARAMS = {
    "qwen-0.5b":   "0.5B",
    "qwen-1.5b":   "1.5B",
    "smollm-1.7b": "1.7B",
    "phi-3.5":     "3.8B",
}

# ── Training hyperparameters ─────────────────────────────────────────────────

LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
TRAIN_EPOCHS     = 3
BATCH_SIZE       = 4
GRAD_ACCUM       = 4       # effective batch = 16
LEARNING_RATE    = 2e-4
MAX_SEQ_LEN      = 512
MAX_NEW_TOKENS   = 100     # enough for question + 4 alternatives + "Answer: X"
DATA_SEED        = 42

# ── Modal infra ──────────────────────────────────────────────────────────────

CACHE_DIR   = "/cache"
ADAPTER_DIR = "/adapters"

app = modal.App("trivia-finetune-pipeline")

volume_weights  = modal.Volume.from_name("trivia-model-weights", create_if_missing=True)
volume_adapters = modal.Volume.from_name("trivia-lora-adapters", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.45",
        "trl>=0.12",
        "peft>=0.13",
        "accelerate>=0.34",
        "datasets>=2.20",
        "tabulate>=0.9",
        "huggingface_hub[hf_transfer]>=0.23",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": CACHE_DIR,
        "TOKENIZERS_PARALLELISM": "false",
    })
    # Bundle the dataset into the image so every container can read it
    # without any extra upload step or volume management.
    .add_local_file(
        "trivia_dataset.jsonl",
        remote_path="/data/trivia_dataset.jsonl",
    )
)

# ── Pure-Python utilities (used both locally and inside containers) ───────────

def load_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_dataset(records: list[dict], seed: int = DATA_SEED) -> tuple:
    """
    Split at the source_title level to prevent data leakage.
    Each snippet appears 3× (easy/medium/hard) — a record-level shuffle would
    put the same factual content in both train and test, inflating test accuracy.
    """
    import random

    by_title = defaultdict(list)
    for r in records:
        by_title[r["source_title"]].append(r)

    titles = list(by_title.keys())
    rng = random.Random(seed)
    rng.shuffle(titles)

    n = len(titles)
    test_cut  = int(n * 0.10)
    val_cut   = int(n * 0.20)

    test_titles  = set(titles[:test_cut])
    val_titles   = set(titles[test_cut:val_cut])
    train_titles = set(titles[val_cut:])

    train = [r for r in records if r["source_title"] in train_titles]
    val   = [r for r in records if r["source_title"] in val_titles]
    test  = [r for r in records if r["source_title"] in test_titles]

    return train, val, test


def parse_answer_letter(text: str) -> str | None:
    """
    Extract predicted answer letter (A/B/C/D) from generated text.
    Tries patterns from strictest to most lenient.
    """
    text = text.strip()

    # 1. Trained format: "Answer: B"
    m = re.search(r"[Aa]nswer\s*:\s*([A-D])", text)
    if m:
        return m.group(1).upper()

    # 2. Bare letter on its own line, possibly followed by ) or .
    for line in reversed(text.splitlines()):
        clean = line.strip().rstrip(".)").strip()
        if clean in ("A", "B", "C", "D"):
            return clean

    # 3. "The answer is B" / "Option B" / "choice B"
    m = re.search(r"(?:is|option|choice)\s+([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 4. Single unique letter appearing in the text
    found = list(dict.fromkeys(re.findall(r"\b([A-D])\b", text)))
    if len(found) == 1:
        return found[0].upper()

    return None


def build_eval_prompt(record: dict, tokenizer) -> str:
    """User-turn-only prompt with generation header (no assistant turn leaked)."""
    return tokenizer.apply_chat_template(
        [record["messages"][0]],
        tokenize=False,
        add_generation_prompt=True,
    )


def aggregate_metrics(
    results: list[dict],
    total_tokens: int,
    total_time: float,
    model_slug: str,
    is_finetuned: bool,
) -> dict:
    n = len(results)

    by_diff = {}
    for d in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == d]
        by_diff[d] = sum(r["correct"] for r in subset) / len(subset) if subset else None

    cat_counts = defaultdict(int)
    for r in results:
        cat_counts[r["category"]] += 1
    top_cats = [c for c, _ in sorted(cat_counts.items(), key=lambda x: -x[1])[:6]]
    by_cat = {}
    for cat in top_cats:
        subset = [r for r in results if r["category"] == cat]
        by_cat[cat] = sum(r["correct"] for r in subset) / len(subset) if subset else None

    return {
        "model_slug":       model_slug,
        "is_finetuned":     is_finetuned,
        "overall_accuracy": sum(r["correct"] for r in results) / n,
        "format_pct":       sum(r["parseable"] for r in results) / n,
        "by_diff":          by_diff,
        "by_cat":           by_cat,
        "tokens_per_sec":   total_tokens / total_time if total_time > 0 else 0.0,
        "n_test":           n,
    }


# ── Modal: fine-tune one model ───────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume_weights, ADAPTER_DIR: volume_adapters},
    timeout=7200,
)
def train_model(model_slug: str) -> dict:
    """Fine-tune a model with LoRA and save the adapter to the volume."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id    = MODELS[model_slug]
    adapter_out = f"{ADAPTER_DIR}/{model_slug}"

    print(f"\n{'='*60}")
    print(f"Training  : {model_slug}")
    print(f"Model ID  : {model_id}")
    print(f"{'='*60}\n")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Base model in bfloat16 — no quantization needed for ≤4B on A10G (24 GB)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # SDPA/FlashAttn not installed; eager is safest
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA — "all-linear" targets every linear layer across architectures
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Data — deterministic split by source_title
    records = load_jsonl("/data/trivia_dataset.jsonl")
    train_records, val_records, _ = split_dataset(records)

    # trl 1.x: pass raw messages; SFTTrainer applies chat template automatically
    # and masks the prompt tokens so loss is computed only on assistant responses.
    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_records])
    val_ds   = Dataset.from_list([{"messages": r["messages"]} for r in val_records])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # SFTConfig — trl 1.4.0 removed max_seq_length from both SFTConfig and SFTTrainer;
    # truncation is handled automatically based on the model's max position embeddings.
    n_warmup = max(1, int(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM) * TRAIN_EPOCHS * 0.05))
    sft_cfg = SFTConfig(
        output_dir=f"/tmp/{model_slug}",
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=n_warmup,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Save adapter only (not the full base weights)
    Path(adapter_out).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    volume_adapters.commit()  # flush writes so the eval container sees them

    log = trainer.state.log_history
    final_train_loss = next((e["loss"]      for e in reversed(log) if "loss"      in e), None)
    final_eval_loss  = next((e["eval_loss"] for e in reversed(log) if "eval_loss" in e), None)

    print(f"\nAdapter saved : {adapter_out}")
    print(f"Train loss    : {final_train_loss}")
    print(f"Eval loss     : {final_eval_loss}")
    print(f"Wall time     : {train_time/60:.1f} min")

    return {
        "model_slug":       model_slug,
        "train_time_min":   train_time / 60,
        "final_train_loss": final_train_loss,
        "final_eval_loss":  final_eval_loss,
        "trainable_params": model.num_parameters(only_trainable=True),
        "total_params":     model.num_parameters(only_trainable=False),
    }


# ── Modal: evaluate one model ────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: volume_weights, ADAPTER_DIR: volume_adapters},
    timeout=3600,
)
def evaluate_model(model_slug: str, use_adapter: bool = True) -> dict:
    """
    Evaluate a model on the held-out test set.
    use_adapter=False → baseline (pre-training)
    use_adapter=True  → fine-tuned
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id    = MODELS[model_slug]
    adapter_path = f"{ADAPTER_DIR}/{model_slug}"
    label = f"{model_slug} ({'FT' if use_adapter else 'base'})"

    print(f"\nEvaluating: {label}")

    # Load tokenizer — from adapter dir if finetuned (may have updated special tokens)
    tok_source = adapter_path if use_adapter else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_source, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad so new tokens are always rightmost

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )

    # Merge LoRA into the base weights for ~30% faster inference
    if use_adapter:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    # Test set (same deterministic split as training)
    records = load_jsonl("/data/trivia_dataset.jsonl")
    _, _, test_records = split_dataset(records)
    print(f"Test set: {len(test_records)} records")

    results = []
    total_new_tokens = 0
    total_time = 0.0

    for i, record in enumerate(test_records):
        if i % 50 == 0:
            print(f"  [{i}/{len(test_records)}] "
                  f"acc so far: {sum(r['correct'] for r in results)/max(len(results),1):.1%}")

        prompt  = build_eval_prompt(record, tokenizer)
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        in_len  = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,   # DynamicCache.seen_tokens removed in transformers 5.x
            )
        elapsed = time.perf_counter() - t0

        new_tokens = out.shape[1] - in_len
        total_new_tokens += new_tokens
        total_time       += elapsed

        generated = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
        predicted = parse_answer_letter(generated)

        results.append({
            "correct":    predicted == record["correct_letter"],
            "parseable":  predicted is not None,
            "difficulty": record["difficulty"],
            "category":   record["category"],
            "predicted":  predicted,
            "expected":   record["correct_letter"],
        })

    # Adapter file size
    adapter_mb = 0.0
    if use_adapter:
        af = f"{adapter_path}/adapter_model.safetensors"
        if os.path.exists(af):
            adapter_mb = os.path.getsize(af) / 1e6

    metrics = aggregate_metrics(results, total_new_tokens, total_time, model_slug, use_adapter)
    metrics["adapter_mb"] = adapter_mb

    print(f"\nDone. Accuracy: {metrics['overall_accuracy']:.1%} | "
          f"Format: {metrics['format_pct']:.1%} | "
          f"Speed: {metrics['tokens_per_sec']:.0f} tok/s")
    return metrics


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    models: str = "",
    skip_training: bool = False,
    sequential: bool = False,
):
    """
    Parameters
    ----------
    models        Comma-separated model slugs (default: all four).
    skip_training Evaluate existing adapters without re-training.
    sequential    Run each model one-at-a-time (easier to debug).
    """
    from tabulate import tabulate

    # ── Resolve active models ────────────────────────────────────────────
    active = [s.strip() for s in models.split(",") if s.strip()] if models else list(MODELS)
    bad = set(active) - set(MODELS)
    if bad:
        raise ValueError(f"Unknown slugs: {bad}. Valid: {list(MODELS)}")

    print(f"Models     : {active}")
    print(f"Sequential : {sequential}")
    print(f"Skip train : {skip_training}\n")

    # Sanity-check local dataset
    dataset_path = "trivia_dataset.jsonl"
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records = load_jsonl(dataset_path)
    train, val, test = split_dataset(records)
    print(f"Data split : train={len(train)} | val={len(val)} | test={len(test)}")
    diff_counts = defaultdict(int)
    for r in test:
        diff_counts[r["difficulty"]] += 1
    print(f"Test diffs : {dict(diff_counts)}\n")

    # ── Training ─────────────────────────────────────────────────────────
    train_results: dict[str, dict] = {}

    if not skip_training:
        print("Starting fine-tuning...\n")
        if sequential:
            for slug in active:
                train_results[slug] = train_model.remote(slug)
        else:
            handles = {slug: train_model.spawn(slug) for slug in active}
            print(f"Spawned {len(handles)} training containers in parallel.")
            for slug, h in handles.items():
                print(f"  Waiting for {slug}...")
                train_results[slug] = h.get()

        print("\nTraining summary:")
        for slug, r in train_results.items():
            print(f"  {slug}: "
                  f"loss={r.get('final_train_loss', '?')!r} "
                  f"eval_loss={r.get('final_eval_loss', '?')!r} "
                  f"time={r.get('train_time_min', 0):.0f} min")
    else:
        print("Skipping training (--skip-training). Using existing adapters.\n")

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\nStarting evaluation (baseline + fine-tuned for each model)...\n")
    eval_results: dict[tuple, dict] = {}

    if sequential:
        for slug in active:
            eval_results[(slug, False)] = evaluate_model.remote(slug, False)
            eval_results[(slug, True)]  = evaluate_model.remote(slug, True)
    else:
        handles = {}
        for slug in active:
            handles[(slug, False)] = evaluate_model.spawn(slug, False)
            handles[(slug, True)]  = evaluate_model.spawn(slug, True)
        print(f"Spawned {len(handles)} evaluation containers in parallel.\n")
        for key, h in handles.items():
            print(f"  Waiting for {key[0]} ({'FT' if key[1] else 'base'})...")
            eval_results[key] = h.get()

    # ── Print results ─────────────────────────────────────────────────────

    def pct(v):
        return f"{v:.1%}" if v is not None else "n/a"

    def delta(base_acc, ft_acc):
        if base_acc is None or ft_acc is None:
            return "n/a"
        d = ft_acc - base_acc
        return f"{d:+.1%}"

    print("\n\n" + "=" * 95)
    print("FINE-TUNING COMPARISON")
    print("=" * 95 + "\n")

    # Main table
    rows = []
    for slug in active:
        base = eval_results.get((slug, False), {})
        ft   = eval_results.get((slug, True),  {})
        tr   = train_results.get(slug, {})

        rows.append([
            slug,
            MODEL_PARAMS.get(slug, "?"),
            pct(base.get("overall_accuracy")),
            pct(ft.get("overall_accuracy")),
            delta(base.get("overall_accuracy"), ft.get("overall_accuracy")),
            pct(ft.get("format_pct")),
            pct(ft.get("by_diff", {}).get("easy")),
            pct(ft.get("by_diff", {}).get("medium")),
            pct(ft.get("by_diff", {}).get("hard")),
            f"{ft.get('tokens_per_sec', 0):.0f}",
            f"{ft.get('adapter_mb', 0):.1f}",
            f"{tr.get('train_time_min', 0):.0f}",
            f"{tr.get('final_train_loss') or 'n/a'!r}",
        ])

    main_headers = [
        "Model", "Params",
        "Base Acc", "FT Acc", "Delta",
        "Format%",
        "Easy", "Med", "Hard",
        "Tok/s", "Adapter(MB)", "Train(min)", "Loss",
    ]
    print(tabulate(rows, headers=main_headers, tablefmt="github"))

    # Per-difficulty breakdown (absolute values for each model)
    print("\n\n--- Accuracy by Difficulty ---\n")
    diff_rows = []
    for slug in active:
        base = eval_results.get((slug, False), {})
        ft   = eval_results.get((slug, True),  {})
        diff_rows.append([
            slug,
            pct(base.get("by_diff", {}).get("easy")),
            pct(ft.get("by_diff", {}).get("easy")),
            pct(base.get("by_diff", {}).get("medium")),
            pct(ft.get("by_diff", {}).get("medium")),
            pct(base.get("by_diff", {}).get("hard")),
            pct(ft.get("by_diff", {}).get("hard")),
        ])
    print(tabulate(
        diff_rows,
        headers=["Model", "Easy(base)", "Easy(FT)", "Med(base)", "Med(FT)", "Hard(base)", "Hard(FT)"],
        tablefmt="github",
    ))

    # Per-category breakdown (fine-tuned only)
    print("\n\n--- Accuracy by Category (fine-tuned) ---\n")
    all_cats: set[str] = set()
    for slug in active:
        ft = eval_results.get((slug, True), {})
        all_cats.update(ft.get("by_cat", {}).keys())

    cat_rows = []
    for cat in sorted(all_cats):
        row = [cat]
        for slug in active:
            ft = eval_results.get((slug, True), {})
            row.append(pct(ft.get("by_cat", {}).get(cat)))
        cat_rows.append(row)
    print(tabulate(cat_rows, headers=["Category"] + active, tablefmt="github"))

    # Speed & size summary
    print("\n\n--- Inference Speed & Model Size ---\n")
    size_rows = []
    for slug in active:
        ft  = eval_results.get((slug, True), {})
        tr  = train_results.get(slug, {})
        base_params = tr.get("total_params",     0)
        lora_params = tr.get("trainable_params", 0)
        size_rows.append([
            slug,
            MODEL_PARAMS.get(slug, "?"),
            f"{base_params/1e6:.0f}M" if base_params else "n/a",
            f"{lora_params/1e6:.2f}M" if lora_params else "n/a",
            f"{lora_params/max(base_params,1)*100:.2f}%" if base_params else "n/a",
            f"{ft.get('adapter_mb', 0):.1f} MB",
            f"{ft.get('tokens_per_sec', 0):.0f} tok/s",
        ])
    print(tabulate(
        size_rows,
        headers=["Model", "Size", "Base Params", "LoRA Params", "LoRA%", "Adapter(disk)", "Speed"],
        tablefmt="github",
    ))

    # Save results to local file
    out = Path("finetune_results.txt")
    with out.open("w", encoding="utf-8") as f:
        f.write(tabulate(rows, headers=main_headers, tablefmt="github"))
        f.write("\n\n")
        f.write(tabulate(diff_rows,
                         headers=["Model", "Easy(base)", "Easy(FT)", "Med(base)",
                                  "Med(FT)", "Hard(base)", "Hard(FT)"],
                         tablefmt="github"))
        f.write("\n\n")
        f.write(tabulate(cat_rows, headers=["Category"] + active, tablefmt="github"))
        f.write("\n\n")
        f.write(tabulate(size_rows,
                         headers=["Model", "Size", "Base Params", "LoRA Params",
                                  "LoRA%", "Adapter(disk)", "Speed"],
                         tablefmt="github"))

    print(f"\nFull results saved to {out.resolve()}")
