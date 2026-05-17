"""
generate_report.py

Runs qwen-0.5b on 10 test samples and saves an HTML report showing:
  - the Wikipedia snippet used as source
  - the model-generated trivia question
  - the four alternatives
  - the answer

Usage:
    python generate_report.py
    python generate_report.py --models qwen-0.5b smollm-1.7b --n 10 --output report.html
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
MAX_NEW_TOKENS = 200

# ── Dataset ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def split_dataset(records, seed=DATA_SEED):
    by_title = defaultdict(list)
    for r in records:
        by_title[r["source_title"]].append(r)
    titles = list(by_title.keys())
    random.Random(seed).shuffle(titles)
    n = len(titles)
    test_titles = set(titles[:int(n * 0.10)])
    val_titles  = set(titles[int(n * 0.10):int(n * 0.20)])
    train_titles = set(titles[int(n * 0.20):])
    return (
        [r for r in records if r["source_title"] in train_titles],
        [r for r in records if r["source_title"] in val_titles],
        [r for r in records if r["source_title"] in test_titles],
    )

def pick_samples(test_records, n, seed=DATA_SEED):
    by_diff = defaultdict(list)
    for r in test_records:
        by_diff[r["difficulty"]].append(r)
    rng = random.Random(seed)
    samples = []
    per_diff = n // 3
    for i, d in enumerate(["easy", "medium", "hard"]):
        k = per_diff + (1 if i < n % 3 else 0)
        samples.extend(rng.sample(by_diff[d], min(k, len(by_diff[d]))))
    rng.shuffle(samples)
    return samples[:n]

# ── Inference ─────────────────────────────────────────────────────────────────

def load_model(model_id, adapter_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path), device_map={"": device})
    model.eval()
    return model, tok, device

def generate(model, tok, record, device):
    prompt = tok.apply_chat_template(
        [record["messages"][0]], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=False,
        )
    elapsed = time.time() - t0
    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip(), elapsed

# ── HTML ─────────────────────────────────────────────────────────────────────

DIFF_COLOR = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trivia Model Report</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; padding: 24px; color: #222; }
  h1 { text-align: center; color: #1a1a2e; margin-bottom: 8px; }
  .subtitle { text-align: center; color: #666; margin-bottom: 32px; font-size: 14px; }
  .model-section { margin-bottom: 48px; }
  .model-title { font-size: 22px; font-weight: bold; color: #fff; background: #1a1a2e;
                 padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; }
  .card { background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
          margin-bottom: 24px; overflow: hidden; }
  .card-header { padding: 14px 20px; display: flex; align-items: center; gap: 12px;
                 border-bottom: 1px solid #eee; }
  .badge { font-size: 11px; font-weight: bold; padding: 3px 10px; border-radius: 20px;
           color: #fff; text-transform: uppercase; }
  .source-title { font-size: 13px; color: #888; margin-left: auto; font-style: italic; }
  .card-body { padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .snippet-box { background: #f8f9fa; border-left: 4px solid #6c757d; padding: 14px 16px;
                 border-radius: 0 8px 8px 0; font-size: 13px; line-height: 1.6; color: #444; }
  .snippet-label { font-size: 11px; font-weight: bold; color: #999; text-transform: uppercase;
                   letter-spacing: 1px; margin-bottom: 8px; }
  .qa-box { display: flex; flex-direction: column; gap: 10px; }
  .question { font-weight: 600; font-size: 15px; line-height: 1.5; color: #1a1a2e; }
  .alternatives { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 6px; }
  .alternatives li { padding: 8px 12px; border-radius: 6px; font-size: 14px;
                     border: 1px solid #e0e0e0; display: flex; align-items: center; gap: 8px; }
  .alt-letter { font-weight: bold; min-width: 20px; }
  .correct { background: #e8f8f0; border-color: #2ecc71 !important; }
  .answer-line { font-size: 13px; color: #888; margin-top: 4px; }
  .answer-line strong { color: #2ecc71; }
  .time-label { font-size: 11px; color: #bbb; text-align: right; margin-top: auto; }
  @media (max-width: 700px) { .card-body { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Trivia Fine-tune Report</h1>
<p class="subtitle">Model-generated questions from Wikipedia snippets &mdash; test split</p>
"""

HTML_FOOTER = "</body></html>\n"


def parse_generated(text):
    """Extract question, alternatives dict, and answer letter from model output."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    question = ""
    alts = {}
    answer = None

    for line in lines:
        m = re.match(r"^([A-D])[).]\s*(.+)$", line)
        if m:
            alts[m.group(1)] = m.group(2)
            continue
        m = re.search(r"[Aa]nswer\s*:\s*([A-D])", line)
        if m:
            answer = m.group(1).upper()
            continue
        if not alts and not answer:
            if question:
                question += " " + line
            else:
                question = line

    return question.strip(), alts, answer


def render_card(idx, record, generated, elapsed):
    diff = record["difficulty"]
    color = DIFF_COLOR.get(diff, "#888")
    snippet = " ".join(record["source_snippets"])

    question, alts, answer = parse_generated(generated)

    alts_html = ""
    for letter in "ABCD":
        text = alts.get(letter, "")
        is_correct = (letter == answer)
        cls = ' class="correct"' if is_correct else ""
        tick = " &#10003;" if is_correct else ""
        alts_html += f'<li{cls}><span class="alt-letter">{letter})</span>{text}{tick}</li>\n'

    answer_display = f"Answer: <strong>{answer}</strong> &mdash; {alts.get(answer, '')}" if answer else "Answer: —"

    return f"""
<div class="card">
  <div class="card-header">
    <span style="font-weight:bold;color:#555">#{idx}</span>
    <span class="badge" style="background:{color}">{diff}</span>
    <span class="source-title">{record['source_title']} &middot; {record['category']}</span>
  </div>
  <div class="card-body">
    <div class="snippet-box">
      <div class="snippet-label">Wikipedia Snippet</div>
      {snippet}
    </div>
    <div class="qa-box">
      <div class="question">{question}</div>
      <ul class="alternatives">{alts_html}</ul>
      <div class="answer-line">{answer_display}</div>
      <div class="time-label">generated in {elapsed:.1f}s</div>
    </div>
  </div>
</div>
"""


def render_model_section(slug, model_id, cards_html):
    return f"""
<div class="model-section">
  <div class="model-title">Model: {slug} &nbsp;<span style="font-weight:normal;font-size:14px;opacity:.7">({model_id})</span></div>
  {cards_html}
</div>
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-0.5b"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--output", default="report.html")
    args = parser.parse_args()

    print("Loading dataset...")
    records = load_jsonl(DATASET_PATH)
    _, _, test = split_dataset(records)
    samples = pick_samples(test, args.n)
    print(f"  {len(samples)} test samples selected\n")

    body = ""

    for slug in args.models:
        if slug not in MODELS:
            print(f"Unknown model: {slug}")
            continue
        model_id = MODELS[slug]
        adapter_path = ADAPTER_DIR / slug
        if not adapter_path.exists():
            print(f"Adapter not found: {adapter_path}")
            continue

        print(f"Loading {slug}...")
        model, tok, device = load_model(model_id, adapter_path)

        cards = ""
        for i, record in enumerate(samples, 1):
            print(f"  [{i}/{len(samples)}] {record['source_title']} ({record['difficulty']})")
            generated, elapsed = generate(model, tok, record, device)
            cards += render_card(i, record, generated, elapsed)

        body += render_model_section(slug, model_id, cards)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    html = HTML_HEADER + body + HTML_FOOTER
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
