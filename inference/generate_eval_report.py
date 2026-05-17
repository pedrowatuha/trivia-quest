"""
generate_eval_report.py

Generates 100 trivia questions using the fine-tuned model, then produces a
self-contained HTML evaluation form.  Your ratings are exported as JSONL
ready for DPO / reward-model training.

Usage:
    python generate_eval_report.py
    python generate_eval_report.py --model smollm-1.7b --n 100 --output eval.html
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

DIFFICULTIES = ["easy", "medium", "hard"]

USER_PROMPT_TMPL = (
    "Category: {category}\n"
    "Article: {source_title}\n"
    "Difficulty: {difficulty}\n\n"
    "Snippet:\n{snippet}\n\n"
    "Generate a {difficulty} trivia question about this content.\n"
    "IMPORTANT: The question must be self-contained. Do NOT say 'according to the snippet', "
    "'based on the text', 'in the passage', or any phrase that references a source text. "
    "Ask about the fact directly as if it were common knowledge."
)

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
    test_titles  = set(titles[:int(n * 0.10)])
    val_titles   = set(titles[int(n * 0.10):int(n * 0.20)])
    train_titles = set(titles[int(n * 0.20):])
    return (
        [r for r in records if r["source_title"] in train_titles],
        [r for r in records if r["source_title"] in val_titles],
        [r for r in records if r["source_title"] in test_titles],
    )

def pick_samples(records, n, seed=DATA_SEED):
    by_diff = defaultdict(list)
    for r in records:
        by_diff[r["difficulty"]].append(r)
    rng = random.Random(seed)
    samples = []
    per = n // 3
    for i, d in enumerate(["easy", "medium", "hard"]):
        k = per + (1 if i < n % 3 else 0)
        pool = by_diff[d]
        samples.extend(rng.sample(pool, min(k, len(pool))))
    rng.shuffle(samples)
    return samples[:n]

# ── Inference ─────────────────────────────────────────────────────────────────

def base_model_id_from_adapter(adapter_path: Path, slug: str) -> str:
    """Read base_model_name_or_path from adapter_config.json, fall back to MODELS dict."""
    cfg = adapter_path / "adapter_config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        if "base_model_name_or_path" in data:
            return data["base_model_name_or_path"]
    return MODELS.get(slug, MODELS["qwen-0.5b"])

def snippet_to_records(raw: list[dict], n: int, seed: int = DATA_SEED) -> list[dict]:
    """Turn raw snippet dicts into pseudo-records with a random difficulty each."""
    rng = random.Random(seed)
    records = []
    # assign difficulties evenly
    per = n // 3
    diff_pool = (["easy"] * per + ["medium"] * per + ["hard"] * (n - 2 * per))
    rng.shuffle(diff_pool)
    for raw_rec, diff in zip(raw[:n], diff_pool):
        snippet_text = " ".join(raw_rec["source_snippets"])
        user_content = USER_PROMPT_TMPL.format(
            category=raw_rec["category"],
            source_title=raw_rec["source_title"],
            difficulty=diff,
            snippet=snippet_text,
        )
        records.append({
            "source_title":    raw_rec["source_title"],
            "category":        raw_rec["category"],
            "difficulty":      diff,
            "source_snippets": raw_rec["source_snippets"],
            "messages": [{"role": "user", "content": user_content}],
        })
    return records

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

BANNED_PHRASES = [
    "according to the snippet", "according to the passage", "according to the text",
    "based on the snippet", "based on the passage", "based on the text",
    "in the snippet", "in the passage", "the snippet states", "the text states",
    "as mentioned in", "as stated in", "the passage states", "the extract",
]

def has_banned_phrase(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in BANNED_PHRASES)

def run_inference(model, tok, record, device, max_retries: int = 4):
    prompt = tok.apply_chat_template(
        [record["messages"][0]], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()

    # Alternate between greedy and slightly sampled on retries to get variation
    for attempt in range(max_retries):
        do_sample = attempt > 0
        temp      = 0.7 + attempt * 0.1
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=temp if do_sample else 1.0,
                pad_token_id=tok.pad_token_id, use_cache=False,
            )
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
        text = tok.decode(new_ids, skip_special_tokens=True).strip()
        if not has_banned_phrase(text):
            break
        if attempt < max_retries - 1:
            print(f"      [retry {attempt+1}] banned phrase detected, regenerating...")

    elapsed = time.time() - t0
    return text, round(elapsed, 1)

def parse_generated(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    question, alts, answer = "", {}, None
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
            question = (question + " " + line).strip()
    return question, alts, answer

# ── HTML generation ───────────────────────────────────────────────────────────

def build_html(entries, model_slug, model_id):
    data_json = json.dumps(entries, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trivia Evaluation — {model_slug}</title>
<style>
  :root {{
    --easy:    #27ae60;
    --medium:  #e67e22;
    --hard:    #e74c3c;
    --accent:  #3498db;
    --bg:      #f0f2f5;
    --card:    #ffffff;
    --border:  #e0e4ea;
    --text:    #1a1a2e;
    --muted:   #888;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 24px; }}

  /* ── Top bar ── */
  .topbar {{ position: sticky; top: 0; z-index: 100; background: var(--text); color: #fff;
             display: flex; align-items: center; gap: 16px; padding: 12px 20px;
             border-radius: 10px; margin-bottom: 28px; box-shadow: 0 4px 12px rgba(0,0,0,.2); }}
  .topbar h1 {{ font-size: 16px; flex: 1; }}
  .progress-wrap {{ flex: 2; background: rgba(255,255,255,.15); border-radius: 20px; height: 10px; overflow: hidden; }}
  .progress-bar  {{ height: 100%; background: #2ecc71; border-radius: 20px; transition: width .3s; }}
  .progress-text {{ font-size: 13px; min-width: 80px; text-align: right; }}
  .export-btn {{ background: #2ecc71; color: #fff; border: none; padding: 9px 20px;
                 border-radius: 7px; cursor: pointer; font-weight: bold; font-size: 14px; }}
  .export-btn:hover {{ background: #27ae60; }}

  /* ── Cards ── */
  .card {{ background: var(--card); border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.07);
           margin-bottom: 24px; overflow: hidden; border: 2px solid transparent; transition: border .2s; }}
  .card.rated {{ border-color: #2ecc71; }}
  .card-header {{ padding: 12px 18px; display: flex; align-items: center; gap: 10px;
                  border-bottom: 1px solid var(--border); }}
  .idx {{ font-size: 13px; font-weight: bold; color: var(--muted); min-width: 28px; }}
  .badge {{ font-size: 11px; font-weight: bold; padding: 3px 10px; border-radius: 20px;
            color: #fff; text-transform: uppercase; letter-spacing: .5px; }}
  .badge.easy   {{ background: var(--easy);   }}
  .badge.medium {{ background: var(--medium); }}
  .badge.hard   {{ background: var(--hard);   }}
  .source {{ font-size: 12px; color: var(--muted); margin-left: auto; font-style: italic; }}

  .card-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}

  /* snippet */
  .snippet-col {{ padding: 16px 18px; border-right: 1px solid var(--border); }}
  .col-label {{ font-size: 10px; font-weight: bold; text-transform: uppercase;
                letter-spacing: 1px; color: var(--muted); margin-bottom: 8px; }}
  .snippet-text {{ font-size: 13px; line-height: 1.65; color: #444; }}

  /* generated Q&A */
  .qa-col {{ padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; }}
  .gen-question {{ font-weight: 600; font-size: 15px; line-height: 1.5; }}
  .alts {{ list-style: none; display: flex; flex-direction: column; gap: 5px; }}
  .alts li {{ padding: 7px 11px; border-radius: 6px; font-size: 13px;
              border: 1px solid var(--border); display: flex; gap: 8px; }}
  .alts li.correct {{ background: #e8f8f0; border-color: #2ecc71; }}
  .alt-letter {{ font-weight: bold; }}
  .gen-answer {{ font-size: 12px; color: var(--muted); }}
  .gen-answer strong {{ color: var(--easy); }}

  /* eval section */
  .eval-row {{ border-top: 1px solid var(--border); padding: 14px 18px;
               display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }}
  .eval-group {{ display: flex; flex-direction: column; gap: 6px; }}
  .eval-label {{ font-size: 11px; font-weight: bold; text-transform: uppercase;
                 letter-spacing: .8px; color: var(--muted); }}

  /* star rating */
  .stars {{ display: flex; flex-direction: row-reverse; gap: 3px; }}
  .stars input {{ display: none; }}
  .stars label {{ font-size: 24px; color: #ccc; cursor: pointer; line-height: 1; transition: color .15s; }}
  .stars input:checked ~ label,
  .stars label:hover,
  .stars label:hover ~ label {{ color: #f1c40f; }}

  /* toggle buttons */
  .toggles {{ display: flex; gap: 6px; }}
  .toggles input {{ display: none; }}
  .toggles label {{ padding: 5px 14px; border-radius: 20px; border: 1.5px solid var(--border);
                    font-size: 12px; font-weight: bold; cursor: pointer; color: var(--muted);
                    transition: all .15s; }}
  .toggles input:checked + label {{ color: #fff; border-color: transparent; }}
  .diff-easy:checked   + label {{ background: var(--easy); }}
  .diff-medium:checked + label {{ background: var(--medium); }}
  .diff-hard:checked   + label {{ background: var(--hard); }}
  .fit-yes:checked     + label {{ background: #2ecc71; }}
  .fit-partial:checked + label {{ background: #e67e22; }}
  .fit-no:checked      + label {{ background: #e74c3c; }}

  @media (max-width: 680px) {{ .card-body {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="topbar">
  <h1>Trivia Eval &mdash; {model_slug}</h1>
  <div class="progress-wrap"><div class="progress-bar" id="pbar" style="width:0%"></div></div>
  <span class="progress-text" id="ptext">0 / {len(entries)} rated</span>
  <button class="export-btn" onclick="exportRatings()">Export JSONL</button>
</div>

<div id="cards"></div>

<div style="text-align:center;margin:32px 0">
  <button class="export-btn" style="font-size:16px;padding:12px 32px" onclick="exportRatings()">
    Export Ratings as JSONL
  </button>
</div>

<script>
const DATA = {data_json};

function render() {{
  const container = document.getElementById('cards');
  DATA.forEach((e, i) => {{
    const id = `q${{i}}`;
    const altsHtml = Object.entries(e.alts).map(([l, t]) => {{
      const cls = l === e.answer ? ' class="correct"' : '';
      const tick = l === e.answer ? ' &#10003;' : '';
      return `<li${{cls}}><span class="alt-letter">${{l}})</span>${{t}}${{tick}}</li>`;
    }}).join('');
    const answerLine = e.answer
      ? `Answer: <strong>${{e.answer}}</strong> &mdash; ${{e.alts[e.answer] ?? ''}}`
      : '<em>no answer parsed</em>';

    container.innerHTML += `
<div class="card" id="card${{i}}">
  <div class="card-header">
    <span class="idx">#${{i+1}}</span>
    <span class="badge ${{e.difficulty}}">${{e.difficulty}}</span>
    <span class="source">${{e.source_title}} &middot; ${{e.category}}</span>
  </div>
  <div class="card-body">
    <div class="snippet-col">
      <div class="col-label">Wikipedia Snippet</div>
      <div class="snippet-text">${{e.snippet}}</div>
    </div>
    <div class="qa-col">
      <div class="col-label">Generated Question</div>
      <div class="gen-question">${{e.question || '<em>(could not parse question)</em>'}}</div>
      <ul class="alts">${{altsHtml}}</ul>
      <div class="gen-answer">${{answerLine}}</div>
    </div>
  </div>
  <div class="eval-row">

    <div class="eval-group">
      <div class="eval-label">Question Quality</div>
      <div class="stars" id="stars${{i}}">
        ${{[5,4,3,2,1].map(v => `
          <input type="radio" name="quality_${{i}}" id="q${{i}}s${{v}}" value="${{v}}" onchange="update()">
          <label for="q${{i}}s${{v}}" title="${{v}} star${{v>1?'s':''}}">&starf;</label>
        `).join('')}}
      </div>
    </div>

    <div class="eval-group">
      <div class="eval-label">Real Difficulty</div>
      <div class="toggles">
        <input class="diff-easy"   type="radio" name="diff_${{i}}" id="de${{i}}" value="easy"   onchange="update()">
        <label for="de${{i}}">Easy</label>
        <input class="diff-medium" type="radio" name="diff_${{i}}" id="dm${{i}}" value="medium" onchange="update()">
        <label for="dm${{i}}">Medium</label>
        <input class="diff-hard"   type="radio" name="diff_${{i}}" id="dh${{i}}" value="hard"   onchange="update()">
        <label for="dh${{i}}">Hard</label>
      </div>
    </div>

    <div class="eval-group">
      <div class="eval-label">Fits Category?</div>
      <div class="toggles">
        <input class="fit-yes"     type="radio" name="fit_${{i}}" id="fy${{i}}" value="yes"     onchange="update()">
        <label for="fy${{i}}">Yes</label>
        <input class="fit-partial" type="radio" name="fit_${{i}}" id="fp${{i}}" value="partial" onchange="update()">
        <label for="fp${{i}}">Partial</label>
        <input class="fit-no"      type="radio" name="fit_${{i}}" id="fn${{i}}" value="no"      onchange="update()">
        <label for="fn${{i}}">No</label>
      </div>
    </div>

  </div>
</div>`;
  }});
}}

function getVal(name) {{
  const el = document.querySelector(`input[name="${{name}}"]:checked`);
  return el ? el.value : null;
}}

function isRated(i) {{
  return getVal(`quality_${{i}}`) && getVal(`diff_${{i}}`) && getVal(`fit_${{i}}`);
}}

function update() {{
  let rated = 0;
  DATA.forEach((_, i) => {{
    if (isRated(i)) {{
      rated++;
      document.getElementById(`card${{i}}`).classList.add('rated');
    }}
  }});
  const pct = Math.round(rated / DATA.length * 100);
  document.getElementById('pbar').style.width = pct + '%';
  document.getElementById('ptext').textContent = rated + ' / ' + DATA.length + ' rated';
}}

function exportRatings() {{
  const lines = DATA.map((e, i) => JSON.stringify({{
    id:                  i,
    model:               "{model_slug}",
    source_title:        e.source_title,
    category:            e.category,
    original_difficulty: e.difficulty,
    snippet:             e.snippet,
    prompt:              e.prompt,
    generated:           e.generated,
    question:            e.question,
    alts:                e.alts,
    answer:              e.answer,
    ratings: {{
      quality:        getVal(`quality_${{i}}`),
      real_difficulty: getVal(`diff_${{i}}`),
      category_fit:   getVal(`fit_${{i}}`),
    }}
  }}));
  const blob = new Blob([lines.join('\\n')], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'ratings_{model_slug}.jsonl';
  a.click();
}}

render();
</script>
</body>
</html>
"""

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="qwen-0.5b")
    parser.add_argument("--n",        type=int, default=100)
    parser.add_argument("--output",   default="eval_report.html")
    parser.add_argument("--snippets", default=None,
                        help="Path to raw Wikipedia snippets JSONL (from fetch_new_topics.py). "
                             "If omitted, samples from the existing trivia dataset.")
    parser.add_argument("--split",    default="all", choices=["test", "val", "all"])
    args = parser.parse_args()

    slug         = args.model
    adapter_path = ADAPTER_DIR / slug
    if not adapter_path.exists():
        print(f"Adapter not found: {adapter_path}")
        return

    model_id = base_model_id_from_adapter(adapter_path, slug)
    print(f"Base model: {model_id}")

    if args.snippets:
        print(f"Loading raw snippets from {args.snippets} ...")
        raw = load_jsonl(Path(args.snippets))
        samples = snippet_to_records(raw, args.n)
        print(f"  Snippets: {len(raw)}  |  Sampling: {len(samples)}\n")
    else:
        print("Loading trivia dataset...")
        records = load_jsonl(DATASET_PATH)
        train, val, test = split_dataset(records)
        pool = {"test": test, "val": val, "all": records}[args.split]
        samples = pick_samples(pool, args.n)
        print(f"  Pool: {len(pool)} records  |  Sampling: {len(samples)}\n")

    print(f"Loading {slug}...")
    model, tok, device = load_model(model_id, adapter_path)

    entries = []
    for i, record in enumerate(samples, 1):
        print(f"  [{i:3d}/{len(samples)}] {record['source_title']} ({record['difficulty']})")
        generated, elapsed = run_inference(model, tok, record, device)
        question, alts, answer = parse_generated(generated)
        entries.append({
            "source_title": record["source_title"],
            "category":     record["category"],
            "difficulty":   record["difficulty"],
            "snippet":      " ".join(record["source_snippets"]),
            "prompt":       record["messages"][0]["content"],
            "generated":    generated,
            "question":     question,
            "alts":         alts,
            "answer":       answer,
            "elapsed":      elapsed,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    html = build_html(entries, slug, model_id)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"\nSaved: {args.output}  ({len(entries)} questions)")
    print("Open in your browser, rate each question, then click 'Export JSONL'.")


if __name__ == "__main__":
    main()
