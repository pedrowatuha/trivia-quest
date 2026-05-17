# 🧠 Trivia Quest

A local-first trivia game powered by a fine-tuned Qwen 2.5 0.5B + LoRA adapter.
Pick any topic — the app fetches a Wikipedia article, then a small language
model writes five original multiple-choice questions from the snippets.

![screen flow](https://img.shields.io/badge/python-3.10%2B-blue) ![framework](https://img.shields.io/badge/backend-FastAPI-009688) ![model](https://img.shields.io/badge/model-Qwen2.5--0.5B%20%2B%20LoRA-purple)

## Features

- **Any topic**: 16 suggested themes plus free-text input
- **80 / 20 snippet mix**: most questions come from the main article; 20% are
  flagged "🃏 Joker Questions" pulled from a linked Wikipedia page
- **Local inference**: runs on CPU; GPU optional via CUDA
- **5-question rounds** with live progress bar, score tracking, and a results
  screen with per-round emoji rating
- **No telemetry**: everything stays on your machine

## Quickstart

```bash
git clone https://github.com/<you>/trivia-quest.git
cd trivia-quest
python -m venv .venv && .venv\Scripts\activate          # Windows
# source .venv/bin/activate                              # macOS / Linux
pip install -r requirements.txt

python start_app.py
```

Open <http://localhost:8000> in your browser.

On the first run, the app downloads:

- The base model (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB) from Hugging Face
- The fine-tuned LoRA adapter (~79 MB) from
  `YOUR_HF_USERNAME/qwen-0.5b-trivia-lora`

Both are cached under `~/.cache/huggingface/` after the first download.

> **Note**: replace the placeholder above with the real Hugging Face repo
> hosting your adapter, or set the `TRIVIA_ADAPTER_REPO` environment variable.

## Project layout

```
app/                          Web app (FastAPI backend + single-page frontend)
  ├── main.py                 API routes + generation orchestration
  ├── model.py                LoRA adapter loading + inference
  ├── wikipedia.py            Topic search, article fetch, snippet selection
  └── static/index.html       Self-contained SPA (CSS + JS inline)

pipeline/                     Dataset construction & evaluation harness
  ├── build_wikipedia_snippet_dataset.py
  ├── fetch_new_topics.py
  ├── generate_trivia_modal.py
  └── finetune_eval_pipeline.py

inference/                    Fine-tuning scripts (SFT + DPO)
  ├── finetune_from_ratings.py
  ├── finetune_dpo.py
  ├── local_inference.py
  ├── generate_report.py
  └── generate_eval_report.py

start_app.py                  Plain `uvicorn` launcher (recommended)
launcher.py                   Double-clickable launcher that opens the browser
```

## How it works

1. **Topic search** — `wikipedia.py` queries the MediaWiki API for the best
   matching article and collects up to 60 linked page titles for "joker"
   questions.
2. **Snippet selection** — 80% of the time it picks ~3 consecutive sentences
   from the main extract, 20% it fetches a linked article and pulls a
   snippet from there.
3. **Question generation** — `model.py` formats a prompt with category,
   article title, difficulty and snippet, then samples from the LoRA-adapted
   Qwen model. Outputs are parsed into `{question, choices, answer_index}`
   and re-rolled if they include banned phrases like "according to the
   snippet".
4. **Frontend** — once all 5 questions are ready the SPA receives the full
   batch and runs the game locally — no per-question API calls.

## Train your own adapter

The full pipeline is under `pipeline/` and `inference/`. Roughly:

```bash
# 1. Build a Wikipedia snippet dataset
python pipeline/build_wikipedia_snippet_dataset.py

# 2. Generate trivia training data (uses Modal for distributed inference)
python pipeline/generate_trivia_modal.py

# 3. Fine-tune with LoRA (SFT then DPO)
python inference/finetune_from_ratings.py
python inference/finetune_dpo.py
```

Then upload your trained adapter to Hugging Face:

```bash
hf upload <your-username>/qwen-0.5b-trivia-lora adapters/qwen-0.5b-v3-dpo
```

and set `TRIVIA_ADAPTER_REPO=<your-username>/qwen-0.5b-trivia-lora`.

## Environment variables

| Variable | Purpose |
|---|---|
| `TRIVIA_ADAPTER_REPO` | Hugging Face repo to fetch the LoRA adapter from |
| `TRIVIA_FAKE_MODEL=1` | Skip model loading; serve dummy questions (useful for UI testing) |
| `HF_HUB_OFFLINE=1` | Force offline mode (requires local adapter + model) |

## License

MIT
