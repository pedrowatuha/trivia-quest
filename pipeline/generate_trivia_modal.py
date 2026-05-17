"""
Modal app — trivia question generator from Wikipedia snippets.

Uses Qwen2.5-7B-Instruct (open-weights, no API key needed) via vLLM on
Modal's GPU infrastructure.  For each snippet it generates three questions
(easy, medium, hard), each with four multiple-choice alternatives.

Setup (one-time):
    pip install modal
    modal setup          # authenticate with your Modal account

Run:
    modal run generate_trivia_modal.py
    modal run generate_trivia_modal.py --input-file wikipedia_snippets_1000_clean.jsonl \
        --output-file trivia_dataset.jsonl --limit 50
"""

import json
import random
import re
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image & model config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GPU_CONFIG = "A10G"          # 24 GB VRAM — comfortably fits 7B in fp16
CACHE_DIR = "/model-cache"   # mounted Modal Volume path

app = modal.App("trivia-generator")

volume = modal.Volume.from_name("trivia-model-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.6",
        "huggingface_hub[hf_transfer]>=0.23",
        "transformers>=4.45",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": CACHE_DIR,
    })
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert trivia question writer. Given a Wikipedia snippet you must \
create one multiple-choice trivia question.

Return ONLY a JSON object with exactly these keys — no markdown, no extra text:

{
  "question": "<one clear question ending with '?'>",
  "correct_answer": "<short correct answer>",
  "wrong_answers": ["<distractor 1>", "<distractor 2>", "<distractor 3>"],
  "explanation": "<one sentence citing the snippet that justifies the answer>"
}

Difficulty rules
----------------
EASY   – Most prominent fact directly stated (a name, place, or definition).
         Wrong answers are related but obviously different.
MEDIUM – Requires combining two facts or a small inference from the snippet.
         Wrong answers are plausible and from the same domain.
HARD   – Tests a specific year, lesser-known name, or subtle distinction.
         Wrong answers are highly plausible — close but wrong.

Additional rules
----------------
- Question must be answerable solely from the snippet provided.
- All four choices must be the same grammatical type (all names, all years, …).
- Do NOT start with "According to the snippet" or "Based on the text".\
"""


def _user_message(record: dict, difficulty: str) -> str:
    snippets = " ".join(record["source_snippets"])
    return (
        f"Category: {record['category']}\n"
        f"Article: {record['source_title']}\n"
        f"Difficulty: {difficulty.upper()}\n\n"
        f"Snippet:\n{snippets}\n\n"
        f"Generate a {difficulty} trivia question about this content."
    )


def _apply_chat_template(tokenizer, record: dict, difficulty: str) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": _user_message(record, difficulty)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Modal class — model loaded once per container
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
class TriviaGenerator:

    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"Loading {MODEL_NAME} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
        )
        self.llm = LLM(
            model=MODEL_NAME,
            download_dir=CACHE_DIR,
            dtype="auto",
            max_model_len=4096,
            enforce_eager=True,   # skip torch.compile to avoid AOT cache serialization bug in vLLM 0.20
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        print("Model ready.")

    @modal.method()
    def generate_for_snippet(self, record: dict) -> list[dict]:
        """Generate easy, medium and hard questions for a single snippet record."""
        difficulties = ("easy", "medium", "hard")

        # Build all three prompts at once — vLLM batches them efficiently
        prompts = [
            _apply_chat_template(self.tokenizer, record, d)
            for d in difficulties
        ]

        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for difficulty, output in zip(difficulties, outputs):
            text = output.outputs[0].text.strip()
            parsed = _parse_output(text, record, difficulty)
            if parsed:
                results.append(_build_record(record, difficulty, parsed))

        return results


# ---------------------------------------------------------------------------
# Parsing helpers (run locally and inside containers)
# ---------------------------------------------------------------------------

def _parse_output(text: str, record: dict, difficulty: str) -> dict | None:
    """Strip markdown fences, parse JSON, validate shape."""
    # Remove ```json ... ``` if the model adds them
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

    # Sometimes models emit text before the JSON — find the first '{'
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        print(f"[WARN] No JSON found for '{record['source_title']}' ({difficulty})")
        return None

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        print(f"[WARN] JSON parse error for '{record['source_title']}' ({difficulty}): {exc}")
        return None

    required = {"question", "correct_answer", "wrong_answers", "explanation"}
    if not required.issubset(data):
        print(f"[WARN] Missing keys for '{record['source_title']}' ({difficulty}): "
              f"{required - data.keys()}")
        return None

    if not isinstance(data["wrong_answers"], list) or len(data["wrong_answers"]) < 3:
        print(f"[WARN] Bad wrong_answers for '{record['source_title']}' ({difficulty})")
        return None

    return data


def _build_record(record: dict, difficulty: str, data: dict) -> dict:
    """Shuffle alternatives, assign letters, build the output record."""
    correct = data["correct_answer"]
    choices = [correct] + data["wrong_answers"][:3]
    random.shuffle(choices)

    letter_map    = {letter: choice for letter, choice in zip("ABCD", choices)}
    correct_letter = next(k for k, v in letter_map.items() if v == correct)

    # Fine-tuning message pair
    user_content      = _user_message(record, difficulty)
    assistant_content = _format_answer(data["question"], letter_map, correct_letter)

    return {
        # provenance
        "category":        record["category"],
        "source_title":    record["source_title"],
        "source_url":      record["source_url"],
        "source_snippets": record["source_snippets"],
        # trivia
        "difficulty":      difficulty,
        "question":        data["question"],
        "alternatives":    letter_map,
        "correct_letter":  correct_letter,
        "correct_answer":  correct,
        "explanation":     data["explanation"],
        # fine-tuning (chat-completion format)
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def _format_answer(question: str, letter_map: dict, correct_letter: str) -> str:
    lines = [question]
    for letter in "ABCD":
        lines.append(f"{letter}) {letter_map[letter]}")
    lines.append(f"\nAnswer: {correct_letter}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    input_file: str = "wikipedia_snippets_1000_clean.jsonl",
    output_file: str = "trivia_dataset.jsonl",
    limit: int = 0,
):
    """
    Parameters
    ----------
    input_file  Path to the Wikipedia snippets JSONL (local file).
    output_file Path where the trivia JSONL will be written (local file).
    limit       Process only the first N records (0 = all).
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path.resolve()}")

    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if limit:
        records = records[:limit]

    total = len(records)
    print(f"Snippets to process  : {total}")
    print(f"Questions to generate: {total * 3}  (easy + medium + hard)")
    print(f"Model                : {MODEL_NAME}")
    print(f"Output               : {output_file}\n")

    generator = TriviaGenerator()

    written = 0
    skipped = 0

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as out:
        for batch in generator.generate_for_snippet.map(records, order_outputs=False):
            if not batch:
                skipped += 1
                continue
            for q in batch:
                out.write(json.dumps(q, ensure_ascii=False) + "\n")
                written += 1

    print(f"\nDone.")
    print(f"  Questions written : {written}")
    print(f"  Snippets skipped  : {skipped}")
    print(f"  Output file       : {output_path.resolve()}")

    # Show one example
    if written:
        with output_path.open(encoding="utf-8") as f:
            ex = json.loads(f.readline())
        print(f"\n--- Example ({ex['difficulty'].upper()}) ---")
        print(f"  Q: {ex['question']}")
        for letter in "ABCD":
            mark = " <-- CORRECT" if letter == ex["correct_letter"] else ""
            print(f"    {letter}) {ex['alternatives'][letter]}{mark}")
        print(f"  Explanation: {ex['explanation']}")
