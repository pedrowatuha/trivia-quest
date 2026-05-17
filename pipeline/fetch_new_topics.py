"""
fetch_new_topics.py

Fetch 50 fresh Wikipedia article snippets that are NOT already in the
existing trivia dataset, and save them in the same JSONL format so they
can be fed directly into generate_eval_report.py.

Usage:
    python pipeline/fetch_new_topics.py
    python pipeline/fetch_new_topics.py --n 50 --out data/wikipedia_new_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

API_URL    = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TriviaDatasetBuilder/2.0 (educational; local user)"

CATEGORIES = [
    "Architecture", "Aviation", "Biology", "Chemistry", "Classical music",
    "Comics", "Cryptography", "Dance", "Ecology", "Economics",
    "Engineering", "Fashion", "Geology", "Health", "Law",
    "Linguistics", "Mathematics", "Medicine", "Mythology", "Neuroscience",
    "Nuclear physics", "Ocean", "Philosophy", "Photography", "Physics",
    "Psychiatry", "Psychology", "Religion", "Robotics", "Sociology",
    "Theater", "Urban planning", "Veterinary medicine", "Video game history",
    "Volcanology", "Astronomy", "Anthropology", "Archaeology", "Botany",
    "Cartography", "Climatology", "Criminology", "Demography", "Diplomacy",
    "Entomology", "Genetics", "Hydrology", "Immunology", "Journalism",
    "Meteorology",
]

BAD_PREFIXES = (
    "File:", "Help:", "Wikipedia:", "Category:", "Template:", "Talk:",
    "Portal:", "Special:", "Module:", "User:", "Draft:",
)

NAVIGATION = (
    "this article", "this section", "see also", "external links",
    "further reading", "references", "^ ", "citation needed",
    "clarification needed",
)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT


def api(params: dict, retries: int = 5) -> dict:
    params.setdefault("format", "json")
    delay = 2.0
    for attempt in range(retries):
        r = SESSION.get(API_URL, params=params, timeout=20)
        if r.status_code == 429:
            wait = delay * (2 ** attempt)
            print(f"    [rate-limited] waiting {wait:.0f}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {retries} retries")


def linked_titles(category: str, limit: int = 80) -> list[str]:
    """Get article titles linked from a category's Wikipedia page."""
    data = api({
        "action": "query", "titles": category,
        "prop": "links", "pllimit": limit, "plnamespace": 0,
    })
    pages = data.get("query", {}).get("pages", {})
    titles = []
    for page in pages.values():
        for link in page.get("links", []):
            t = link["title"]
            if not any(t.startswith(p) for p in BAD_PREFIXES):
                titles.append(t)
    return titles


def fetch_extract(title: str) -> str | None:
    data = api({
        "action": "query", "titles": title,
        "prop": "extracts", "exintro": True,
        "explaintext": True, "redirects": True,
    })
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "extract" in page and page["extract"].strip():
            return page["extract"]
    return None


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"==+[^=]+=+", "", text)
    return text.strip()


def sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", clean(text))
    return [s.strip() for s in raw if len(s.strip()) > 30]


def is_useful(s: str) -> bool:
    sl = s.lower()
    return not any(n in sl for n in NAVIGATION) and not s.startswith("^")


def pick_snippet(sents: list[str], n: int = 3) -> list[str]:
    useful = [s for s in sents if is_useful(s)]
    if len(useful) < 2:
        return []
    start = random.randint(0, max(0, len(useful) - n))
    return useful[start: start + n]


def make_record(title: str, category: str, snippet: list[str]) -> dict:
    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    return {
        "category":       category,
        "source_title":   title,
        "source_url":     url,
        "source_snippets": snippet,
    }


def load_existing_titles(path: Path) -> set[str]:
    if not path.exists():
        return set()
    titles = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                titles.add(json.loads(line)["source_title"])
            except Exception:
                pass
    return titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=50)
    parser.add_argument("--out", default="data/wikipedia_new_50.jsonl")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_titles(ROOT / "data" / "trivia_dataset.jsonl")
    print(f"Existing titles to avoid: {len(existing)}")

    rng = random.Random(args.seed)
    cats = list(CATEGORIES)
    rng.shuffle(cats)

    records = []
    seen    = set(existing)
    cat_idx = 0

    while len(records) < args.n and cat_idx < len(cats):
        category = cats[cat_idx]
        cat_idx += 1
        print(f"  Category: {category}")

        try:
            titles = linked_titles(category)
            time.sleep(1.5)
        except Exception as e:
            print(f"    [skip] {e}")
            time.sleep(3.0)
            continue

        rng.shuffle(titles)
        for title in titles:
            if len(records) >= args.n:
                break
            if title in seen:
                continue
            try:
                extract = fetch_extract(title)
                time.sleep(1.0)
            except Exception:
                continue
            if not extract:
                continue
            sents  = sentences(extract)
            snippet = pick_snippet(sents)
            if not snippet:
                continue
            seen.add(title)
            records.append(make_record(title, category, snippet))
            print(f"    [{len(records):2d}/{args.n}] {title}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} snippets -> {out_path}")


if __name__ == "__main__":
    main()
