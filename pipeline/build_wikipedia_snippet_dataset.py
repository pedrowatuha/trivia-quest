#!/usr/bin/env python3
"""
Build a JSONL dataset of real English Wikipedia snippets.

This script retrieves article text from Wikipedia through the MediaWiki Action
API. It does not use AI models and does not generate questions, answers,
alternatives, explanations, or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote

try:
    import requests
except ImportError as exc:
    print(
        "Missing dependency: requests. Install it with: python -m pip install requests",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "WikipediaSnippetDatasetBuilder/1.0 "
    "(educational dataset tool; contact: local user)"
)

DEFAULT_CATEGORIES = [
    "Sports",
    "History",
    "Geography",
    "Science",
    "Arts",
    "Literature",
    "Film",
    "Technology",
    "Nature",
    "Cuisine",
    "Videogames",
    "Economics",
]

BAD_TITLE_PREFIXES = (
    "File:",
    "Help:",
    "Wikipedia:",
    "Category:",
    "Template:",
    "Talk:",
    "Portal:",
    "Special:",
    "Module:",
    "User:",
    "Draft:",
    "Book:",
    "TimedText:",
    "MediaWiki:",
)

NAVIGATION_PHRASES = (
    "this article needs",
    "this section needs",
    "main article:",
    "see also",
    "external links",
    "further reading",
    "references",
)

ABBREVIATION_END_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|No|Fig|Eq|Vol|Inc|Ltd|Co|vs|etc|e\.g|i\.e)\.$",
    re.I,
)
ROMAN_NUMERAL_END_RE = re.compile(r"\b[IVXLCDM]+\.$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a JSONL dataset of real Wikipedia article snippets."
    )
    parser.add_argument("--out", required=True, help="Path to the output JSONL file.")
    parser.add_argument(
        "--num-records",
        required=True,
        type=int,
        help="Number of snippet records to write.",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Optional comma-separated seed topic pages, e.g. Sports,History,Science.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-pages-per-category",
        type=int,
        default=200,
        help="Maximum linked article titles to keep from each seed category.",
    )
    parser.add_argument(
        "--sentences-per-record",
        type=int,
        default=5,
        help="Preferred number of sentences per snippet record.",
    )
    parser.add_argument(
        "--snippets-per-page",
        type=int,
        default=1,
        help="Maximum snippet records to draw from each retrieved article.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between uncached API requests.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".wiki_cache",
        help="Directory for cached API responses.",
    )
    return parser.parse_args()


class WikipediaClient:
    """Small MediaWiki API client with JSON caching, sleeping, and retries."""

    def __init__(self, cache_dir: Path, sleep_seconds: float) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch a JSON API response, using the local cache when possible."""
        full_params = dict(params)
        full_params.setdefault("format", "json")
        full_params.setdefault("formatversion", "2")

        cache_path = self._cache_path(full_params)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        data = self._request_with_retries(full_params)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return data

    def _cache_path(self, params: Dict[str, Any]) -> Path:
        cache_key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _request_with_retries(self, params: Dict[str, Any]) -> Dict[str, Any]:
        max_attempts = 5
        for attempt in range(max_attempts):
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

            try:
                response = self.session.get(API_URL, params=params, timeout=30)

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else 2**attempt
                    logging.warning("Rate limited by Wikipedia; sleeping %.1fs", delay)
                    time.sleep(delay)
                    continue

                if 500 <= response.status_code < 600:
                    delay = 2**attempt
                    logging.warning(
                        "Wikipedia server error %s; retrying in %.1fs",
                        response.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                response.encoding = "utf-8"
                return response.json()

            except (requests.RequestException, ValueError) as exc:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"API request failed after retries: {exc}") from exc

                delay = 2**attempt
                logging.warning("Transient request failure: %s; retrying in %.1fs", exc, delay)
                time.sleep(delay)

        raise RuntimeError("API request failed after retries")


def split_categories(value: Optional[str]) -> List[str]:
    if value is None:
        return DEFAULT_CATEGORIES
    categories = [item.strip() for item in value.split(",") if item.strip()]
    if not categories:
        raise ValueError("--categories was provided but no category names were found")
    return categories


def is_probably_article_title(title: str) -> Tuple[bool, str]:
    """Return whether a linked page title looks suitable for article snippets."""
    if not title:
        return False, "empty title"
    if title.startswith(BAD_TITLE_PREFIXES):
        return False, "non-article namespace"
    if title.lower().startswith("list of"):
        return False, "list page"
    if title.endswith("(disambiguation)"):
        return False, "disambiguation title"
    if title.startswith("#"):
        return False, "page fragment"
    return True, ""


def collect_linked_titles(
    client: WikipediaClient,
    seed_title: str,
    max_pages: int,
) -> List[str]:
    """Retrieve and filter article links from one seed page, respecting continuation."""
    titles: List[str] = []
    seen: Set[str] = set()
    continuation: Dict[str, Any] = {}

    while len(titles) < max_pages:
        params: Dict[str, Any] = {
            "action": "query",
            "prop": "links",
            "titles": seed_title,
            "plnamespace": 0,
            "pllimit": "max",
            "redirects": 1,
        }
        params.update(continuation)

        data = client.get(params)
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            logging.warning("Skipping seed %r: no pages returned", seed_title)
            break

        for page in pages:
            if page.get("missing"):
                logging.warning("Skipping seed %r: page missing", seed_title)
                continue
            for link in page.get("links", []):
                title = link.get("title", "")
                keep, reason = is_probably_article_title(title)
                if not keep:
                    logging.info("Skipped linked page %r from %r: %s", title, seed_title, reason)
                    continue
                if title in seen:
                    continue
                seen.add(title)
                titles.append(title)
                if len(titles) >= max_pages:
                    break
            if len(titles) >= max_pages:
                break

        continuation = data.get("continue", {})
        if not continuation:
            break

    logging.info("Collected %d linked titles from %r", len(titles), seed_title)
    return titles


def fetch_article_extract(client: WikipediaClient, title: str) -> Optional[Dict[str, Any]]:
    """Retrieve plain-text article extract and metadata for one title."""
    params = {
        "action": "query",
        "prop": "extracts|info|pageprops",
        "titles": title,
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "redirects": 1,
    }
    data = client.get(params)
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        logging.info("Skipped %r: no page data returned", title)
        return None

    page = pages[0]
    if page.get("missing"):
        logging.info("Skipped %r: missing page", title)
        return None
    if "disambiguation" in page.get("pageprops", {}):
        logging.info("Skipped %r: disambiguation page", title)
        return None

    extract = page.get("extract", "")
    if not extract.strip():
        logging.info("Skipped %r: empty extract", title)
        return None

    return {
        "title": page.get("title", title),
        "url": page.get("fullurl") or wikipedia_url(page.get("title", title)),
        "extract": extract,
    }


def wikipedia_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="_()")


def clean_text(text: str) -> str:
    """Remove citation markers and normalize whitespace while preserving plain text."""
    text = re.sub(r"\[(?:\d+|citation needed|clarification needed|note \d+)\]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(extract: str) -> List[str]:
    """Split plain text into sentences without requiring external NLP libraries."""
    cleaned_lines = []
    for line in extract.splitlines():
        line = clean_text(line)
        if line:
            cleaned_lines.append(line)

    text = " ".join(cleaned_lines)
    rough_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = merge_sentence_fragments(clean_text(sentence) for sentence in rough_sentences)
    return [sentence for sentence in sentences if is_useful_sentence(sentence)]


def merge_sentence_fragments(sentences: Iterable[str]) -> List[str]:
    """Join fragments caused by common abbreviations such as 'Dr.' or 'IV.'."""
    merged: List[str] = []

    for sentence in sentences:
        if not sentence:
            continue

        if merged and should_merge_with_next(merged[-1]):
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)

    return merged


def should_merge_with_next(sentence: str) -> bool:
    if ABBREVIATION_END_RE.search(sentence):
        return True
    if ROMAN_NUMERAL_END_RE.search(sentence):
        return True
    return False


def is_useful_sentence(sentence: str) -> bool:
    """Filter out short, navigational, coordinate-like, or low-information sentences."""
    if len(sentence) < 45:
        return False
    if len(sentence.split()) < 8:
        return False
    if not sentence.endswith((".", "!", "?")):
        return False

    lowered = sentence.lower()
    if any(phrase in lowered for phrase in NAVIGATION_PHRASES):
        return False

    letters = sum(1 for char in sentence if char.isalpha())
    digits = sum(1 for char in sentence if char.isdigit())
    if letters == 0:
        return False
    if digits > letters:
        return False

    # Coordinates and infobox-like fragments are not useful training snippets.
    if re.search(r"\d+\u00b0\s*\d*[\u2032']?\s*\d*[\u2033\"]?\s*[NSEW]", sentence):
        return False
    if re.fullmatch(r"[\d\s,.;:()/\-\u2013\u2014]+", sentence):
        return False

    return True


def choose_snippets(
    sentences: Sequence[str],
    preferred_count: int,
    max_snippets: int,
    seen_snippets: Set[str],
) -> List[List[str]]:
    """Select compact non-overlapping snippets, preferring earlier sentences."""
    if len(sentences) < 3:
        return []

    count = max(3, min(8, preferred_count))
    snippets: List[List[str]] = []

    for start in range(0, len(sentences), count):
        if len(snippets) >= max_snippets:
            break

        snippet = list(sentences[start : start + count])
        if len(snippet) < 3:
            continue

        snippet_key = "\n".join(snippet)
        if snippet_key in seen_snippets:
            continue

        seen_snippets.add(snippet_key)
        snippets.append(snippet)

    return snippets


def make_record(
    category: str,
    article: Dict[str, Any],
    snippet: List[str],
    retrieved_at: str,
) -> Dict[str, Any]:
    return {
        "category": category,
        "source_title": article["title"],
        "source_url": article["url"],
        "source_snippets": snippet,
        "retrieved_at": retrieved_at,
        "license": "CC BY-SA",
        "source": "Wikipedia",
    }


def build_title_pool(
    client: WikipediaClient,
    categories: Sequence[str],
    max_pages_per_category: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """Collect linked titles for all categories and return shuffled work items."""
    work_items: List[Tuple[str, str]] = []
    for category in categories:
        titles = collect_linked_titles(client, category, max_pages_per_category)
        rng.shuffle(titles)
        work_items.extend((category, title) for title in titles)

    rng.shuffle(work_items)
    return work_items


def write_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON object to the output file and flush it immediately."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def existing_snippet_keys(path: Path) -> Set[str]:
    """Avoid duplicate snippets when appending to an existing JSONL output file."""
    keys: Set[str] = set()
    if not path.exists():
        return keys

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Could not parse existing JSONL line %d; ignoring", line_number)
                continue
            snippets = item.get("source_snippets", [])
            if isinstance(snippets, list):
                keys.add("\n".join(str(sentence) for sentence in snippets))
    return keys


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.num_records <= 0:
        raise ValueError("--num-records must be positive")
    if args.max_pages_per_category <= 0:
        raise ValueError("--max-pages-per-category must be positive")
    if args.sentences_per_record <= 0:
        raise ValueError("--sentences-per-record must be positive")
    if args.snippets_per_page <= 0:
        raise ValueError("--snippets-per-page must be positive")

    categories = split_categories(args.categories)
    rng = random.Random(args.seed)
    client = WikipediaClient(Path(args.cache_dir), args.sleep)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_snippets = existing_snippet_keys(output_path)
    records_written = 0

    work_items = build_title_pool(client, categories, args.max_pages_per_category, rng)
    if not work_items:
        logging.error("No candidate linked article titles were found.")
        return 1

    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    retrieved_at = retrieved_at.replace("+00:00", "Z")

    for category, title in work_items:
        if records_written >= args.num_records:
            break

        article = fetch_article_extract(client, title)
        if article is None:
            continue

        sentences = split_into_sentences(article["extract"])
        snippets = choose_snippets(
            sentences,
            args.sentences_per_record,
            args.snippets_per_page,
            seen_snippets,
        )
        if not snippets:
            logging.info("Skipped %r: fewer than 3 usable or unique sentences", title)
            continue

        for snippet in snippets:
            if records_written >= args.num_records:
                break
            record = make_record(category, article, snippet, retrieved_at)
            write_jsonl_record(output_path, record)
            records_written += 1
            logging.info("Wrote %d/%d: %s", records_written, args.num_records, article["title"])

    if records_written < args.num_records:
        logging.warning(
            "Only wrote %d of %d requested records. Try increasing "
            "--max-pages-per-category or adding categories.",
            records_written,
            args.num_records,
        )

    logging.info("Done. Output written to %s", output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; already-written JSONL records were preserved.", file=sys.stderr)
        raise SystemExit(130)
