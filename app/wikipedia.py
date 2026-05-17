"""
Wikipedia fetching for the trivia app.

load_topic(theme) -> TopicCache
TopicCache.pick_snippet() -> (sentences, source_title)
  40% chance: main article
  60% chance: a page linked from the main article
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

import requests

API_URL    = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TriviaApp/1.0 (educational; local)"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT

BAD_PREFIXES = (
    "File:", "Help:", "Wikipedia:", "Category:", "Template:", "Talk:",
    "Portal:", "Special:", "Module:", "User:", "Draft:", "List of",
)

NAV_PHRASES = (
    "this article", "this section", "see also", "external links",
    "further reading", "references", "^ ", "citation needed",
)


# ── Wikipedia API helpers ─────────────────────────────────────────────────────

def _api(params: dict, retries: int = 4) -> dict:
    params.setdefault("format", "json")
    delay = 1.0
    for attempt in range(retries):
        try:
            r = SESSION.get(API_URL, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(delay * (2 ** attempt))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
    return {}


def search_page(query: str) -> str | None:
    data = _api({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 3,
    })
    results = data.get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def fetch_extract(title: str) -> str | None:
    data = _api({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "redirects": True,
    })
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "").strip()
        if text and len(text) > 120:
            return text
    return None


def fetch_linked_titles(title: str, limit: int = 60) -> list[str]:
    data = _api({
        "action": "query",
        "titles": title,
        "prop": "links",
        "pllimit": limit,
        "plnamespace": 0,
    })
    pages = data.get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        for link in page.get("links", []):
            t = link["title"]
            if not any(t.startswith(p) for p in BAD_PREFIXES):
                out.append(t)
    return out


# ── Text processing ───────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"==+[^=]+=+", "", text)
    return text.strip()


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", _clean(text))
    return [s.strip() for s in raw if len(s.strip()) > 40]


def _is_useful(s: str) -> bool:
    low = s.lower()
    return not any(n in low for n in NAV_PHRASES) and not s.startswith("^")


def extract_to_snippet(extract: str, n: int = 3) -> list[str]:
    sents = [s for s in _sentences(extract) if _is_useful(s)]
    if len(sents) < 2:
        return []
    start = random.randint(0, max(0, len(sents) - n))
    return sents[start: start + n]


# ── TopicCache ────────────────────────────────────────────────────────────────

@dataclass
class TopicCache:
    theme: str
    found_title: str
    main_extract: str
    linked_titles: list[str]
    _cache: dict[str, str] = field(default_factory=dict)
    _tried: set[str] = field(default_factory=set)

    def __post_init__(self):
        self._cache[self.found_title] = self.main_extract

    def pick_snippet(self) -> tuple[list[str], str, bool]:
        """Return (snippet_sentences, source_title, is_joker).
        80% chance: main article (is_joker=False).
        20% chance: a linked page  (is_joker=True).
        """
        if self.linked_titles and random.random() < 0.20:
            untried = [t for t in self.linked_titles if t not in self._tried]
            random.shuffle(untried)
            for title in untried[:6]:
                self._tried.add(title)
                if title not in self._cache:
                    extract = fetch_extract(title)
                    if extract:
                        self._cache[title] = extract
                if title in self._cache:
                    snippet = extract_to_snippet(self._cache[title])
                    if snippet:
                        return snippet, title, True   # joker

        # Main page (80 %)
        snippet = extract_to_snippet(self.main_extract)
        if not snippet:
            snippet = [self.main_extract[:400]]
        return snippet, self.found_title, False


def load_topic(theme: str) -> TopicCache:
    title = search_page(theme)
    if not title:
        raise ValueError(f"No Wikipedia article found for '{theme}'")
    extract = fetch_extract(title)
    if not extract:
        raise ValueError(f"Could not load article '{title}'")
    linked = fetch_linked_titles(title, limit=60)
    random.shuffle(linked)
    return TopicCache(
        theme=theme,
        found_title=title,
        main_extract=extract,
        linked_titles=linked,
    )
