# Wikipedia Snippet Dataset Builder

This standalone tool creates a JSONL dataset of real snippets retrieved from
English Wikipedia through the MediaWiki Action API. It does not use any AI
model, and it does not generate trivia questions, answers, alternatives,
explanations, labels, or fabricated data.

## Install dependencies

Use Python 3.9 or newer. The only third-party dependency is `requests`.

```bash
python -m pip install requests
```

## Example command

```bash
python build_wikipedia_snippet_dataset.py --out wikipedia_snippets.jsonl --num-records 1000 --categories Sports,History,Science,Geography
```

Useful options:

- `--seed 42` controls random sampling.
- `--max-pages-per-category 200` limits linked article candidates per seed page.
- `--sentences-per-record 5` requests compact snippets, clamped to 3-8 sentences.
- `--snippets-per-page 1` controls how many non-overlapping records to draw from each article.
- `--sleep 0.2` waits between uncached API requests.
- `--cache-dir .wiki_cache` stores cached API responses.

## JSONL output format

Each line is one JSON object:

```json
{
  "category": "Sports",
  "source_title": "2022 FIFA World Cup",
  "source_url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup",
  "source_snippets": [
    "The 2022 FIFA World Cup was the 22nd FIFA World Cup.",
    "It took place in Qatar from 20 November to 18 December 2022.",
    "Argentina won the final against France."
  ],
  "retrieved_at": "2026-05-13T12:00:00Z",
  "license": "CC BY-SA",
  "source": "Wikipedia"
}
```

The dataset contains only real Wikipedia snippets retrieved by the script.
There are no generated trivia labels or model-created fields.
