"""
FastAPI backend for the trivia game.

Retrieves a topic first, then generates questions one at a time in a background
thread.  Each completed question is exposed immediately through /api/status so
the frontend can start the game while the remaining questions are still being
generated.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import model as M
from app import wikipedia as W

STATIC_DIR    = Path(__file__).parent / "static"
NUM_QUESTIONS = 5
RETRY_LIMIT   = 6          # snippet attempts per question slot

SUGGESTED_THEMES = [
    {"label": "Space",                   "emoji": "🚀"},
    {"label": "Ancient Rome",            "emoji": "🏛️"},
    {"label": "Ocean",                   "emoji": "🌊"},
    {"label": "Dinosaurs",               "emoji": "🦕"},
    {"label": "World War II",            "emoji": "⚔️"},
    {"label": "Ancient Egypt",           "emoji": "🏺"},
    {"label": "Black Holes",             "emoji": "🌌"},
    {"label": "Evolution",               "emoji": "🧬"},
    {"label": "Medieval Europe",         "emoji": "🏰"},
    {"label": "Climate Change",          "emoji": "🌡️"},
    {"label": "The Human Brain",         "emoji": "🧠"},
    {"label": "Quantum Physics",         "emoji": "⚛️"},
    {"label": "Amazon Rainforest",       "emoji": "🌿"},
    {"label": "Volcanoes",               "emoji": "🌋"},
    {"label": "Artificial Intelligence", "emoji": "🤖"},
    {"label": "Renaissance Art",         "emoji": "🎨"},
]

# Single-worker executor used only for model loading at startup
_load_executor = ThreadPoolExecutor(max_workers=1)


# ── Game state ────────────────────────────────────────────────────────────────

class GameState:
    def __init__(self):
        self._lock = threading.Lock()
        self._do_reset()

    def _do_reset(self):
        self.status: str = "idle"   # idle | loading | generating | ready | error
        self.theme: str = ""
        self.found_title: str = ""
        self.questions: list[dict] = []
        self.progress: int = 0      # questions generated so far
        self.error: str = ""

    def reset(self):
        with self._lock:
            self._do_reset()


state = GameState()


# ── Background generation (runs in a daemon thread) ───────────────────────────

def _generate_all_blocking() -> None:
    """Load Wikipedia + generate NUM_QUESTIONS questions sequentially."""
    theme = state.theme

    # Step 1 — search Wikipedia
    try:
        print(f"[game] loading topic: {theme}")
        cache = W.load_topic(theme)
        with state._lock:
            if state.status == "idle":      # reset() was called → abort
                return
            state.found_title = cache.found_title
            state.status = "generating"
        print(f"[game] Wikipedia ready: '{cache.found_title}'")
    except ValueError as e:
        with state._lock:
            state.status = "error"
            state.error = str(e)
        return
    except Exception as e:
        with state._lock:
            state.status = "error"
            state.error = f"Unexpected error: {e}"
        print(f"[game] topic load error: {e}")
        return

    # Step 2 — generate questions one at a time and publish each one immediately.
    try:
        for slot in range(NUM_QUESTIONS):
            with state._lock:
                if state.status == "idle":      # cancelled mid-run
                    return
            print(f"[gen] question {slot + 1}/{NUM_QUESTIONS}…")
            for attempt in range(RETRY_LIMIT):
                snippet, source_title, is_joker = cache.pick_snippet()
                if not snippet:
                    continue
                q = M.generate_question(
                    snippet=snippet,
                    source_title=source_title,
                    category=theme,
                )
                if q:
                    q["is_joker"] = is_joker
                    with state._lock:
                        if state.status == "idle":
                            return
                        state.questions.append(q)
                        state.progress = len(state.questions)
                    print(f"[gen] Q{slot + 1} done (joker={is_joker})")
                    break
            else:
                print(f"[gen] Q{slot + 1} failed after {RETRY_LIMIT} attempts — skipping")
    except Exception as e:
        # Any unexpected crash inside generation: surface it instead of
        # leaving the frontend polling forever on status="generating".
        print(f"[gen] fatal error: {e!r}")
        with state._lock:
            state.status = "error"
            state.error = f"Generation crashed: {e}"
        return

    with state._lock:
        if state.status == "idle":
            return
        question_count = len(state.questions)
        if question_count:
            state.status = "ready"
        else:
            state.status = "error"
            state.error = "Could not generate any questions — please try a different topic."
    print(f"[gen] complete: {question_count}/{NUM_QUESTIONS} questions ready")


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_load_executor, M.load_model)
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    # Disable browser caching for the SPA shell so stale frontends never
    # outlive a backend change (browsers love to cache index.html aggressively).
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma":        "no-cache",
            "Expires":       "0",
        },
    )


@app.get("/api/themes")
async def get_themes():
    return {"themes": SUGGESTED_THEMES}


class StartBody(BaseModel):
    theme: str


@app.post("/api/start")
async def start_game(body: StartBody):
    theme = body.theme.strip()
    if not theme:
        raise HTTPException(400, "Theme cannot be empty")

    state.reset()
    with state._lock:
        state.status = "loading"
        state.theme = theme

    # Daemon thread — runs independently; modifies state directly
    threading.Thread(target=_generate_all_blocking, daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
async def get_status():
    with state._lock:
        return {
            "status":      state.status,
            "found_title": state.found_title,
            "progress":    state.progress,
            "total":       NUM_QUESTIONS,
            "questions":   list(state.questions),
            "error":       state.error,
        }


@app.post("/api/reset")
async def reset_game():
    state.reset()
    return {"ok": True}
