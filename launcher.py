"""
TriviaQuest launcher.
Double-click to start the server and open the browser.
Polls localhost:8000 until the server is ready (model loads in ~3-10s).
"""

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def find_python() -> str | None:
    for cmd in ["py", "python", "python3"]:
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def server_ready(host: str = "127.0.0.1", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def alert(msg: str) -> None:
    try:
        import tkinter.messagebox as mb
        mb.showerror("Trivia Quest", msg)
    except Exception:
        print(msg)


if __name__ == "__main__":
    # When frozen by PyInstaller, exe lives next to start_app.py
    base = (
        Path(sys.executable).parent
        if getattr(sys, "frozen", False)
        else Path(__file__).parent
    )

    python = find_python()
    if not python:
        alert(
            "Python not found.\n\n"
            "Please make sure Python is installed and available in PATH."
        )
        sys.exit(1)

    start_script = base / "start_app.py"
    if not start_script.exists():
        alert(f"Could not find start_app.py next to the launcher.\nExpected: {start_script}")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["DISABLE_SYMLINKS_WARNING"] = "1"

    print("Starting Trivia Quest…")
    proc = subprocess.Popen(
        [python, str(start_script)],
        cwd=str(base),
        env=env,
    )

    # Wait up to 120 s for the server to come up (model loading takes a few seconds)
    print("Waiting for server (loading AI model)…")
    for i in range(120):
        if proc.poll() is not None:
            alert("The server exited unexpectedly. Check the terminal window for errors.")
            sys.exit(1)
        if server_ready():
            break
        time.sleep(1)
    else:
        alert("Server did not start in time. Check the terminal for errors.")
        sys.exit(1)

    print("Server ready — opening browser…")
    webbrowser.open("http://localhost:8000")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
