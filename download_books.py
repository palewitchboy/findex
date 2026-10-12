"""Download plain-text books from Project Gutenberg into data/."""

import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path("data")
TARGET_FILES = 120          # stop after this many books
FIRST_ID, LAST_ID = 100, 600
DELAY = 1.0                 # seconds between requests: be polite to the server

OUT.mkdir(exist_ok=True)
saved = len(list(OUT.glob("*.txt")))

for book_id in range(FIRST_ID, LAST_ID):
    if saved >= TARGET_FILES:
        break
    target = OUT / f"pg{book_id}.txt"
    if target.exists():
        continue
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            target.write_bytes(resp.read())
        saved += 1
        print(f"saved {target.name} ({saved}/{TARGET_FILES})")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"skip {book_id}: {exc}")
    time.sleep(DELAY)

print("done:", saved, "files in", OUT.resolve())