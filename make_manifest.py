"""Write corpus_manifest.md: which Gutenberg books are in data/."""

from itertools import islice
from pathlib import Path

lines = ["| File | Title | Size (KB) |", "|---|---|---|"]
for path in sorted(Path("data").glob("*.txt")):
    title = "?"
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in islice(f, 60):  # the header is in the first lines
            if "Title:" in line:
                title = line.split("Title:", 1)[1].strip().replace("|", "/")
                break
    lines.append(f"| {path.name} | {title} | {path.stat().st_size // 1000} |")

Path("corpus_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines) - 2, "books listed")