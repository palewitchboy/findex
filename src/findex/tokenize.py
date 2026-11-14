"""M2 - streaming tokenizer.

Pipeline: NFC-normalize -> casefold -> ``re.finditer`` over a word pattern.

Tokenization policy
-------------------
* **Letters/digits:** a token is a run of Unicode letters and digits
  (``[^\\W_]+`` - Unicode-aware ``\\w`` without the underscore). Cyrillic,
  accented Latin, CJK all work.
* **Digits:** kept. ``2026``, ``python3`` and ``3d`` are tokens.
* **Apostrophes:** an apostrophe *inside* a word is kept, so ``don't`` and
  ``п'ять`` are one token each. Typographic apostrophes (U+2019, U+02BC) are
  first mapped to ASCII ``'`` so ``п’ять`` and ``п'ять`` are the same token.
  Leading/trailing apostrophes (quotes) are dropped.

  Lab 8 measured that mapping: ``str.translate`` with a dict walks every
  character through a Python-level table - 1 075 ms for a 5.4 MB book, 39% of
  a serial index build. Three guarded ``str.replace`` calls do the same job in
  12.7 ms (85x) because ``replace`` is a C two-way search that only copies when
  the character is present.
* **Hyphens:** a hyphen *inside* a word is kept (``well-known``, ``IT-фахівець``).
  Dashes surrounded by spaces and leading/trailing hyphens are separators.
* **Case:** ``str.casefold()`` (``Straße`` == ``STRASSE``), not ``lower()``.
* **Unicode:** NFC first, so ``cafe\\u0301`` == ``café``.
* **Known limitation:** a combining mark with no precomposed NFC form is not a
  word character and would split a token. Rare in practice.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_APOSTROPHES = ("\u2019", "\u02bc", "\u2018")

_WORD = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized word tokens from ``text`` lazily."""
    text = unicodedata.normalize("NFC", text).casefold()
    for apostrophe in _APOSTROPHES:
        if apostrophe in text:
            text = text.replace(apostrophe, "'")
    for match in _WORD.finditer(text):
        yield match.group()
