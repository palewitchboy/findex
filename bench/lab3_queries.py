"""The ten hand-labelled queries behind Lab 3's precision@5 table.

Relevance is a judgement, so it is written down here instead of being computed:
for each query, the path stems of the documents a human would accept in the
top 5. Stems (``pg108``) rather than doc ids, because ids depend on the order
``rglob`` happens to return and would silently point at another book after a
re-download.

The labels are deliberately *generous*: a document counts as relevant if it is
genuinely about the query, not if it merely mentions every word. That is the
point of measuring - a scorer that only matches tokens will happily rank a
dictionary above the novel.

``PHRASE_CANDIDATES`` feeds the phrase-vs-AND sanity check: the benchmark tries
each bigram and reports the one whose two words co-occur in many more documents
than they appear adjacently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryCase:
    """One labelled query: the string, the relevant stems, and why."""

    query: str
    relevant: tuple[str, ...]
    note: str


#: Ten queries, hand-labelled against the 120-book Gutenberg corpus.
QUERIES: tuple[QueryCase, ...] = (
    QueryCase(
        query="sherlock holmes watson",
        relevant=("pg108", "pg221"),
        note="the two editions of The Return of Sherlock Holmes",
    ),
    QueryCase(
        query="treasure island pirates silver",
        relevant=("pg120",),
        note="Treasure Island",
    ),
    QueryCase(
        query="tarzan apes jungle",
        relevant=("pg106", "pg236"),
        note="Jungle Tales of Tarzan and The Jungle Book",
    ),
    QueryCase(
        query="moonstone diamond india",
        relevant=("pg155",),
        note="The Moonstone",
    ),
    QueryCase(
        query="dorian gray portrait",
        relevant=("pg174",),
        note="The Picture of Dorian Gray",
    ),
    QueryCase(
        query="phantom opera christine",
        relevant=("pg175",),
        note="The Phantom of the Opera",
    ),
    QueryCase(
        query="daguerreotype photography plate",
        relevant=("pg167", "pg168"),
        note="the two handbooks of early photography",
    ),
    QueryCase(
        query="census population united states",
        relevant=("pg115",),
        note="United States Census Figures",
    ),
    QueryCase(
        query="heart darkness kurtz",
        relevant=("pg219",),
        note="Heart of Darkness",
    ),
    QueryCase(
        query="aeneid troy juno",
        relevant=("pg227", "pg228"),
        note="the Aeneid in Latin and in English",
    ),
)


#: Bigrams tried by the phrase-vs-AND check; the benchmark picks the one with
#: the widest gap between "both words" and "both words, adjacent". The list was
#: chosen by measurement: all six are ordinary phrases whose words are common
#: separately, which is exactly where AND is a bad substitute for a phrase.
PHRASE_CANDIDATES: tuple[str, ...] = (
    "father brown",
    "red sea",
    "great war",
    "black dog",
    "sea captain",
    "golden age",
)
