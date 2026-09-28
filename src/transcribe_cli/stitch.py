# SPDX-License-Identifier: Apache-2.0
"""Pure overlap detection and transcript joining."""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Sequence

EDGE_TOKENS = 32  # tokens compared at each side of a join
MIN_PHRASE = 3  # shortest shared phrase accepted as an overlap
MIN_HAN_PHRASE = 6  # Han phrases are denser, so require more characters
MAX_LEADING_MISMATCHES = 6  # differently recognised words tolerated before a shared phrase
FUZZY_WORD_RATIO = 0.6
TRIM_CHARS = " \t\r\n.,!?;:。？！、，"


def join_chunks(texts: Sequence[str]) -> str:
    parts: list[str] = []
    last_char = ""
    for text in texts:
        text = text.strip()
        if text:
            if parts and not ("一" <= last_char <= "鿿" and "一" <= text[0] <= "鿿"):
                parts.append(" ")
            parts.append(text)
            last_char = text[-1]
    return "".join(parts)


def text_tokens(text: str) -> list[tuple[str, int]]:
    # Keep Han characters separate even when they touch Latin text.
    pattern = r"[㐀-鿿]|[^\W_㐀-鿿]+(?:['’][^\W_㐀-鿿]+)*"
    return [(unicodedata.normalize("NFKC", match.group()).casefold(), match.end())
            for match in re.finditer(pattern, text, re.UNICODE)]


def _is_han_phrase(words: Sequence[str]) -> bool:
    return all("㐀" <= word <= "鿿" for word in words)


def _acceptable_phrase(phrase: Sequence[str]) -> bool:
    if len(set(phrase)) < 2:
        return False
    return not (_is_han_phrase(phrase) and len(phrase) < MIN_HAN_PHRASE)


def overlap_offset(previous: str, following: str) -> int | None:
    left = text_tokens(previous)[-EDGE_TOKENS:]
    right = text_tokens(following)[:EDGE_TOKENS]
    left_words = [word for word, _ in left]
    right_words = [word for word, _ in right]
    for size in range(min(len(left), len(right)), MIN_PHRASE - 1, -1):
        phrase = left_words[-size:]
        if not _acceptable_phrase(phrase):
            continue
        if phrase == right_words[:size]:
            return right[size - 1][1]
    # Permit a few differently recognized words before a shared edge phrase.
    for block in difflib.SequenceMatcher(None, left_words, right_words, autojunk=False).get_matching_blocks():
        if (block.size < MIN_PHRASE or block.a + block.size != len(left)
                or not 1 <= block.b <= MAX_LEADING_MISMATCHES or block.a < block.b):
            continue
        phrase = left_words[block.a:block.a + block.size]
        if not _acceptable_phrase(phrase):
            continue
        preceding = zip(left_words[block.a - block.b:block.a], right_words[:block.b], strict=False)
        if all(difflib.SequenceMatcher(None, a, b).ratio() >= FUZZY_WORD_RATIO for a, b in preceding):
            return right[block.b + block.size - 1][1]
    return None


def stitch_chunks(texts: Sequence[str]) -> tuple[str, int]:
    """Join transcripts of overlapping audio; returns the text and the count of unaligned joins."""
    parts: list[str] = []
    unresolved = 0
    previous = ""
    for text in texts:
        text = text.strip()
        original = text
        if previous and text:
            offset = overlap_offset(previous, text)
            if offset is None:
                unresolved += 1
            else:
                text = text[offset:].lstrip(TRIM_CHARS)
        parts.append(text)
        previous = original
    return join_chunks(parts), unresolved
