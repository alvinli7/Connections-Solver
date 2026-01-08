from __future__ import annotations
from typing import List, Tuple

def parse_words(text: str) -> List[str]:
    """
    Parse 16 Connections entries from raw text
    Rules:
    - Prefer newline-separated entries 
    - If there are no newlines, allow comma-separated entries
    - Preserve internal spaces in phrases
    - Normalize by stripping, collapsing multiple spaces, and uppercasing
    """
    raw = text.strip()

    if "\n" in raw:
        parts = [line.strip() for line in raw.splitlines()]
    else:
        parts = [p.strip() for p in raw.split(",")]

    words: List[str] = []
    for p in parts:
        if not p:
            continue
        p = " ".join(p.split())
        words.append(p.upper())
    return words

def validate_words(words: List[str]) -> Tuple[bool, str]:
    if len(words) != 16:
        return False, f"Expected 16 entries, got {len(words)}."

    seen = set()
    dups = []
    for w in words:
        if w in seen:
            dups.append(w)
        seen.add(w)

    if dups:
        return False, f"Duplicate entries found: {sorted(set(dups))}"
    return True, ""


