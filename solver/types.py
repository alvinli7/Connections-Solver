from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

Word = str
Group = Tuple[Word, Word, Word, Word]

@dataclass(frozen=True)
class PuzzleInput:
    words: List[Word]
