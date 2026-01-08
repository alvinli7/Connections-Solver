from __future__ import annotations
from itertools import combinations
from typing import List
from .types import Group, Word

def generate_groups(words: List[Word]) -> List[Group]:
    """
    Generate all 4-word groups from the given list of words.
    For 16 words, this returns 1820 groups.
    """
    return list(combinations(words, 4))
