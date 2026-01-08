from __future__ import annotations
from typing import List, Tuple, Set
from .types import Group

ScoredGroup = Tuple[Group, float]

def solve_best_4(candidates: List[ScoredGroup], top_n: int = 5, min_group_score: float = 0.20,) -> List[Tuple[float, List[ScoredGroup]]]:
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    solutions: List[Tuple[float, List[ScoredGroup]]] = []

    def backtrack(index: int, picked: List[ScoredGroup], used_words: Set[str], total_score: float) -> None:
        # Base case
        if len(picked) == 4:
            scores = [s for _, s in picked]
            if min(scores) < min_group_score:
                return
            solutions.append((total_score, picked.copy()))
            solutions.sort(key=lambda x: x[0], reverse=True)
            if len(solutions) > top_n:
                solutions.pop()
            return

        if index >= len(candidates):
            return

        cutoff = solutions[-1][0] if len(solutions) >= top_n else float("-inf")

        remaining = 4 - len(picked)
        optimistic = total_score
        for j in range(remaining):
            if index + j < len(candidates):
                optimistic += candidates[index + j][1]

        if optimistic <= cutoff:
            return

        for i in range(index, len(candidates)):
            group, score = candidates[i]
            group_set = set(group)
            if used_words & group_set:
                continue

            picked.append((group, score))
            backtrack(i + 1, picked, used_words | group_set, total_score + score)
            picked.pop()

    backtrack(0, [], set(), 0.0)
    return solutions
