from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from .groups import generate_groups
from .io_utils import parse_words, validate_words 
from .scorer import SemanticScorer, normalize_entry
from .search import solve_best_4
import math

Group = Tuple[str, str, str, str]
ScoredGroup = Tuple[Group, float]

@dataclass
class SolveParams:
    top_n: int = 5
    k: int = 400
    cap: int = 80
    min_score: float = -0.25     
    explain: bool = False
    debug: bool = False

def zscore_normalize(scored: List[ScoredGroup]) -> Tuple[List[ScoredGroup], float, float]:
    scores = [s for _, s in scored]
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(var)

    if std < 1e-9:
        return [(g, 0.0) for g, _ in scored], mean, std

    return [(g, (s - mean) / std) for g, s in scored], mean, std

def prune_topk(scored: List[ScoredGroup], k: int) -> List[ScoredGroup]:
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

def prune_word_cap(scored: List[ScoredGroup], cap_per_word: int) -> Tuple[List[ScoredGroup], Dict[str, int]]:
    kept: List[ScoredGroup] = []
    counts: Dict[str, int] = {}

    for g, s in sorted(scored, key=lambda x: x[1], reverse=True):
        ok = True
        for w in g:
            if counts.get(w, 0) >= cap_per_word:
                ok = False
                break
        if ok:
            kept.append((g, s))
            for w in g:
                counts[w] = counts.get(w, 0) + 1
    return kept, counts

def solve_connections(words: List[str], params: SolveParams, scorer: SemanticScorer) -> Dict[str, Any]:
    # Core solver entrypoint for BOTH CLI and Web

    words = [normalize_entry(w) for w in words]

    ok, msg = validate_words(words)
    if not ok:
        return {"ok": False, "error": msg}

    groups = generate_groups(words)
    scorer.embed_words(words)

    # Score all groups (raw)
    scored = scorer.score_all_groups(groups)

    # Z-score normalize across the board (MATCH CLI)
    scored_norm, mu, sigma = zscore_normalize(scored)

    # Prune using normalized scores
    topk = prune_topk(scored_norm, params.k)
    capped, counts = prune_word_cap(topk, params.cap)

    # Search
    solutions = solve_best_4(capped, top_n=params.top_n, min_group_score=params.min_score)

    # Build response
    resp: Dict[str, Any] = {
        "ok": True,
        "words": words,
        "params": {
            "top_n": params.top_n,
            "k": params.k,
            "cap": params.cap,
            "min_score": params.min_score,
            "explain": params.explain,
        },
        "candidate_stats": {
            "total_groups": len(groups),
            "kept_after_topk": len(topk),
            "kept_after_cap": len(capped),
            "most_common_words": sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8],
        },
        "solutions": [],
    }

    if not solutions:
        resp["ok"] = False
        resp["error"] = "No valid solution found under current settings."
        return resp

    # Confidence gap
    if len(solutions) >= 2:
        gap = float(solutions[0][0] - solutions[1][0])
    else:
        gap = None
    resp["confidence_gap"] = gap

    # Put solutions into JSON-friendly structure
    for total, sol in solutions:
        item = {
            "total": float(total),
            "groups": [
                {
                    "group": list(g),
                    "score": float(s),
                    "explain": (scorer.explain_group(g) if params.explain else None),
                }
                for g, s in sol
            ],
        }
        resp["solutions"].append(item)
    return resp
