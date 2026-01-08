from __future__ import annotations
import math
import json
import argparse
from pathlib import Path
from time import perf_counter
from typing import List, Tuple, Any, Dict, Optional
from solver.io_utils import parse_words, validate_words
from solver.groups import generate_groups
from solver.scorer import SemanticScorer
from solver.search import solve_best_4  

Group = Tuple[str, str, str, str]
ScoredGroup = Tuple[Group, float]

def confidence_label(gap: Optional[float]) -> str:
    if gap is None:
        return "n/a"
    if gap > 0.8:
        return "high confidence"
    if gap > 0.3:
        return "medium confidence"
    return "ambiguous"

def zscore_normalize(scored: List[ScoredGroup]) -> tuple[List[ScoredGroup], float, float]:
    
    # Z-score normalize scores across this board: z = (s - mean) / std
    # Returns normalized list, mean, std

    scores = [s for _, s in scored]
    mean = sum(scores) / len(scores)

    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(var)

    # Safety: if std is ~0, normalization would fail
    if std < 1e-9:
        normalized = [(g, 0.0) for g, _ in scored]
        return normalized, mean, std

    normalized = [(g, (s - mean) / std) for g, s in scored]
    return normalized, mean, std

def read_input_text(args: argparse.Namespace) -> str:
    if args.words:
        return args.words

    if args.file:
        return Path(args.file).read_text(encoding="utf-8")

    print("Paste 16 entries (one per line). End with Ctrl+Z then Enter (Windows) / Ctrl+D (macOS/Linux):")
    return "".join(iter(input, ""))  

def prune_topk(scored, k: int):
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

def prune_word_cap(scored, cap_per_word: int):
    """
    Keep highest-scoring groups while capping how often a word can appear.
    """
    kept = []
    counts = {}
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

def main():
    parser = argparse.ArgumentParser(description="NYT Connections solver (CLI)")
    parser.add_argument("--words", type=str, help="Words/phrases as newline-separated or comma-separated text.")
    parser.add_argument("--file", type=str, help="Path to text file containing 16 entries (one per line recommended).")
    parser.add_argument("--top-n", type=int, default=5, help="Number of solutions to return.")
    parser.add_argument("--k", type=int, default=400, help="Top-K groups to keep before word-capping.")
    parser.add_argument("--cap", type=int, default=80, help="Max times a word can appear in candidate groups.")
    parser.add_argument("--min-score", type=float, default=-0.25, help="Minimum group score allowed in final solution.")
    parser.add_argument("--explain", action="store_true", help="Print explanation breakdown for the best solution.")
    parser.add_argument("--debug", action="store_true", help="Print debug diagnostics.")
    parser.add_argument("--json", action="store_true", help="Print JSON output (optional).")
    parser.add_argument("--more", action="store_true", help="Include additional solutions (2..N) in output.")
    parser.add_argument("--details", action="store_true", help="Include extra debug info (words/params/stats).")
    
    args = parser.parse_args()
    text = read_input_text(args)
    words = parse_words(text)

    ok, msg = validate_words(words)
    if not ok:
        print(f"Input error: {msg}")
        print("Tip: paste 16 entries, one per line. Phrases like 'ICE CREAM' should be on one line.")
        return

    if args.debug:
        print(f"Words (count={len(words)}): {words}")

    # Generate all 4-word groups (1820)
    groups = generate_groups(words)
    if args.debug:
        print(f"Total 4-word groups: {len(groups)}")

    scorer = SemanticScorer()
    scorer.embed_words(words)

    t0 = perf_counter()
    scored = scorer.score_all_groups(groups)
    normalized_scored, mu, sigma = zscore_normalize(scored)

    if args.debug:
        print(f"Raw score mean={mu:.4f} std={sigma:.4f}")
    t1 = perf_counter()

    if args.debug:
        print(f"Scoring completed in {(t1 - t0):.3f} seconds")

    topk = prune_topk(normalized_scored, args.k)
    capped, counts = prune_word_cap(topk, args.cap)

    if args.debug:
        # show the most frequent words among candidates
        common = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]
        print("\nMost common words in capped candidates:")
        for w, c in common:
            print(f"{w} {c}")
        print(f"\nCandidate groups kept: {len(capped)}")

    # Search
    solutions = solve_best_4(capped, top_n=args.top_n, min_group_score=args.min_score)

    if not solutions:
        print("\nNo valid solutions found with current settings.")
        print("Try one of:")
        print("  - increase --k (e.g., 600)")
        print("  - increase --cap (e.g., 120)")
        print("  - lower --min-score (e.g., 0.10)")
        return
    gap = None
    label = "n/a"
    if len(solutions) >= 2:
        gap = solutions[0][0] - solutions[1][0]
        label = confidence_label(gap)

    # JSON
    if args.json:
        payload = {
            "words": words,
            "params": {
                "top_n": args.top_n,
                "k": args.k,
                "cap": args.cap,
                "min_score": args.min_score,
            },
            "normalization": {
                "mean": mu,
                "std": sigma,
            },
            "solutions": [],
            "confidence_gap": gap,
            "confidence_label": label,
        }

        for total, sol in solutions:
            item = {
                "total": float(total),
                "groups": [
                    {
                        "group": list(g),
                        "score": float(s),
                        "explain": (scorer.explain_group(g) if args.explain else None),
                    }
                    for g, s in sol
                ],
            }
            payload["solutions"].append(item)

        print(json.dumps(payload, indent=2))
        return

    # Defualt Ouput
    best_total, best_sol = solutions[0]

    print("\nBest solution")
    print(f"Total score: {best_total:.4f}")

    if gap is None:
        print("Confidence gap: n/a (only one solution found)")
    else:
        print(f"Confidence gap: {gap:.4f} ({label})")

    print("\nGroups:")
    for i, (g, s) in enumerate(best_sol, start=1):
        print(f"{i}. {', '.join(g)}")
        print(f"   Group score: {s:.4f}")

        if args.explain:
            info = scorer.explain_group(g)

            print("   Why this group:")
            if isinstance(info, dict):
                if "avg_pair_similarity" in info:
                    print(f"     avg pair sim: {info.get('avg_pair_similarity')}")
                if "min_pair_similarity" in info:
                    print(f"     min pair sim: {info.get('min_pair_similarity')}")
                if "heuristic_total" in info:
                    print(f"     heuristic total: {info.get('heuristic_total')}")
                if "final_score" in info:
                    print(f"     final score: {info.get('final_score')}")
                if "heuristics" in info and info["heuristics"] is not None:
                    print(f"     heuristics: {info['heuristics']}")
            else:
                # Fallback if explain_group returns something else
                print(f"     {info}")

    # Explain Solutions
    if args.more:
        print("\nOther solutions:")
        for idx, (total, sol) in enumerate(solutions[1:], start=2):
            print(f"\nSolution #{idx} (Total Score: {total:.4f})")
            for g, s in sol:
                print(f"  {s:.4f}  ({', '.join(g)})")

    # Details (words / parameters / stats)
    if args.details:
        print("\nDetails")
        print(f"Words (count={len(words)}): {words}")
        print(f"Total 4-word groups: {len(groups)}")
        print(f"Prune Strategy A (Top-K): kept {len(topk)} groups")
        print(f"Prune Strategy C (Top-K + word cap): kept {len(capped)} groups")

if __name__ == "__main__":
    main()
