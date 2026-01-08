from __future__ import annotations
import re
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from .types import Group, Word

LEXICON_CATEGORIES: dict[str, set[str]] = {
    "COLOURS": {
        "RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE", "PINK", "BROWN",
        "BLACK", "WHITE", "GRAY", "GREY", "VIOLET", "INDIGO",
    },
    "DAYS": {"MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"},
    "MONTHS": {
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
        "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    },
    "PLANETS": {"MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"},
    "DIRECTIONS": {"NORTH", "SOUTH", "EAST", "WEST", "UP", "DOWN", "LEFT", "RIGHT"},
    "METALS": {"IRON", "GOLD", "SILVER", "COPPER", "TIN", "LEAD", "ZINC", "NICKEL"},
    "INSTRUMENTS": {"DRUM", "DRUMS", "PIANO", "GUITAR", "VIOLIN", "CELLO", "TRUMPET", "FLUTE", "CLARINET", "TROMBONE", "VIOLA", "BASSOON", "SAXOPHONE"},
    "CURRENCIES": {"DOLLAR", "EURO", "YEN", "POUND", "RUPEE", "PESO", "FRANC", "KRONA", "WON"},
    "ANIMALS_COMMON": {"DOG", "CAT", "HORSE", "COW", "PIG", "SHEEP", "GOAT", "MOUSE", "RAT", "LION", "TIGER", "BEAR", "WHALE", "DOLPHIN", "FISH", "SNAKE"},
    "TRANSPORT": {"CAR", "TRAIN", "PLANE", "BOAT", "BUS", "TRUCK", "BIKE", "SHIP", "SUBWAY"},
}

PHRASE_GLUERS_PREFIX: dict[str, set[str]] = {
    "BUCKET": {"LIST", "SEAT", "BRIGADE", "HAT"},
    "ICE": {"CREAM", "CUBE", "PACK", "AGE"},
    "BLACK": {"HOLE", "OUT", "JACK", "MAIL"},
    "WHITE": {"HOUSE", "NOISE", "OUT", "WASH"},
    "GREEN": {"HOUSE", "LIGHT", "ROOM", "TEA"},
    "SILVER": {"LINING", "SCREEN", "BULLET", "FOX"},
    "GOLD": {"RUSH", "MEDAL", "MINE", "STANDARD"},
}

PHRASE_GLUERS_SUFFIX: dict[str, set[str]] = {
    "LINE": {"PUNCH", "BASE", "AIR", "PIPE"},   
    "BALL": {"BASE", "FOOT", "SOFT", "DODGE"},
    "TIME": {"SHOW", "BED", "PRIME", "HALF"},
}

_word_re = re.compile(r"[^A-Z0-9 ]+")

def normalize_entry(w: str) -> str:
    w = " ".join(w.strip().upper().split())
    w = _word_re.sub("", w).strip()
    return w

def tokenize(w: str) -> list[str]:
    return w.split(" ")

def all_anagrams(words: list[str]) -> bool:
    if any(" " in w for w in words):
        return False
    sigs = ["".join(sorted(w)) for w in words]
    return len(set(sigs)) == 1

def has_consistent_punc_shape(words: list[str]) -> bool:
    features = []
    for w in words:
        features.append(((" " in w), ("'" in w), ("-" in w), ("." in w)))
    return len(set(features)) == 1

class SemanticScorer:
    """
    Loads a pretrained sentence-transformer model and provides:
    - word -> embedding caching
    - group scoring using average pairwise cosine similarity
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Load once (expensive). Reuse for all scoring.
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[Word, np.ndarray] = {}

    def embed_words(self, words: List[Word]) -> None:
        
        # Embed all words and store in cache --> faster than embedding per-group.
        missing = [w for w in words if w not in self.cache]
        if not missing:
            return

        vecs = self.model.encode(missing, convert_to_numpy=True, normalize_embeddings=True)
        for w, v in zip(missing, vecs):
            self.cache[w] = v

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:

        # Cosine similarity. If embeddings are normalized, cosine = dot product.
        return float(np.dot(a, b))
    
    def heuristic_bonus (self, group: Group) -> float:
        words = [normalize_entry(w) for w in group]
        bonus = 0.0

        lengths = {len(w) for w in group}
        if len(lengths) == 1:
            bonus += 0.05
        
        plural_flags = [w.endswith('S') for w in group]
        if all(plural_flags) or not any(plural_flags):
            bonus += 0.08
        
        if all_anagrams(words):
            bonus += 0.2

        if has_consistent_punc_shape(words):
            bonus += 0.08
    
        wset = set(words)
        for _, vocab in LEXICON_CATEGORIES.items():
            if wset.issubset(vocab):
                bonus += 1.0
                break

        tokens = [tokenize(w) for w in words]
        first_tokens = [t[0] for t in tokens if t]
        last_tokens = [t[-1] for t in tokens if t]

        if len(first_tokens) == 4 and len(set(first_tokens)) == 1:
            bonus += 0.20
        if len(last_tokens) == 4 and len(set(last_tokens)) == 1:
            bonus += 0.20

        if all(" " not in w for w in words):
            for glue, completions in PHRASE_GLUERS_PREFIX.items():
                if wset.issubset(completions):
                    bonus += 0.22
                    break
            for glue, completions in PHRASE_GLUERS_SUFFIX.items():
                if wset.issubset(completions):
                    bonus += 0.22
                    break

        gap = self.outlier_gap(group)
        if gap > 0.1:
            bonus -= 0.05

        return bonus
    
    def group_pair_sims(self, group: Group) -> list[float]:
        v = [self.cache[w] for w in group]
        pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        return [float(np.dot(v[a], v[b])) for a, b in pairs]

    def outlier_gap(self, group: Group) -> float:
        v = [self.cache[w] for w in group]
        pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        per = [0.0, 0.0, 0.0, 0.0]
        cnt = [0, 0, 0, 0]
        for a, b in pairs:
            sim = float(np.dot(v[a], v[b]))
            per[a] += sim; per[b] += sim
            cnt[a] += 1;  cnt[b] += 1
        per = [per[i] / cnt[i] for i in range(4)]
        return max(per) - min(per)

    def score_group(self, group: Group) -> float:
        """
        Score a 4-word group using average pairwise cosine similarity.
        There are 6 pairs in a 4-element group.
        """
        v = [self.cache[w] for w in group]

        # Pairwise similarities: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
        sims = [
            self.cosine(v[0], v[1]),
            self.cosine(v[0], v[2]),
            self.cosine(v[0], v[3]),
            self.cosine(v[1], v[2]),
            self.cosine(v[1], v[3]),
            self.cosine(v[2], v[3]),
        ]
        avg = sum(sims) / len(sims)
        min_pair = min(sims)

        semantic = 0.75 * avg + 0.25 * min_pair
        heuristic = self.heuristic_bonus(group)

        return semantic + heuristic

    def score_all_groups(self, groups: List[Group]) -> List[Tuple[Group, float]]:

        # Score every group and return list of (group, score)
        scored: List[Tuple[Group, float]] = []
        for g in groups:
            scored.append((g, self.score_group(g)))
        return scored
    
    def explain_group(self, group: Group) -> dict:
        sims = self.group_pair_sims(group)
        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)

        gap = self.outlier_gap(group)

        # Heuristic breakdown 
        bonus = 0.0
        details = {}

        lengths = {len(w) for w in group}
        same_len = (len(lengths) == 1)
        if same_len:
            bonus += 0.05
        details["same_length_bonus"] = 0.05 if same_len else 0.0

        plural_flags = [w.endswith("S") for w in group]
        plural_ok = (all(plural_flags) or not any(plural_flags))
        if plural_ok:
            bonus += 0.05
        details["plural_consistency_bonus"] = 0.08 if plural_ok else 0.0

        outlier_pen = 0.0
        if gap > 0.10:
            outlier_pen = -0.05
            bonus += outlier_pen
        details["outlier_penalty"] = outlier_pen
        details["outlier_gap"] = round(gap, 3)

        semantic = 0.75 * avg_sim + 0.25 * min_sim
        final = semantic + bonus


        return {
            "avg_pair_similarity": round(avg_sim, 3),
            "min_pair_similarity": round(min_sim, 3),
            "heuristic_total": round(bonus, 3),
            "final_score": round(final, 3),
            "heuristics": details,
        }

    