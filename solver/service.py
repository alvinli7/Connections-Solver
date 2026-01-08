from __future__ import annotations
from .scorer import SemanticScorer

# Load once when the server imports this module
SCORER = SemanticScorer()
