"""Exact Okapi BM25 over passage bodies; no questions enter the index."""
import math
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .common import RetrievalResult, make_result, validate_passages

LEXICAL_PATTERN = r"[^\W_]+"
LEXICAL_DESCRIPTION = (
    "Unicode casefold followed by Unicode alphanumeric runs (underscores excluded); "
    "punctuation splits tokens; no stopword removal or stemming; body text only."
)


def lexical_tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    return re.findall(LEXICAL_PATTERN, text.casefold(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, passages, *, k1=1.5, b=0.75, epsilon=0.25):
        if not all(math.isfinite(v) for v in (k1, b, epsilon)):
            raise ValueError("BM25 parameters must be finite")
        if k1 <= 0 or not 0 <= b <= 1 or epsilon < 0:
            raise ValueError("Require k1 > 0, 0 <= b <= 1, epsilon >= 0")
        # Sorting once makes stable score sorting resolve ties by passage ID,
        # independent of input record order.
        self.passages = sorted(validate_passages(passages), key=lambda p: p["passage_id"])
        tokens = [lexical_tokens(p["text"]) for p in self.passages]
        if not any(tokens):
            raise ValueError("Corpus contains no lexical tokens")
        self.index = BM25Okapi(tokens, k1=k1, b=b, epsilon=epsilon)
        self.configuration = {"implementation": "rank_bm25.BM25Okapi", "k1": k1,
                              "b": b, "epsilon": epsilon,
                              "lexical_preprocessing": LEXICAL_DESCRIPTION,
                              "tie_break": "passage_id ascending"}

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        tokens = lexical_tokens(query)
        if not tokens:
            raise ValueError("Query must contain at least one lexical token")
        scores = self.index.get_scores(tokens)
        if not np.isfinite(scores).all():
            raise ValueError("BM25 returned nonfinite scores")
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [make_result(self.passages[int(i)], scores[i], rank)
                for rank, i in enumerate(order, start=1)]
