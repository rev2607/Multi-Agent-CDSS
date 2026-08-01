"""Sparse (BM25-style) vector encoding for hybrid search.

We map tokens to stable integer indices and use TF weights so Qdrant
sparse vectors can score lexical medical terms (drug names, ICD-like codes, etc.).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List

# Large vocabulary space for hashed sparse features
SPARSE_VOCAB = 2**18  # 262144


def tokenize(text: str) -> List[str]:
    # Keep alphanumerics and common medical separators (e.g. beta-blocker, 5-HT)
    raw = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-\+/]*", text.lower())
    return [t for t in raw if len(t) > 1 or t.isdigit()]


def token_index(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % SPARSE_VOCAB


class SparseEncoder:
    """Encode text into Qdrant-compatible sparse vectors {indices, values}."""

    def encode(self, text: str) -> Dict[str, List]:
        tokens = tokenize(text)
        if not tokens:
            return {"indices": [0], "values": [1.0]}
        counts = Counter(tokens)
        dl = sum(counts.values())
        # BM25-ish term weights without corpus DF (local demo: TF with length norm)
        avgdl = 200.0
        k1, b = 1.5, 0.75
        indices: List[int] = []
        values: List[float] = []
        # Merge collisions by summing weights
        bucket: Dict[int, float] = {}
        for tok, tf in counts.items():
            idx = token_index(tok)
            # Simple IDF proxy from token length / hash (stable, not corpus-based)
            idf = 1.0 + math.log(1.0 + (len(tok) / 3.0))
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            weight = idf * (tf * (k1 + 1)) / denom
            bucket[idx] = bucket.get(idx, 0.0) + float(weight)
        for idx, val in sorted(bucket.items()):
            indices.append(idx)
            values.append(val)
        return {"indices": indices, "values": values}

    def encode_batch(self, texts: List[str]) -> List[Dict[str, List]]:
        return [self.encode(t) for t in texts]
