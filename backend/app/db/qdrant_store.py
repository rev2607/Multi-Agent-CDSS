"""Local Qdrant store with hybrid dense + sparse vectors."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import settings

logger = logging.getLogger(__name__)

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"


class ScoredHit:
    """Minimal scored-point shim compatible with hybrid retriever."""

    __slots__ = ("id", "score", "payload")

    def __init__(self, id: Any, score: float, payload: Optional[Dict[str, Any]]) -> None:
        self.id = id
        self.score = score
        self.payload = payload or {}


class QdrantStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(path=str(settings.qdrant_path))
        # Create only if missing; prefer known dim, else wait for first upsert
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            self._sync_dim_from_collection()
        else:
            self._ensure_collection(settings.dense_vector_size)

    def _sync_dim_from_collection(self) -> None:
        try:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                dim = vectors[DENSE_NAME].size
            else:
                dim = vectors.size  # type: ignore[union-attr]
            object.__setattr__(settings, "dense_vector_size", dim)
        except Exception as e:
            logger.debug("Could not sync dense dim: %s", e)

    def _ensure_collection(self, dim: int) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            return
        logger.info("Creating Qdrant collection %s (dense dim=%s)", self.collection, dim)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_NAME: qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=False),
                )
            },
        )
        object.__setattr__(settings, "dense_vector_size", dim)

    def recreate_if_dim_mismatch(self, dim: int) -> None:
        """If embedding dim differs from collection, recreate (local demo only)."""
        try:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                current = vectors[DENSE_NAME].size
            else:
                current = vectors.size  # type: ignore[union-attr]
            if current == dim:
                return
            logger.warning(
                "Recreating collection %s: dim %s -> %s", self.collection, current, dim
            )
            self.client.delete_collection(self.collection)
            object.__setattr__(settings, "dense_vector_size", dim)
            self._ensure_collection(dim)
        except Exception as e:
            logger.warning("dim check failed: %s", e)

    def upsert_chunks(
        self,
        *,
        texts: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[Dict[str, List]],
        payloads: Sequence[Dict[str, Any]],
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if not texts:
            return []
        dim = len(dense_vectors[0])
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection not in names:
            self._ensure_collection(dim)
        elif dim != settings.dense_vector_size:
            self.recreate_if_dim_mismatch(dim)

        point_ids = list(ids) if ids else [str(uuid.uuid4()) for _ in texts]
        points: List[qm.PointStruct] = []
        for i, text in enumerate(texts):
            sparse = sparse_vectors[i]
            payload = dict(payloads[i])
            payload["text"] = text
            points.append(
                qm.PointStruct(
                    id=point_ids[i],
                    vector={
                        DENSE_NAME: list(dense_vectors[i]),
                        SPARSE_NAME: qm.SparseVector(
                            indices=list(sparse["indices"]),
                            values=list(sparse["values"]),
                        ),
                    },
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)
        return point_ids

    def search_dense(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 12,
        filters: Optional[qm.Filter] = None,
    ) -> List[ScoredHit]:
        resp = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            using=DENSE_NAME,
            limit=limit,
            query_filter=filters,
            with_payload=True,
        )
        return [
            ScoredHit(id=p.id, score=float(p.score or 0.0), payload=p.payload)
            for p in (resp.points or [])
        ]

    def search_sparse(
        self,
        sparse: Dict[str, List],
        *,
        limit: int = 12,
        filters: Optional[qm.Filter] = None,
    ) -> List[ScoredHit]:
        resp = self.client.query_points(
            collection_name=self.collection,
            query=qm.SparseVector(
                indices=list(sparse["indices"]),
                values=list(sparse["values"]),
            ),
            using=SPARSE_NAME,
            limit=limit,
            query_filter=filters,
            with_payload=True,
        )
        return [
            ScoredHit(id=p.id, score=float(p.score or 0.0), payload=p.payload)
            for p in (resp.points or [])
        ]

    def count(self) -> int:
        try:
            return int(self.client.count(self.collection).count)
        except Exception:
            return 0


@lru_cache
def get_qdrant_store() -> QdrantStore:
    return QdrantStore()
