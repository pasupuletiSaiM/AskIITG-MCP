"""
cache.py – Hybrid three-layer cache for the CampusLife retriever.

L1  QueryEmbeddingCache  – In-memory LRU (process-local).
                           Embeddings are deterministic per model, so sharing
                           across instances adds no value, and Redis RTT would
                           actually slow down what we're trying to skip.

L2  ResultCache          – Upstash Redis exact-match cache (REST API).
                           Caches complete list[RetrievedChunk] results keyed
                           by sha256(query|k|rerank). Shared across instances,
                           survives restarts.

L3  SemanticCache        – Local cosine-similarity index + Upstash Redis.
                           Maintains an in-memory list of (embedding, redis_key)
                           pairs. On lookup, computes dot product similarity
                           (= cosine similarity for L2-normalised BGE vectors)
                           and returns the cached result if similarity exceeds
                           the configured threshold. Catches queries that are
                           semantically identical but phrased differently.

Usage
-----
    from cache import QueryEmbeddingCache, ResultCache, SemanticCache

    emb_cache      = QueryEmbeddingCache(maxsize=512)
    res_cache      = ResultCache(ttl_seconds=300)
    semantic_cache = SemanticCache(similarity_threshold=0.92, maxsize=256)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

from upstash_redis import Redis

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_key(raw: str) -> str:
    """Stable sha256 hex key, prefixed for namespace isolation."""
    return "clr:" + hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# L1 – In-Memory Embedding Cache  (unchanged from original)
# ---------------------------------------------------------------------------

class _LRUCache:
    """Thread-safe in-memory LRU cache (no TTL — embeddings are deterministic)."""

    def __init__(self, maxsize: int = 512):
        self._maxsize = maxsize
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "backend": "in-memory",
                "size": len(self._store),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


class QueryEmbeddingCache:
    """
    L1 cache: query string -> BGE embedding vector (list[float]).
    In-memory LRU, no TTL (embeddings never go stale for the same model).
    """

    def __init__(self, maxsize: int = 512):
        self._cache = _LRUCache(maxsize=maxsize)

    def get(self, query: str) -> Optional[list]:
        return self._cache.get(_make_key(query))

    def set(self, query: str, vector: list) -> None:
        self._cache.set(_make_key(query), vector)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> dict:
        return {"layer": "L1_embedding", **self._cache.stats}


# ---------------------------------------------------------------------------
# L2 – Upstash Redis Result Cache
# ---------------------------------------------------------------------------

class ResultCache:
    """
    L2 cache: (query, k, rerank) -> list[RetrievedChunk].

    Backed by Upstash Redis via their REST API (no TCP socket, works anywhere).
    Results are JSON-serialised so they survive process restarts and are shared
    across multiple server instances.

    Parameters
    ----------
    ttl_seconds : int
        Redis EXPIRE for each entry (default 300 = 5 min).
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._redis = Redis(
            url=Config.UPSTASH_REDIS_REST_URL,
            token=Config.UPSTASH_REDIS_REST_TOKEN,
        )
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Key
    # ------------------------------------------------------------------

    def _key(self, query: str, k: int, rerank: bool) -> str:
        raw = f"{query}|k={k}|rerank={rerank}"
        return _make_key(raw)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(results: list) -> str:
        """Convert list[RetrievedChunk] -> JSON string."""
        return json.dumps(
            [dataclasses.asdict(chunk) for chunk in results]
        )

    @staticmethod
    def _deserialize(data: str) -> list:
        """Convert JSON string -> list[RetrievedChunk]."""
        # Import here to avoid circular import
        from retriever import RetrievedChunk
        return [
            RetrievedChunk(**item)
            for item in json.loads(data)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str, k: int, rerank: bool) -> Optional[list]:
        try:
            raw = self._redis.get(self._key(query, k, rerank))
        except Exception as e:
            # Redis unavailable — degrade gracefully, log for visibility
            logger.warning("ResultCache GET failed (degrading to miss): %s", e)
            raw = None

        with self._lock:
            if raw is None:
                self._misses += 1
                return None
            self._hits += 1

        return self._deserialize(raw)

    def set(self, query: str, k: int, rerank: bool, results: list) -> None:
        try:
            self._redis.set(
                self._key(query, k, rerank),
                self._serialize(results),
                ex=self._ttl,
            )
        except Exception:
            pass  # Degrade gracefully if Redis is unreachable

    def invalidate(self, query: str, k: int, rerank: bool) -> None:
        try:
            self._redis.delete(self._key(query, k, rerank))
        except Exception:
            pass

    def clear(self) -> None:
        """Flush all clr:* keys — only touches this app's namespace."""
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match="clr:*", count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass
        with self._lock:
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "layer": "L2_results",
                "backend": "upstash-redis",
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


# ---------------------------------------------------------------------------
# L3 – Semantic Cache (cosine-similarity index + Upstash Redis)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402 — numpy is a project dependency


class SemanticCache:
    """
    L3 cache: finds semantically similar cached queries.

    Maintains a local in-memory list of (embedding_vector, redis_key) pairs.
    On lookup, computes cosine similarity between the incoming query embedding
    and every stored embedding. Since BGE normalises its output vectors
    (normalize_embeddings=True), cosine similarity equals the dot product.

    If the best match exceeds `similarity_threshold`, the result is fetched
    from Upstash Redis using the matched key. ResultCache already stored the
    result there during the original pipeline run — no duplicate Redis write.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity to count as a cache hit (default 0.92).
    maxsize : int
        Maximum number of entries in the local index (default 256).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        maxsize: int = 256,
    ):
        self._threshold = similarity_threshold
        self._maxsize = maxsize
        self._redis = Redis(
            url=Config.UPSTASH_REDIS_REST_URL,
            token=Config.UPSTASH_REDIS_REST_TOKEN,
        )
        self._lock = threading.Lock()
        # Each entry: (L2-normalised np.ndarray float32, redis_key str)
        self._index: list[tuple[np.ndarray, str]] = []
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Key — must match ResultCache._key() exactly so we reuse its data
    # ------------------------------------------------------------------

    @staticmethod
    def _key(query: str, k: int, rerank: bool) -> str:
        raw = f"{query}|k={k}|rerank={rerank}"
        return _make_key(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query_vector: list, k: int, rerank: bool) -> Optional[list]:
        """
        Return cached results for the most similar stored query, or None.
        """
        with self._lock:
            if not self._index:
                self._misses += 1
                return None
            index_snapshot = list(self._index)

        vec = np.array(query_vector, dtype=np.float32)

        best_sim = -1.0
        best_key: Optional[str] = None
        for stored_vec, redis_key in index_snapshot:
            sim = float(np.dot(vec, stored_vec))
            if sim > best_sim:
                best_sim = sim
                best_key = redis_key

        if best_sim < self._threshold or best_key is None:
            with self._lock:
                self._misses += 1
            return None

        # Fetch the result from Redis using the matched key
        try:
            raw = self._redis.get(best_key)
        except Exception as e:
            logger.warning("SemanticCache Redis GET failed: %s", e)
            with self._lock:
                self._misses += 1
            return None

        if raw is None:
            # Redis entry expired — remove stale index entry
            with self._lock:
                self._index = [(v, k) for v, k in self._index if k != best_key]
                self._misses += 1
            return None

        with self._lock:
            self._hits += 1

        logger.info("SemanticCache HIT (cosine_sim=%.4f)", best_sim)
        return ResultCache._deserialize(raw)

    def register(self, query_vector: list, query: str, k: int, rerank: bool) -> None:
        """
        Register a new (embedding, key) pair after a full pipeline run.

        ResultCache.set() already stored the result in Redis — this method
        just adds the embedding to the local similarity index so future
        semantically similar queries can find it.
        """
        key = self._key(query, k, rerank)
        vec = np.array(query_vector, dtype=np.float32)
        with self._lock:
            self._index.append((vec, key))
            if len(self._index) > self._maxsize:
                self._index.pop(0)  # evict oldest entry

    def clear(self) -> None:
        with self._lock:
            self._index.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "layer": "L3_semantic",
                "backend": "local-cosine-index + upstash-redis",
                "threshold": self._threshold,
                "index_size": len(self._index),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }
