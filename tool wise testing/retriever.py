from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from langsmith import traceable

from config import Config
from models import ModelPool
from cache import QueryEmbeddingCache, ResultCache, SemanticCache


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


class CampusLifeRetriever:
    """
    Production retriever for the campus_life_info tool.

    Pipeline (cache miss):
        Query
            ↓
        [L2 Result Cache miss]
            ↓
        [L1 Embedding Cache miss] → Embed (BGE) → store in L1
            ↓
        Qdrant ANN Search
            ↓
        CrossEncoder Rerank
            ↓
        Store in L2 → Return Top-K

    Pipeline (L2 hit – fastest path):
        Query → L2 hit → Return cached results  (~0 ms)

    Pipeline (L1 hit only):
        Query → L2 miss → L1 hit → Qdrant + Rerank → store in L2 → Return
    """

    _client = None

    # ------------------------------------------------------------------
    # Shared cache instances (process-level singletons)
    # ------------------------------------------------------------------

    _embed_cache = QueryEmbeddingCache(maxsize=Config.EMBED_CACHE_MAXSIZE)
    _result_cache = ResultCache(
        ttl_seconds=Config.RESULT_CACHE_TTL_SECONDS,
    )
    _semantic_cache = SemanticCache(
        similarity_threshold=Config.SEMANTIC_CACHE_THRESHOLD,
        maxsize=Config.SEMANTIC_CACHE_MAXSIZE,
    )

    @classmethod
    def client(cls):
        """
        Create a single shared Qdrant client.
        """
        if cls._client is None:
            cls._client = QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY,
                timeout=30,
            )

        return cls._client

    @classmethod
    def cache_stats(cls) -> dict:
        """Return hit-rate statistics for all cache layers."""
        return {
            "embed_cache":    cls._embed_cache.stats,
            "result_cache":   cls._result_cache.stats,
            "semantic_cache": cls._semantic_cache.stats,
        }

    @traceable(name="qdrant_retrieve_and_rerank")
    def search(
        self,
        query: str,
        k: int = Config.DEFAULT_K,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:

        # -------------------------------------------------
        # L2 – Full Result Cache (fastest path)
        # -------------------------------------------------

        cached_result = self._result_cache.get(query, k, rerank)
        if cached_result is not None:
            return cached_result

        # -------------------------------------------------
        # 1. Embed Query  (L1 – Embedding Cache)
        # -------------------------------------------------

        embedder = ModelPool.embedder()

        query_vector = self._embed_cache.get(query)
        if query_vector is None:
            query_vector = embedder.encode(
                Config.BGE_QUERY_PREFIX + query,
                normalize_embeddings=True,
            ).tolist()
            self._embed_cache.set(query, query_vector)

        # -------------------------------------------------
        # L3 – Semantic Cache
        # -------------------------------------------------

        semantic_result = self._semantic_cache.get(query_vector, k, rerank)
        if semantic_result is not None:
            return semantic_result

        # -------------------------------------------------
        # 2. Dense Search (Qdrant)
        # -------------------------------------------------

        response = self.client().query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_vector,
            limit=Config.CANDIDATES,
            with_payload=True,
        )

        hits = response.points

        if not hits:
            # Cache the empty result so we don't hammer Qdrant again
            self._result_cache.set(query, k, rerank, [])
            return []

        texts = []
        payloads = []
        dense_scores = []

        for point in hits:

            payload = point.payload or {}

            texts.append(payload.get("text", ""))

            payloads.append(payload)

            dense_scores.append(point.score)

        # -------------------------------------------------
        # 3. Cross Encoder Reranking
        # -------------------------------------------------

        if rerank:

            reranker = ModelPool.reranker()

            scores = reranker.predict(
                [
                    (query, text)
                    for text in texts
                ]
            )

            order = np.argsort(-np.asarray(scores))

            results = [
                RetrievedChunk(
                    text=texts[i],
                    score=float(scores[i]),
                    metadata={
                        key: v
                        for key, v in payloads[i].items()
                        if key != "text"
                    },
                )
                for i in order[:k]
            ]

            self._result_cache.set(query, k, rerank, results)
            self._semantic_cache.register(query_vector, query, k, rerank)
            return results

        # -------------------------------------------------
        # 4. Without Reranking
        # -------------------------------------------------

        order = np.argsort(-np.asarray(dense_scores))

        results = [
            RetrievedChunk(
                text=texts[i],
                score=float(dense_scores[i]),
                metadata={
                    key: v
                    for key, v in payloads[i].items()
                    if key != "text"
                },
            )
            for i in order[:k]
        ]

        self._result_cache.set(query, k, rerank, results)
        self._semantic_cache.register(query_vector, query, k, rerank)
        return results