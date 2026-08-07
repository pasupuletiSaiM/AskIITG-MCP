from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from langsmith import traceable

from config import Config
from models import ModelPool


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


class CampusLifeRetriever:
    """
    Production retriever for the campus_life_info tool.

    Pipeline:
        Query
            ↓
        Embed (BGE)
            ↓
        Qdrant ANN Search
            ↓
        CrossEncoder Rerank
            ↓
        Return Top-K
    """

    _client = None

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

    @traceable(name="qdrant_retrieve_and_rerank")
    def search(
        self,
        query: str,
        k: int = Config.DEFAULT_K,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:

        # -------------------------------------------------
        # 1. Embed Query
        # -------------------------------------------------

        embedder = ModelPool.embedder()

        query_vector = embedder.encode(
            Config.BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
        ).tolist()

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

            return [
                RetrievedChunk(
                    text=texts[i],
                    score=float(scores[i]),
                    metadata={
                        k: v
                        for k, v in payloads[i].items()
                        if k != "text"
                    },
                )
                for i in order[:k]
            ]

        # -------------------------------------------------
        # 4. Without Reranking
        # -------------------------------------------------

        order = np.argsort(-np.asarray(dense_scores))

        return [
            RetrievedChunk(
                text=texts[i],
                score=float(dense_scores[i]),
                metadata={
                    k: v
                    for k, v in payloads[i].items()
                    if k != "text"
                },
            )
            for i in order[:k]
        ]