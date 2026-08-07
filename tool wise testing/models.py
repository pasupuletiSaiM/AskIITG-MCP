from sentence_transformers import SentenceTransformer, CrossEncoder

from config import Config


class ModelPool:
    """
    Loads models only once and reuses them for every request.
    """

    _embedder = None
    _reranker = None

    @classmethod
    def embedder(cls):

        if cls._embedder is None:
            print("Loading embedding model...")
            cls._embedder = SentenceTransformer(
                Config.EMBED_MODEL
            )

        return cls._embedder

    @classmethod
    def reranker(cls):

        if cls._reranker is None:
            print("Loading reranker...")
            cls._reranker = CrossEncoder(
                Config.RERANK_MODEL
            )

        return cls._reranker