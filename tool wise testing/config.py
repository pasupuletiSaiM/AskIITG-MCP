from dotenv import load_dotenv
import os

load_dotenv()

class Config:

    #------------------------ Server ------------------------
    
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    
    #------------------------ Qdrant ------------------------
    
    QDRANT_URL= os.getenv("QDRANT_URL")
    QDRANT_API_KEY= os.getenv("QDRANT_API_KEY")
    
    #------------------------ Collection ------------------------
    
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    
    #------------------------ models ------------------------
    
    EMBED_MODEL = "BAAI/bge-base-en-v1.5"
    RERANK_MODEL = "BAAI/bge-reranker-base"
    BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
    )

    #------------------------ Retrieval ------------------------

    DEFAULT_K = 5
    CANDIDATES = 20

    #------------------------ logfire ------------------------

    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_SERVICE = os.getenv("LOGFIRE_SERVICE")

    #------------------------ langsmith ------------------------

    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
