# Campus Life MCP Server

An MCP (Model Context Protocol) server that provides information about IIT Guwahati campus life using a Retrieval-Augmented Generation (RAG) pipeline backed by Qdrant Cloud.

The server exposes a single MCP tool:

- **campus_life_info** – Retrieves relevant campus life information from Qdrant and returns it to the MCP client (Claude Desktop).

---

# Architecture

```
Claude Desktop
        │
        ▼
Cloudflare Tunnel
        │
        ▼
FastMCP HTTP Server
        │
        ▼
  ┌─────────────────────────────┐
  │      CampusLifeRetriever    │
  │                             │
  │  L2 Result Cache (Redis) ──►│──► Cache Hit → Return instantly
  │         │ miss              │
  │  L1 Embed Cache (memory) ──►│──► Cache Hit → Skip BGE inference
  │         │ miss              │
  │  BGE Embed → Qdrant → Rerank│
  └─────────────────────────────┘
        │
        ▼
    Qdrant Cloud
        │
        ▼
  Upstash Redis
        │
        ▼
   Retrieved Chunks
        │
        ▼
   Claude Desktop
```

---

# Project Structure

```
campus-life-mcp/
│
├── server.py          # FastMCP server and MCP tool
├── retriever.py       # Qdrant retrieval pipeline with cache integration
├── cache.py           # Two-layer cache (in-memory L1 + Upstash Redis L2)
├── models.py          # Embedding & reranker models (BGE)
├── observe.py         # Logfire tracing
├── config.py          # Configuration & environment variables
├── .env               # Environment variables (never commit this)
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Retrieval Pipeline

```
User Query
      │
      ▼
L2 Result Cache (Upstash Redis)
      │ HIT → return in ~1ms (skips everything below)
      │ MISS
      ▼
L1 Embedding Cache (in-memory LRU)
      │ HIT → skip BGE inference
      │ MISS
      ▼
BGE Embed Query
      │
      ▼
Qdrant Dense Search
      │
      ▼
CrossEncoder Reranking
      │
      ▼
Store in L1 & L2 Cache
      │
      ▼
Top K Chunks → Return to Claude
```

---

# Cache Design

## L1 — Query Embedding Cache (In-Memory LRU)

| Property | Value |
|---|---|
| Backend | Python `OrderedDict` (process memory) |
| Key | `sha256(query)` |
| Value | BGE embedding vector (`list[float]`) |
| Capacity | 512 entries |
| TTL | None (embeddings are deterministic) |
| Shared across instances | ❌ |
| Survives restart | ❌ |

## L2 — Result Cache (Upstash Redis)

| Property | Value |
|---|---|
| Backend | Upstash Redis (REST API) |
| Key | `sha256(query \| k \| rerank)` |
| Value | `list[RetrievedChunk]` (JSON serialised) |
| TTL | 300 seconds (5 minutes) |
| Shared across instances | ✅ |
| Survives restart | ✅ |

> **Graceful degradation**: if Upstash Redis is unreachable, the server continues to function normally — it simply falls through to Qdrant on every request.

---

# Prerequisites

- Python 3.11+
- uv
- Cloudflare Tunnel (`cloudflared`)
- Claude Desktop
- Qdrant Cloud account
- Upstash account (free tier works)

---

# Setup

## 1. Clone the repository

```bash
git clone <repository-url>
cd campus-life-mcp
```

---

## 2. Install dependencies

```bash
uv sync
```

---

## 3. Create `.env`

Fill in the following values:

```text
# Qdrant
QDRANT_URL=
QDRANT_API_KEY=
COLLECTION_NAME=

# Logfire
LOGFIRE_TOKEN=
LOGFIRE_SERVICE=campus-life-mcp

# LangSmith
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=campus-life

# Upstash Redis (L2 result cache)
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
```

### Getting Upstash Redis credentials

1. Go to [console.upstash.com](https://console.upstash.com)
2. Create a new **Redis** database (free tier)
3. Copy **REST URL** and **REST Token** from the database dashboard

---

## 4. Start the MCP Server

```bash
uv run server.py
```

The server starts on:

```
http://localhost:8000
MCP endpoint: http://localhost:8000/mcp
```

---

# Expose using Cloudflare Tunnel

Open another terminal and run:

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare will generate a public URL:

```
https://xxxxx.trycloudflare.com
```

---

# Connect to Claude Desktop

Add the remote MCP server using the generated Cloudflare URL.

```json
{
  "mcpServers": {
    "campus-life": {
      "url": "https://xxxxx.trycloudflare.com/mcp"
    }
  }
}
```

Restart Claude Desktop. The tool **campus_life_info** should now appear.

---

# Example Queries

```
Tell me about Kapili Hostel.
```

```
What clubs are available at IIT Guwahati?
```

```
What sports facilities are available on campus?
```

---

# Observability

## Logfire

Records tool calls, query execution, latency, and exceptions.

## LangSmith

Traces the full retrieval chain including embedding, Qdrant search, and reranking steps.

## Cache Stats

Inspect cache hit rates at runtime:

```python
from retriever import CampusLifeRetriever
print(CampusLifeRetriever.cache_stats())
```

Example output:

```json
{
  "embed_cache": {
    "layer": "L1_embedding",
    "backend": "in-memory",
    "hits": 42,
    "misses": 5,
    "hit_rate": 0.8936
  },
  "result_cache": {
    "layer": "L2_results",
    "backend": "upstash-redis",
    "ttl_seconds": 300,
    "hits": 38,
    "misses": 9,
    "hit_rate": 0.8085
  }
}
```

---

# Technologies Used

- FastMCP
- Qdrant Cloud
- Sentence Transformers (BGE Base `bge-base-en-v1.5`)
- BGE CrossEncoder Reranker (`bge-reranker-base`)
- Upstash Redis (L2 result cache)
- Logfire
- LangSmith
- Cloudflare Tunnel
- uv

---

# Future Improvements

- Multiple MCP tools
- Metadata filtering
- Streaming responses
- Docker support
- Render / Railway deployment
