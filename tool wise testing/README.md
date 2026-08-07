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
CampusLifeRetriever
        │
        ▼
Qdrant Cloud
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
├── retriever.py       # Qdrant retrieval pipeline
├── models.py          # Embedding & reranker models
├── observe.py         # Logfire tracing
├── config.py          # Configuration
├── .env.example       # Environment variables template
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
Generate Embedding (BGE)
      │
      ▼
Qdrant Dense Search
      │
      ▼
CrossEncoder Reranking
      │
      ▼
Top K Chunks
      │
      ▼
Return to Claude
```

---

# Prerequisites

- Python 3.11+
- uv
- Cloudflare Tunnel (`cloudflared`)
- Claude Desktop
- Qdrant Cloud account

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

Copy the example file.

```bash
cp .env.example .env
```

Fill the values:

```text
QDRANT_URL=

QDRANT_API_KEY=

COLLECTION_NAME=

LOGFIRE_TOKEN=

LOGFIRE_SERVICE=campus-life-mcp
```

---

## 4. Start the MCP Server

```bash
uv run server.py
```

The server starts on

```
http://localhost:8000
```

---

# Expose using Cloudflare Tunnel

Open another terminal and run

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare will generate a public URL similar to

```
https://xxxxx.trycloudflare.com
```

---

# Connect to Claude Desktop

Add the remote MCP server using the generated Cloudflare URL.

Example:

```json
{
  "mcpServers": {
    "campus-life": {
      "url": "https://xxxxx.trycloudflare.com/mcp"
    }
  }
}
```

Restart Claude Desktop.

The tool **campus_life_info** should now appear.

---

# Example Query

```
Tell me about Kapili Hostel.

```

or

```
What clubs are available at IIT Guwahati?

```

or

```
What sports facilities are available on campus?

```

---

# Logging

The server uses **Logfire** for observability.

It records:

- Tool calls
- Query execution
- Latency
- Exceptions

---

# Technologies Used

- FastMCP
- Qdrant Cloud
- Sentence Transformers (BGE Base)
- BGE CrossEncoder Reranker
- Logfire
- Cloudflare Tunnel
- uv

---

# Future Improvements

- Multiple MCP tools
- Metadata filtering
- Streaming responses
- Docker support
- Render deployment
