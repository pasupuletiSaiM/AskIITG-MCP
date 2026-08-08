from mcp.server.fastmcp import FastMCP
from langsmith import traceable

from config import Config
from retriever import CampusLifeRetriever
from observe import observe
from models import ModelPool


mcp = FastMCP(
    name="campus-life",
    host=Config.HOST,
    port=Config.PORT,
)

retriever = CampusLifeRetriever()


@traceable(name="campus_life_info_mcp_tool")
@observe("campus_life_info")
@mcp.tool()
def campus_life_info(
    query: str,
    k: int = Config.DEFAULT_K,
):
    """
    Search IIT Guwahati campus life information.

    Examples
    --------
    • Hostels
    • Clubs
    • Sports
    • Techniche
    • Alcheringa
    • Gymkhana
    • SAC
    • Hospital
    """

    results = retriever.search(
        query=query,
        k=k,
    )

    return {
        "query": query,
        "results": [
            {
                "text": chunk.text,
                "score": round(chunk.score, 4),
                "metadata": chunk.metadata,
            }
            for chunk in results
        ],
    }


if __name__ == "__main__":
    print("⏳ Pre-warming models (this may take a minute on first run)...")
    ModelPool.embedder()
    ModelPool.reranker()
    print("✅ Models ready. Starting MCP server...")
    mcp.run(
        transport="streamable-http",
    )