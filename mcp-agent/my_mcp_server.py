from mcp.server.fastmcp import FastMCP

import os, sys

# Go up one level from mcp-agent/ to AI_Lab/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from setup.config import TAVILY_API_KEY

# 🏗️ Create the server
mcp = FastMCP("My First MCP Server")

# Save notes in your project folder
NOTES_FILE = "/Users/nagarjunamaddi/PycharmProjects/AI_Lab/mcp-agent/notes.txt"

# 🛠️ Tool 1 — expose search as an MCP tool
@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for real-time, current, or breaking information
    that may not be in training data, such as today's news,
    live scores, current weather, or recent announcements."""
    from tavily import TavilyClient
    import os
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily.search(
        query=query,
        max_results=3,
        include_answer=True,
        include_raw_content=False
    )
    if response.get("answer"):
        return response["answer"]
    results = [r["content"] for r in response["results"]]
    return "\n\n".join(results)


# 🛠️ Tool 2 — expose a simple calculator
@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression like '15 + 30'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# 🛠️ Tool 3 — expose memory save
@mcp.tool()
def save_note(note: str) -> str:
    """Save a note to a file."""
    with open(NOTES_FILE, "a") as f:
        f.write(note + "\n")
    return "Note saved!"


# 🛠️ Tool 4 — expose memory read
@mcp.tool()
def read_notes() -> str:
    """Read all saved notes."""
    if not __import__("os").path.exists(NOTES_FILE):
        return "No notes found."
    with open(NOTES_FILE, "r") as f:
        return f.read()


# ▶️ Run the server
if __name__ == "__main__":
    mcp.run()
