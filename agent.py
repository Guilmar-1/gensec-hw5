"""Interactive LangChain agent for the local Study Notes MCP server."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient


# Resolve paths from this file so the command works from the project directory
# without requiring the project to live at a particular absolute path.
PROJECT_DIR = Path(__file__).resolve().parent
MCP_SERVER_PATH = PROJECT_DIR / "mcp-server.py"
MEMORY_FILE_PATH = PROJECT_DIR / "memory.jsonl"


async def build_agent() -> tuple[Any, MultiServerMCPClient]:
    """Connect both MCP servers and build a Gemini-powered study assistant."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Set it in your shell before starting the agent."
        )
    if not MCP_SERVER_PATH.is_file():
        raise FileNotFoundError(f"Study Notes MCP server not found: {MCP_SERVER_PATH}")

    # This starts mcp-server.py with the Python interpreter running this agent.
    # When launched from .venv/bin/python, this is the HW4 virtual environment.
    mcp_client = MultiServerMCPClient(
        {
            "study_notes": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(MCP_SERVER_PATH)],
            },
            # This official package release runs on the installed Node 20 runtime.
            # The current 2.x release requires Node 24 or later.
            "memory": {
                "transport": "stdio",
                "command": "npx",
                "args": ["--yes", "@modelcontextprotocol/server-memory@1.0.0"],
                "env": {"MEMORY_FILE_PATH": str(MEMORY_FILE_PATH)},
            },
        }
    )
    tools = await mcp_client.get_tools()

    # Set GEMINI_MODEL to override the default model name.
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a helpful study assistant. Use the Study Notes tools to answer "
            "questions from the user's saved notes. Search notes before answering "
            "questions about their saved material. Create notes when the user asks "
            "you to save or remember information. Use Memory tools to maintain a "
            "knowledge graph: create_entities adds entities with initial observations, "
            "add_observations appends facts to existing entities, create_relations "
            "connects entities, search_nodes finds matching graph entries, "
            "open_nodes reads named entities and their relationships, and read_graph "
            "reads the complete graph. Use both kinds of storage when asked. Confirm "
            "changes clearly, and never claim an action succeeded unless its tool "
            "result confirms success."
        ),
    )
    return agent, mcp_client


async def ask(agent: Any, question: str) -> str:
    """Send one message to the agent and return its final response text."""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    final_message = result["messages"][-1]
    content = final_message.content
    if isinstance(content, str):
        return content
    # Gemini can return structured content blocks; render them readably.
    return "\n".join(
        str(block.get("text", block)) if isinstance(block, dict) else str(block)
        for block in content
    )


async def main() -> None:
    """Run a simple interactive study-notes chat in the terminal."""
    try:
        agent, _mcp_client = await build_agent()
    except Exception as exc:
        raise SystemExit(f"Could not start the Study Notes agent: {exc}") from exc

    print("Study Notes assistant ready. Type 'quit' or 'exit' to stop.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            print(f"Assistant: {await ask(agent, question)}")
        except Exception as exc:
            print(f"Assistant: I couldn't complete that request: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
