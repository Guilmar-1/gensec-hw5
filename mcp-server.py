"""SQLite-backed study notes MCP server."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP


# Keep the database beside this script regardless of the server's launch directory.
DATABASE_PATH = Path(__file__).resolve().parent / "study_notes.db"
mcp = FastMCP("Study Notes")


def _connect() -> sqlite3.Connection:
    """Open the local database and ensure its table exists."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            note_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    return connection


def _validated_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    """Require string input, trimming whitespace for titles and IDs."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    cleaned = value.strip()
    if not allow_empty and not cleaned:
        raise ValueError(f"{field} cannot be empty.")
    return cleaned


def _validated_tags(tags: Any) -> list[str]:
    """Normalize tags to a list of non-empty strings."""
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("tags must be a list of strings.")
    normalized = [tag.strip() for tag in tags]
    if any(not tag for tag in normalized):
        raise ValueError("tags cannot contain empty values.")
    return normalized


def _as_note(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a SQLite row into a JSON-friendly note."""
    return {
        "note_id": row["note_id"],
        "title": row["title"],
        "content": row["content"],
        "tags": json.loads(row["tags"]),
    }


@mcp.tool()
def create_note(title: str, content: str, tags: list[str]) -> dict[str, Any]:
    """Create a study note and return it with its generated ID."""
    try:
        title = _validated_text(title, "title")
        content = _validated_text(content, "content", allow_empty=True)
        tags = _validated_tags(tags)
    except ValueError as exc:
        return {"error": str(exc)}

    note = {"note_id": str(uuid.uuid4()), "title": title, "content": content, "tags": tags}
    with _connect() as connection:
        connection.execute(
            "INSERT INTO notes (note_id, title, content, tags) VALUES (?, ?, ?, ?)",
            (note["note_id"], note["title"], note["content"], json.dumps(tags)),
        )
    return note


@mcp.tool()
def list_notes() -> list[dict[str, Any]]:
    """List all saved notes, ordered by title and then ID."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT note_id, title, content, tags FROM notes ORDER BY title COLLATE NOCASE, note_id"
        ).fetchall()
    return [_as_note(row) for row in rows]


@mcp.tool()
def get_note(note_id: str) -> dict[str, Any]:
    """Retrieve one note by its ID."""
    try:
        note_id = _validated_text(note_id, "note_id")
    except ValueError as exc:
        return {"error": str(exc)}
    with _connect() as connection:
        row = connection.execute(
            "SELECT note_id, title, content, tags FROM notes WHERE note_id = ?", (note_id,)
        ).fetchone()
    return _as_note(row) if row else {"error": f"No note found with ID '{note_id}'."}


@mcp.tool()
def search_notes(query: str) -> dict[str, Any]:
    """Search note titles, content, and tags without case sensitivity."""
    try:
        query = _validated_text(query, "query")
    except ValueError as exc:
        return {"error": str(exc), "notes": []}
    pattern = f"%{query}%"
    with _connect() as connection:
        rows = connection.execute(
            """SELECT note_id, title, content, tags FROM notes
               WHERE title LIKE ? COLLATE NOCASE
                  OR content LIKE ? COLLATE NOCASE
                  OR tags LIKE ? COLLATE NOCASE
               ORDER BY title COLLATE NOCASE, note_id""",
            (pattern, pattern, pattern),
        ).fetchall()
    notes = [_as_note(row) for row in rows]
    return {"notes": notes, "message": "" if notes else "No notes matched your search."}


@mcp.tool()
def update_note(note_id: str, title: str, content: str, tags: list[str]) -> dict[str, Any]:
    """Replace all editable fields on an existing note."""
    try:
        note_id = _validated_text(note_id, "note_id")
        title = _validated_text(title, "title")
        content = _validated_text(content, "content", allow_empty=True)
        tags = _validated_tags(tags)
    except ValueError as exc:
        return {"error": str(exc)}

    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE notes SET title = ?, content = ?, tags = ? WHERE note_id = ?",
            (title, content, json.dumps(tags), note_id),
        )
        row = connection.execute(
            "SELECT note_id, title, content, tags FROM notes WHERE note_id = ?", (note_id,)
        ).fetchone()
    return _as_note(row) if cursor.rowcount else {"error": f"No note found with ID '{note_id}'."}


@mcp.tool()
def delete_note(note_id: str) -> dict[str, Any]:
    """Delete a note by its ID."""
    try:
        note_id = _validated_text(note_id, "note_id")
    except ValueError as exc:
        return {"error": str(exc)}
    with _connect() as connection:
        cursor = connection.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
    if cursor.rowcount:
        return {"success": True, "message": f"Deleted note '{note_id}'."}
    return {"error": f"No note found with ID '{note_id}'."}


if __name__ == "__main__":
    # Keep startup banners off stdout so stdio MCP clients receive clean JSON-RPC.
    mcp.run(show_banner=False)
