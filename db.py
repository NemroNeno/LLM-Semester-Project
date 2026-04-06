from __future__ import annotations

import json
import sqlite3
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from settings import DB_PATH, SUMMARY_RECENT_MESSAGE_COUNT, SUMMARY_TRIGGER_TOKENS
from langchain_core.messages.utils import count_tokens_approximately


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT, context_count INTEGER, rag_references TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    if "rag_references" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN rag_references TEXT")
    conn.commit()
    conn.close()


def list_chats() -> list[dict[str, str]]:
    conn = get_db()
    chats = conn.execute("SELECT id, title FROM chats ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"id": chat["id"], "title": chat["title"]} for chat in chats]


def parse_rag_references(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        return []
    return []


def list_chat_messages(chat_id: str) -> list[dict[str, Any]]:
    conn = get_db()
    messages = conn.execute(
        "SELECT role, content, context_count, rag_references FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
        (chat_id,),
    ).fetchall()
    conn.close()

    return [
        {
            "role": message["role"],
            "content": message["content"],
            "context_count": message["context_count"],
            "rag_references": parse_rag_references(message["rag_references"]),
        }
        for message in messages
    ]


def should_summarize_next_turn(chat_id: str | None, message: str) -> bool:
    if not chat_id:
        return False

    conn = get_db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
        (chat_id,),
    ).fetchall()
    conn.close()

    messages: list[BaseMessage] = []
    for row in rows:
        role = row["role"]
        content = str(row["content"])
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))
    if len(messages) <= SUMMARY_RECENT_MESSAGE_COUNT + 2:
        return False

    return count_tokens_approximately(messages) > SUMMARY_TRIGGER_TOKENS