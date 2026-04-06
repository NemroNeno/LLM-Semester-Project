from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
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

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            file_extension TEXT NOT NULL,
            mime_type TEXT,
            source_type TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            extracted_text TEXT,
            extracted_text_length INTEGER NOT NULL DEFAULT 0,
            qa_count INTEGER NOT NULL DEFAULT 0,
            vector_ids TEXT,
            latest_job_id TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deleted_at TIMESTAMP
        )
        """
    )
    document_columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "progress" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN progress INTEGER NOT NULL DEFAULT 0")
    if "extracted_text" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN extracted_text TEXT")
    if "extracted_text_length" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN extracted_text_length INTEGER NOT NULL DEFAULT 0")
    if "qa_count" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN qa_count INTEGER NOT NULL DEFAULT 0")
    if "vector_ids" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN vector_ids TEXT")
    if "latest_job_id" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN latest_job_id TEXT")
    if "error_message" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN error_message TEXT")
    if "updated_at" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    if "deleted_at" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN deleted_at TIMESTAMP")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            message TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
        """
    )
    job_columns = {row["name"] for row in conn.execute("PRAGMA table_info(ingestion_jobs)").fetchall()}
    if "status" not in job_columns:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN status TEXT NOT NULL DEFAULT 'queued'")
    if "stage" not in job_columns:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN stage TEXT NOT NULL DEFAULT 'queued'")
    if "progress" not in job_columns:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN progress INTEGER NOT NULL DEFAULT 0")
    if "message" not in job_columns:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN message TEXT")
    if "error_message" not in job_columns:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN error_message TEXT")
    if "updated_at" not in job_columns:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    if "finished_at" not in job_columns:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN finished_at TIMESTAMP")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            pair_index INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            vector_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
        """
    )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_deleted_at ON documents(deleted_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_document_id ON ingestion_jobs(document_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_document_id ON document_qa_pairs(document_id)")
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_document_record(
    *,
    document_id: str,
    original_filename: str,
    stored_path: str,
    file_extension: str,
    mime_type: str | None,
    source_type: str,
) -> None:
    conn = get_db()
    conn.execute(
        """
        INSERT INTO documents (
            id, original_filename, stored_path, file_extension, mime_type, source_type,
            status, progress, extracted_text_length, qa_count, latest_job_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            document_id,
            original_filename,
            stored_path,
            file_extension,
            mime_type,
            source_type,
            "queued",
            0,
            0,
            0,
            None,
        ),
    )
    conn.commit()
    conn.close()


def update_document_record(
    document_id: str,
    **fields: Any,
) -> None:
    if not fields:
        return

    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values())
    values.append(document_id)
    conn = get_db()
    conn.execute(
        f"UPDATE documents SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        values,
    )
    conn.commit()
    conn.close()


def get_document_record(document_id: str) -> dict[str, Any] | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_document_records(*, include_deleted: bool = False) -> list[dict[str, Any]]:
    conn = get_db()
    query = "SELECT * FROM documents"
    params: tuple[Any, ...] = ()
    if not include_deleted:
        query += " WHERE deleted_at IS NULL"
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_ingestion_job(document_id: str, job_id: str) -> None:
    conn = get_db()
    conn.execute(
        """
        INSERT INTO ingestion_jobs (id, document_id, status, stage, progress, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (job_id, document_id, "queued", "queued", 0),
    )
    conn.execute("UPDATE documents SET latest_job_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (job_id, document_id))
    conn.commit()
    conn.close()


def update_ingestion_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    error_message: str | None = None,
    finished: bool = False,
) -> None:
    updates: list[str] = []
    values: list[Any] = []
    if status is not None:
        updates.append("status = ?")
        values.append(status)
    if stage is not None:
        updates.append("stage = ?")
        values.append(stage)
    if progress is not None:
        updates.append("progress = ?")
        values.append(progress)
    if message is not None:
        updates.append("message = ?")
        values.append(message)
    if error_message is not None:
        updates.append("error_message = ?")
        values.append(error_message)
    if finished:
        updates.append("finished_at = CURRENT_TIMESTAMP")

    if not updates:
        return

    updates.append("updated_at = CURRENT_TIMESTAMP")
    conn = get_db()
    conn.execute(
        f"UPDATE ingestion_jobs SET {', '.join(updates)} WHERE id = ?",
        [*values, job_id],
    )
    conn.commit()
    conn.close()


def get_ingestion_job(job_id: str) -> dict[str, Any] | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_ingestion_job(document_id: str) -> dict[str, Any] | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM ingestion_jobs WHERE document_id = ? ORDER BY created_at DESC LIMIT 1",
        (document_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_document_qa_pairs(document_id: str, qa_pairs: list[dict[str, str]]) -> list[str]:
    conn = get_db()
    vector_ids: list[str] = []
    for index, qa_pair in enumerate(qa_pairs):
        vector_id = qa_pair.get("vector_id") or f"doc:{document_id}:qa:{index}"
        conn.execute(
            """
            INSERT INTO document_qa_pairs (document_id, pair_index, question, answer, vector_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                index,
                qa_pair["question"],
                qa_pair["answer"],
                vector_id,
            ),
        )
        vector_ids.append(vector_id)

    conn.commit()
    conn.close()
    return vector_ids


def list_document_qa_pairs(document_id: str) -> list[dict[str, Any]]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM document_qa_pairs WHERE document_id = ? ORDER BY pair_index ASC",
        (document_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_document_qa_pairs(document_id: str) -> list[str]:
    existing_pairs = list_document_qa_pairs(document_id)
    vector_ids = [str(row["vector_id"]) for row in existing_pairs]
    conn = get_db()
    conn.execute("DELETE FROM document_qa_pairs WHERE document_id = ?", (document_id,))
    conn.commit()
    conn.close()
    return vector_ids


def soft_delete_document(document_id: str, *, error_message: str | None = None) -> None:
    update_document_record(
        document_id,
        status="deleted",
        progress=100,
        error_message=error_message,
        deleted_at=utc_now_iso(),
    )