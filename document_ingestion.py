from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, UploadFile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pypdf import PdfReader

from db import (
    create_document_record,
    create_ingestion_job,
    delete_document_qa_pairs,
    get_document_record,
    get_ingestion_job,
    get_latest_ingestion_job,
    list_document_qa_pairs,
    list_document_records,
    save_document_qa_pairs,
    soft_delete_document,
    update_document_record,
    update_ingestion_job,
)
from rag import get_vector_store, message_to_text
from settings import (
    INGESTION_DEFAULT_QA_COUNT,
    INGESTION_MAX_CHUNK_CHARS,
    OPENROUTER_BASE_URL,
    OPENROUTER_INGEST_MODEL,
    UPLOAD_DIRECTORY,
)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv", ".xlsx", ".xls", ".xlsm"}


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    job_id: str


def ensure_upload_directory() -> Path:
    upload_directory = Path(UPLOAD_DIRECTORY)
    upload_directory.mkdir(parents=True, exist_ok=True)
    return upload_directory


def generate_document_id() -> str:
    return str(uuid.uuid4())


def generate_job_id() -> str:
    return str(uuid.uuid4())


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().strip()


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "uploaded_document"


def save_upload_file(upload_file: UploadFile, document_id: str) -> Path:
    upload_directory = ensure_upload_directory()
    extension = get_file_extension(upload_file.filename or "")
    stored_filename = f"{document_id}{extension}"
    stored_path = upload_directory / stored_filename

    with stored_path.open("wb") as handle:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    return stored_path


def extract_txt_text(file_path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="latin-1", errors="ignore")


def extract_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"[Page {page_number}]\n{text}")
    return "\n\n".join(pages)


def dataframe_to_text(dataframe: pd.DataFrame, *, sheet_name: str | None = None) -> str:
    normalized = dataframe.copy()
    normalized = normalized.fillna("")
    csv_text = normalized.to_csv(index=False)
    if sheet_name:
        return f"[Sheet: {sheet_name}]\n{csv_text}"
    return csv_text


def extract_csv_text(file_path: Path) -> str:
    dataframe = pd.read_csv(file_path)
    return dataframe_to_text(dataframe)


def extract_excel_text(file_path: Path) -> str:
    sheets = pd.read_excel(file_path, sheet_name=None)
    sheet_blocks: list[str] = []
    for sheet_name, dataframe in sheets.items():
        if dataframe.empty:
            continue
        sheet_blocks.append(dataframe_to_text(dataframe, sheet_name=str(sheet_name)))
    return "\n\n".join(sheet_blocks)


def extract_text(file_path: Path) -> str:
    extension = file_path.suffix.lower()
    if extension == ".txt":
        return extract_txt_text(file_path)
    if extension == ".pdf":
        return extract_pdf_text(file_path)
    if extension == ".csv":
        return extract_csv_text(file_path)
    if extension in {".xlsx", ".xls", ".xlsm"}:
        return extract_excel_text(file_path)

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")


def split_text_for_qa(text: str, max_chars: int = INGESTION_MAX_CHUNK_CHARS) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return []

    paragraphs = [block.strip() for block in re.split(r"\n{2,}", cleaned) if block.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_size = 0

    def flush_current() -> None:
        nonlocal current_parts, current_size
        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_size = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush_current()
            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        if current_size + len(paragraph) + 2 > max_chars:
            flush_current()

        current_parts.append(paragraph)
        current_size += len(paragraph) + 2

    flush_current()
    return chunks


def get_ingest_chat_model() -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY before running document ingestion.")

    headers: dict[str, str] = {}
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
    app_title = os.getenv("OPENROUTER_APP_TITLE")
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if app_title:
        headers["X-Title"] = app_title

    return ChatOpenAI(
        model=OPENROUTER_INGEST_MODEL,
        api_key=SecretStr(api_key),
        base_url=OPENROUTER_BASE_URL,
        temperature=0.0,
        default_headers=headers or None,
    )


def parse_qa_payload(text: str) -> list[dict[str, str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    pairs = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(pairs, list):
        return []

    cleaned_pairs: list[dict[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()
        if question and answer:
            cleaned_pairs.append({"question": question, "answer": answer})
    return cleaned_pairs


def generate_qa_pairs_for_chunk(
    chat_model: ChatOpenAI,
    *,
    filename: str,
    chunk_text: str,
    target_count: int,
) -> list[dict[str, str]]:
    prompt = [
        SystemMessage(
            content=(
                "You create concise question-answer pairs from source documents. "
                "Return JSON only, using this exact schema: {\"pairs\": [{\"question\": string, \"answer\": string}]}. "
                "Use only facts supported by the provided text. "
                "Do not invent details, and do not include markdown or commentary."
            )
        ),
        HumanMessage(
            content=(
                f"Source file: {filename}\n"
                f"Create {target_count} question-answer pairs from the following extracted text.\n\n"
                f"{chunk_text}"
            )
        ),
    ]

    response = chat_model.invoke(prompt)
    response_text = message_to_text(response)
    pairs = parse_qa_payload(response_text)
    if pairs:
        return pairs[:target_count]

    fallback_question = f"What is the main information in {filename}?"
    fallback_answer = chunk_text[:800].strip()
    if not fallback_answer:
        fallback_answer = "No extractable text was found in this document chunk."
    return [{"question": fallback_question, "answer": fallback_answer}]


def build_vector_documents(
    *,
    document_id: str,
    original_filename: str,
    extracted_text: str,
    qa_pairs: list[dict[str, str]],
) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    vector_ids: list[str] = []

    for pair_index, qa_pair in enumerate(qa_pairs):
        vector_id = f"doc:{document_id}:qa:{pair_index}"
        documents.append(
            Document(
                page_content=f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}",
                metadata={
                    "source_document_id": document_id,
                    "source_filename": original_filename,
                    "qa_pair_index": pair_index,
                    "source_type": "uploaded_document",
                    "extracted_text": extracted_text[:4000],
                },
            )
        )
        vector_ids.append(vector_id)

    return documents, vector_ids


def process_document_ingestion(document_id: str, job_id: str) -> None:
    document_record = get_document_record(document_id)
    if not document_record:
        update_ingestion_job(
            job_id,
            status="failed",
            stage="missing_document",
            progress=100,
            error_message="Document record was not found.",
            finished=True,
        )
        return

    try:
        update_ingestion_job(job_id, status="running", stage="saving", progress=5, message="Document saved")
        file_path = Path(document_record["stored_path"])

        update_ingestion_job(job_id, stage="extracting", progress=20, message="Extracting text")
        extracted_text = extract_text(file_path)
        extracted_text = extracted_text.strip()

        if not extracted_text:
            raise RuntimeError("No extractable text was found in the uploaded document.")

        update_document_record(
            document_id,
            extracted_text=extracted_text,
            extracted_text_length=len(extracted_text),
            status="extracting",
            progress=20,
        )

        chat_model = get_ingest_chat_model()
        chunks = split_text_for_qa(extracted_text)
        if not chunks:
            chunks = [extracted_text]

        target_pairs_per_chunk = max(1, INGESTION_DEFAULT_QA_COUNT)
        qa_pairs: list[dict[str, str]] = []
        total_chunks = len(chunks)

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunk_progress = 20 + int(60 * (chunk_index / max(1, total_chunks)))
            update_ingestion_job(
                job_id,
                stage="generating",
                progress=min(chunk_progress, 80),
                message=f"Generating QA pairs for chunk {chunk_index} of {total_chunks}",
            )
            chunk_pairs = generate_qa_pairs_for_chunk(
                chat_model,
                filename=document_record["original_filename"],
                chunk_text=chunk_text,
                target_count=target_pairs_per_chunk,
            )
            qa_pairs.extend(chunk_pairs)

        if not qa_pairs:
            raise RuntimeError("No QA pairs could be generated for the uploaded document.")

        update_ingestion_job(job_id, stage="saving_qa", progress=85, message="Saving QA pairs")
        vector_store = get_vector_store()
        vector_documents, vector_ids = build_vector_documents(
            document_id=document_id,
            original_filename=document_record["original_filename"],
            extracted_text=extracted_text,
            qa_pairs=qa_pairs,
        )

        vector_store.add_documents(documents=vector_documents, ids=vector_ids)
        save_document_qa_pairs(
            document_id,
            [{**pair, "vector_id": vector_id} for pair, vector_id in zip(qa_pairs, vector_ids, strict=False)],
        )

        update_document_record(
            document_id,
            status="completed",
            progress=100,
            qa_count=len(qa_pairs),
            vector_ids=json.dumps(vector_ids),
        )
        update_ingestion_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Document ingestion completed successfully.",
            finished=True,
        )
    except Exception as exc:
        update_document_record(document_id, status="failed", progress=100, error_message=str(exc))
        update_ingestion_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            error_message=str(exc),
            finished=True,
        )


def ingest_uploaded_document(upload_file: UploadFile) -> IngestionResult:
    filename = upload_file.filename or "uploaded_document"
    extension = get_file_extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")

    document_id = generate_document_id()
    job_id = generate_job_id()
    stored_path = save_upload_file(upload_file, document_id)
    mime_type = upload_file.content_type or mimetypes.guess_type(filename)[0]

    create_document_record(
        document_id=document_id,
        original_filename=sanitize_filename(filename),
        stored_path=str(stored_path),
        file_extension=extension,
        mime_type=mime_type,
        source_type="uploaded_file",
    )
    create_ingestion_job(document_id, job_id)

    thread = threading.Thread(target=process_document_ingestion, args=(document_id, job_id), daemon=True)
    thread.start()

    return IngestionResult(document_id=document_id, job_id=job_id)


def delete_document(document_id: str) -> None:
    document_record = get_document_record(document_id)
    if not document_record:
        raise HTTPException(status_code=404, detail="Document not found.")

    vector_ids = []
    raw_vector_ids = document_record.get("vector_ids")
    if raw_vector_ids:
        try:
            parsed = json.loads(raw_vector_ids)
            if isinstance(parsed, list):
                vector_ids = [str(item) for item in parsed]
        except json.JSONDecodeError:
            vector_ids = []

    if not vector_ids:
        vector_ids = [row["vector_id"] for row in list_document_qa_pairs(document_id)]

    vector_store = get_vector_store()
    if vector_ids:
        try:
            vector_store.delete(ids=vector_ids)
        except Exception:
            pass

    delete_document_qa_pairs(document_id)

    stored_path = document_record.get("stored_path")
    if stored_path:
        try:
            Path(stored_path).unlink(missing_ok=True)
        except Exception:
            pass

    soft_delete_document(document_id)


def list_documents() -> list[dict[str, Any]]:
    return list_document_records(include_deleted=False)


def get_document_status(document_id: str) -> dict[str, Any]:
    document_record = get_document_record(document_id)
    if not document_record:
        raise HTTPException(status_code=404, detail="Document not found.")

    latest_job = get_latest_ingestion_job(document_id)

    return {
        "document": document_record,
        "job": latest_job,
    }