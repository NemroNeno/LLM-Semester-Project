import json
import os
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

QA_DIRECTORY = Path(os.getenv("QA_DIRECTORY", "sheets_qa"))
PERSIST_DIRECTORY = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "banking_qa")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))


def build_page_content(question: str, answer: str) -> str:
    answer_text = answer if answer else "No answer provided in source data."
    return f"Question: {question}\nAnswer: {answer_text}"


def load_qa_documents(qa_directory: Path) -> tuple[list[Document], list[str], int]:
    if not qa_directory.exists():
        raise FileNotFoundError(f"QA directory does not exist: {qa_directory}")

    json_files = sorted(qa_directory.glob("qa_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No qa_*.json files found in: {qa_directory}")

    documents: list[Document] = []
    document_ids: list[str] = []

    for json_file in json_files:
        rows = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {json_file}")

        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {json_file} index {index}")

            missing_fields = [key for key in ("question", "answer", "sheet") if key not in row]
            if missing_fields:
                missing_text = ", ".join(missing_fields)
                raise ValueError(f"Missing field(s) {missing_text} in {json_file} index {index}")

            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            sheet = str(row["sheet"]).strip()

            if not question:
                raise ValueError(f"Empty question in {json_file} index {index}")
            if not sheet:
                raise ValueError(f"Empty sheet name in {json_file} index {index}")

            metadata = {
                "sheet": sheet,
                "source_file": json_file.name,
                "record_index": index,
                "question": question,
                "has_answer": bool(answer),
            }

            documents.append(
                Document(
                    page_content=build_page_content(question=question, answer=answer),
                    metadata=metadata,
                )
            )
            document_ids.append(f"{json_file.stem}:{index}")

    return documents, document_ids, len(json_files)


def reset_collection_if_present(client: chromadb.PersistentClient, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name)
    except NotFoundError:
        return


def ingest_documents() -> None:
    documents, document_ids, file_count = load_qa_documents(QA_DIRECTORY)

    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        validate_model_on_init=True,
    )

    client = chromadb.PersistentClient(path=str(PERSIST_DIRECTORY))
    reset_collection_if_present(client, COLLECTION_NAME)

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        vector_store.add_documents(
            documents=documents[start:end],
            ids=document_ids[start:end],
        )

    print(f"Processed files: {file_count}")
    print(f"Processed records: {len(documents)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Persist directory: {PERSIST_DIRECTORY}")
    print("Collection refresh strategy: reset existing collection before ingest")


if __name__ == "__main__":
    ingest_documents()