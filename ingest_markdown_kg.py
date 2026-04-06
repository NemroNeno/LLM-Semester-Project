from __future__ import annotations

import asyncio
import re
from pathlib import Path

from datetime import datetime, timezone

from tqdm import tqdm

from knowledge_graph import _ensure_client, _episode_type_text, kg_enabled


MARKDOWN_DIR = Path("sheets_markdown")
MAX_CHUNK_CHARS = 4000


def split_markdown(content: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", content).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    current = []
    current_size = 0

    for block in [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]:
        if len(block) > max_chars:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_size = 0
            for start in range(0, len(block), max_chars):
                piece = block[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        if current_size + len(block) + 2 > max_chars and current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_size = 0

        current.append(block)
        current_size += len(block) + 2

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks


async def main() -> None:
    if not kg_enabled():
        print("KG ingestion skipped: KG_ENABLED is false.")
        return

    if not MARKDOWN_DIR.exists():
        print(f"Directory not found: {MARKDOWN_DIR}")
        return

    markdown_files = sorted(MARKDOWN_DIR.glob("*.md"))
    if not markdown_files:
        print("No markdown files found in sheets_markdown.")
        return

    print(f"Found {len(markdown_files)} markdown files. Ingesting into knowledge graph...")
    success_count = 0

    client = await _ensure_client()
    episode_type = _episode_type_text()

    for markdown_file in tqdm(markdown_files, desc="Ingesting markdown files", unit="file"):
        content = markdown_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"Skipping empty file: {markdown_file.name}")
            continue

        chunks = split_markdown(content)
        if not chunks:
            print(f"Skipping empty file after normalization: {markdown_file.name}")
            continue

        chunk_ids: list[str] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            try:
                result = await client.add_episode(
                    name=f"sheet:{markdown_file.stem}:{chunk_index}",
                    episode_body=chunk,
                    source=episode_type,
                    source_description=f"sheets_markdown/{markdown_file.name}",
                    group_id="nust-markdown",
                    reference_time=datetime.now(timezone.utc),
                )
                episode = getattr(result, "episode", None)
                episode_id = str(getattr(episode, "uuid", "") or "")
                if episode_id:
                    chunk_ids.append(episode_id)
            except Exception as exc:
                print(f"Failed {markdown_file.name} chunk {chunk_index}: {exc}")

        if chunk_ids:
            success_count += 1
            print(f"Ingested {markdown_file.name} -> {len(chunk_ids)} episode(s)")
        else:
            print(f"Skipped {markdown_file.name}: no KG episodes were created.")

    print(f"Completed markdown KG ingestion. {success_count}/{len(markdown_files)} files ingested.")


if __name__ == "__main__":
    asyncio.run(main())
