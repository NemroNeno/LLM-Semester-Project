from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from threading import Lock, Thread

from neo4j import GraphDatabase

from settings import (
    GRAPHITI_API_KEY,
    GRAPHITI_BASE_URL,
    GRAPHITI_EMBED_MODEL,
    GRAPHITI_LLM_MODEL,
    GRAPHITI_RERANK_MODEL,
    KG_DYNAMIC_GROUP_ID,
    KG_ENABLED,
    KG_STATIC_GROUP_ID,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)

try:
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType
except Exception:  # pragma: no cover - graceful fallback when optional deps are missing
    Graphiti = None  # type: ignore[assignment]
    OpenAIRerankerClient = None  # type: ignore[assignment]
    OpenAIEmbedder = None  # type: ignore[assignment]
    OpenAIEmbedderConfig = None  # type: ignore[assignment]
    OpenAIGenericClient = None  # type: ignore[assignment]
    LLMConfig = None  # type: ignore[assignment]
    EpisodeType = None  # type: ignore[assignment]


_GRAPHITI_CLIENT = None
_GRAPHITI_READY = False
_GRAPHITI_LOCK = Lock()
_ASYNC_LOCK = Lock()
_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_ASYNC_THREAD: Thread | None = None


class KnowledgeGraphUnavailable(RuntimeError):
    pass


def kg_enabled() -> bool:
    return bool(KG_ENABLED)


def _run_async(coro, *, timeout: float | None = 30.0):
    global _ASYNC_LOOP, _ASYNC_THREAD

    with _ASYNC_LOCK:
        if _ASYNC_LOOP is None or _ASYNC_THREAD is None or not _ASYNC_THREAD.is_alive():
            _ASYNC_LOOP = asyncio.new_event_loop()

            def _loop_runner() -> None:
                assert _ASYNC_LOOP is not None
                asyncio.set_event_loop(_ASYNC_LOOP)
                _ASYNC_LOOP.run_forever()

            _ASYNC_THREAD = Thread(target=_loop_runner, name="graphiti-async-loop", daemon=True)
            _ASYNC_THREAD.start()

    assert _ASYNC_LOOP is not None
    future = asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
    return future.result(timeout=timeout)


def _episode_type_text():
    if EpisodeType is None:
        raise KnowledgeGraphUnavailable("Graphiti EpisodeType is unavailable.")
    return EpisodeType.text


async def _build_client():
    if not kg_enabled():
        raise KnowledgeGraphUnavailable("Knowledge graph is disabled by configuration.")
    if (
        Graphiti is None
        or OpenAIGenericClient is None
        or LLMConfig is None
        or OpenAIEmbedder is None
        or OpenAIEmbedderConfig is None
        or OpenAIRerankerClient is None
        or EpisodeType is None
        or GRAPHITI_API_KEY is None
    ):
        raise KnowledgeGraphUnavailable(
            "Graphiti is unavailable. Install dependencies and set GRAPHITI_API_KEY/OPENROUTER_API_KEY."
        )

    llm_config = LLMConfig(
        api_key=GRAPHITI_API_KEY,
        model=GRAPHITI_LLM_MODEL,
        base_url=GRAPHITI_BASE_URL,
    )
    llm_client = OpenAIGenericClient(
        config=llm_config
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=GRAPHITI_API_KEY,
            embedding_model=GRAPHITI_EMBED_MODEL,
            embedding_dim=1536,
            base_url=GRAPHITI_BASE_URL,
        )
    )
    reranker = OpenAIRerankerClient(config=LLMConfig(api_key=GRAPHITI_API_KEY, model=GRAPHITI_RERANK_MODEL, base_url=GRAPHITI_BASE_URL))

    return Graphiti(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker,
    )


async def _ensure_client():
    global _GRAPHITI_CLIENT, _GRAPHITI_READY

    try:
        if _GRAPHITI_CLIENT is None:
            _GRAPHITI_CLIENT = await _build_client()

        if not _GRAPHITI_READY:
            await _GRAPHITI_CLIENT.build_indices_and_constraints()
            _GRAPHITI_READY = True
    except Exception as exc:
        raise KnowledgeGraphUnavailable(str(exc)) from exc

    return _GRAPHITI_CLIENT


def _get_client_sync():
    global _GRAPHITI_CLIENT, _GRAPHITI_READY

    if _GRAPHITI_CLIENT is not None and _GRAPHITI_READY:
        return _GRAPHITI_CLIENT

    with _GRAPHITI_LOCK:
        if _GRAPHITI_CLIENT is not None and _GRAPHITI_READY:
            return _GRAPHITI_CLIENT
        try:
            _GRAPHITI_CLIENT = _run_async(_build_client(), timeout=30.0)
            _run_async(_GRAPHITI_CLIENT.build_indices_and_constraints(), timeout=60.0)
            _GRAPHITI_READY = True
        except Exception as exc:
            raise KnowledgeGraphUnavailable(str(exc)) from exc
        return _GRAPHITI_CLIENT


def ingest_text_episode(
    *,
    name: str,
    body: str,
    source_description: str,
    group_id: str,
) -> str:
    if not body.strip() or not kg_enabled():
        return ""

    try:
        client = _get_client_sync()
        result = _run_async(
            client.add_episode(
                name=name,
                episode_body=body,
                source=_episode_type_text(),
                source_description=source_description,
                group_id=group_id,
                reference_time=datetime.now(timezone.utc),
            ),
            timeout=60.0,
        )
    except Exception:
        return ""

    episode = getattr(result, "episode", None)
    return str(getattr(episode, "uuid", "") or "")


def search_knowledge_graph(query: str, *, top_k: int) -> list[str]:
    if not query.strip() or not kg_enabled():
        return []

    try:
        client = _get_client_sync()
        results = _run_async(
            client.search(
                query=query,
                group_ids=[KG_STATIC_GROUP_ID, KG_DYNAMIC_GROUP_ID],
                num_results=top_k,
            ),
            timeout=12.0,
        )
    except Exception:
        return _fallback_neo4j_keyword_search(query, top_k)

    references: list[str] = []
    for index, edge in enumerate(results, start=1):
        fact = str(getattr(edge, "fact", "")).strip()
        if not fact:
            continue
        group_id = str(getattr(edge, "group_id", ""))
        source_desc = str(getattr(edge, "source_description", ""))
        source_label = source_desc or ("Static markdown" if group_id == KG_STATIC_GROUP_ID else "Uploaded document")
        references.append(f"KG Source {index}\nGroup: {group_id or 'unknown'}\nSource: {source_label}\nFact: {fact}")

    return references


def _fallback_neo4j_keyword_search(query: str, top_k: int) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2]
    if not tokens:
        return []

    cypher = """
    MATCH ()-[e:RELATES_TO]->()
    WHERE e.group_id IN $group_ids
      AND any(token IN $tokens WHERE toLower(coalesce(e.fact, '')) CONTAINS token)
    RETURN e.fact AS fact, e.group_id AS group_id, e.name AS edge_name
    LIMIT $limit
    """

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session(database="neo4j") as session:
                rows = session.run(
                    cypher,
                    group_ids=[KG_STATIC_GROUP_ID, KG_DYNAMIC_GROUP_ID],
                    tokens=tokens,
                    limit=max(1, top_k),
                ).data()
    except Exception:
        return []

    references: list[str] = []
    for index, row in enumerate(rows, start=1):
        fact = str(row.get("fact") or "").strip()
        if not fact:
            continue
        group_id = str(row.get("group_id") or "")
        edge_name = str(row.get("edge_name") or "")
        source_label = edge_name or ("Static markdown" if group_id == KG_STATIC_GROUP_ID else "Uploaded document")
        references.append(
            f"KG Source {index}\nGroup: {group_id or 'unknown'}\nSource: {source_label}\nFact: {fact}"
        )

    return references


def delete_episodes(episode_ids: list[str]) -> None:
    if not episode_ids or not kg_enabled():
        return

    try:
        client = _get_client_sync()
    except Exception:
        return

    for episode_id in episode_ids:
        if not episode_id:
            continue
        try:
            _run_async(client.remove_episode(episode_id), timeout=20.0)
        except Exception:
            continue


async def ingest_markdown_file_async(*, file_stem: str, content: str) -> str:
    try:
        client = await _ensure_client()
        result = await client.add_episode(
            name=f"sheet:{file_stem}",
            episode_body=content,
            source=_episode_type_text(),
            source_description=f"sheets_markdown/{file_stem}.md",
            group_id=KG_STATIC_GROUP_ID,
            reference_time=datetime.now(timezone.utc),
        )
        episode = getattr(result, "episode", None)
        return str(getattr(episode, "uuid", "") or "")
    except Exception:
        return ""
