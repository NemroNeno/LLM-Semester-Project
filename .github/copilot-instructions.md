# LLM Semester Project Instructions

## Scope

These instructions apply to the entire workspace.

## Project Context

This repository is for a LangGraph-based RAG chatbot using FastAPI, Ollama, ChromaDB, and local JSON banking Q&A data.

## Required Workflow

1. Before planning or implementing any task related to LangChain, LangGraph, Ollama integrations, embeddings, vector stores, prompt/state handling, or FastAPI integration with those components, fetch the relevant official documentation with the `mcp_langchain-doc_SearchDocsByLangChain` tool.
2. Use the documentation results to ground decisions before proposing architecture, implementation details, or API usage.
3. Do not rely on memory for LangChain or LangGraph APIs when the docs tool can verify the current interface or recommended pattern.

## Task Planning Rules

1. Use the Shrimp Task Manager to plan the complete project before substantial implementation begins.
2. The primary planning tool is `mcp_shrimp-task-m_plan_task`.
3. If task decomposition is needed, use `mcp_shrimp-task-m_split_tasks` to break the project into concrete implementation tasks with dependencies and verification criteria.
4. Keep the task list aligned with the actual repository state. Update task definitions when scope changes.

## Task Execution Rules

1. When executing a planned Shrimp task, use `mcp_shrimp-task-m_execute_task` for the active task before making implementation decisions.
2. Follow the execution guidance returned by the task tool instead of bypassing it.
3. After completing a task, use `mcp_shrimp-task-m_verify_task` to verify the task against its acceptance criteria.
4. Do not mark task work as complete until verification has been performed.

## Implementation Expectations

1. Prefer minimal, maintainable structure: a single-file app or minimal multi-file layout unless a clear split improves the project.
2. Keep retrieval, memory trimming, generation, ingestion, and FastAPI endpoints easy to trace.
3. Preserve local-first behavior: Ollama for inference and embeddings, ChromaDB for persistent local storage, and local JSON files as the source corpus.
4. When implementing LangGraph state or nodes, keep the code consistent with current official docs and this project's RAG workflow.

## Verification Expectations

1. Validate changed files for syntax or type issues when practical.
2. Prefer root-cause fixes over superficial patches.
3. Keep changes focused on the active task and avoid unrelated refactors.
