# EXAID

EXAID is a small experimental Python project for capturing short "live" traces from multiple agents, buffering those traces, and producing concise summaries that are stored in an in-memory summary graph. It is intended as a minimal prototype for workflows such as live note-taking, agent introspection, or multi-agent reasoning where intermediate traces are captured and condensed over time.

This README documents the project's purpose and provides a short description of each file so contributors and users can quickly understand the code layout.

Quick start

1. Create and activate a virtual environment (recommended):

    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1

2. Install dependencies:

    pip install -r requirements.txt

3. Run an example (project is a prototype; scripts may be minimal):

    python orchestrator_agent.py

High-level design

The system is organized around a few small modules:

- `exaid.py` — EXAID orchestrator class
   - Purpose: A lightweight orchestrator object that holds a `SummaryState` (an in-memory graph of traces and summaries), manages registered agents, and collects traces via a `TraceBuffer`. When the buffer reaches a threshold the EXAID instance requests a summary and writes the resulting summary back into the `SummaryState`.
   - Key symbols: `EXAID.addAgent`, `EXAID.addTrace`, `EXAID._on_buffer_full`.

- `orchestrator_agent.py` — example driver / orchestrator
   - Purpose: Demonstrates how to wire multiple pieces together. Intended to run the system in a simple flow: create agents/orchestrator, send traces, and exercise the summarization pipeline. This file can be used as the main entrypoint for local tests and demos.

- `summarizer_agent.py` — summarization wrapper
   - Purpose: Contains the `summarize` function which wraps calls to the project's LLM integration (via `llm.py`) and produces short condensed summaries from input text.
   - Note: The current `summarize` implementation is synchronous in the repository copy and directly calls `llm.invoke` with a simple prompt.

- `buffer.py` — simple TraceBuffer
   - Purpose: Implements `TraceBuffer`, a tiny buffer that accumulates short traces (tuples of agent id and text). When the buffer reaches `chunk_threshold` it calls a callback (provided by `EXAID`) with the combined chunks and resets itself.

- `llm.py` — language-model client configuration
   - Purpose: Holds an LLM client instance used by `summarizer_agent.py`. The current code initializes a `langchain_openai.OpenAI` client pointing at a local base_url and a placeholder API key. Replace or configure this with your model provider and credentials when running for real.

- `summary_state.py` — in-memory summary graph/state
   - Purpose: A tiny in-memory structure which keeps per-agent traces, summaries, and feedback. It provides helper methods to add agents and append new trace/summary/feedback nodes. This file is intentionally simple and meant to be replaced by a more robust storage layer in production.

- `tools/normalizer_tool.py` — placeholder normalizer
   - Purpose: Intended to contain text-normalization helpers (e.g., stripping noisy tokens, normalizing whitespace, or simple cleaning). The file is currently empty and acts as a placeholder for future tooling.

- `requirements.txt` — Python dependencies
   - Purpose: Lists the project's external Python dependencies. Currently contains minimal langchain packages; you may need to pin additional dependencies or update versions depending on your environment.

Development notes and suggestions

- The project is a prototype: some modules contain minimal or placeholder code (for example, `tools/normalizer_tool.py` is empty). Expect to iterate on the summarization prompt and the LLM client configuration.

- If you plan to run this end-to-end, update `llm.py` to point at a reachable LLM provider and set credentials via environment variables or a secrets manager. Avoid committing real API keys into the repository.

- Consider adding unit tests for `buffer.py` and `summary_state.py` to lock in expected behaviour for buffering and state updates.

Files summary (one-line each)

- `exaid.py`: Orchestrator class that collects traces, buffers them, and records summaries into `SummaryState`.
- `orchestrator_agent.py`: Example runner that wires the system together for demos or local testing.
- `summarizer_agent.py`: Small wrapper that constructs summarization prompts and calls the LLM client.
- `buffer.py`: `TraceBuffer` implementation; triggers a callback when the threshold is reached.
- `llm.py`: LLM client configuration for local or remote model endpoints.
- `summary_state.py`: Lightweight in-memory graph to store per-agent traces, summaries, and feedback.
- `tools/normalizer_tool.py`: Placeholder for normalization utilities (currently empty).
- `requirements.txt`: Pin or list runtime dependencies for the project.

License: MIT
