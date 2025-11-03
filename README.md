
# EXAID

EXAID is an experimental Python project for capturing short, live traces from multiple agents, buffering those traces, and producing concise summaries using an LLM. It is designed as a minimal prototype for medical multi-agent reasoning workflows, where specialized agents (e.g., InfectiousDiseaseAgent, HematologyAgent, OncologyAgent) collaborate on clinical cases, and their reasoning traces are captured and condensed into structured summaries optimized for physician understanding.

## Quick Start

1. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\Activate.ps1
   # On Unix/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the LLM client** in `llm.py` with your API credentials and endpoint.

4. **Run the demo:**

   ```bash
   python demo.py
   ```

5. **Use the EXAID class in your code:**

   ```python
   import asyncio
   from exaid import EXAID

   async def main():
       exaid = EXAID()
       # Add traces for any agent (agent_id, text)
       summary = await exaid.received_trace("agent_1", "Some trace text")
       if summary:
           print("Updated summary:", summary)
           # Get summary as JSON
           json_summary = exaid.get_summary_json(summary)
           print(json_summary)
   
   asyncio.run(main())
   ```


## High-level Design

The system is organized around a few small modules:

- `exaid.py` — EXAID orchestrator class
  - Purpose: Collects traces from agents, buffers them, and produces summaries using an LLM. Maintains a list of all summaries.
  - Key methods:
    - `received_trace(agent_id, text)` — Call this to add a trace for an agent. If a summary is triggered, it returns the new `AgentSummary` object.
    - `latest_summary()` — Returns the most recent summary or "No summaries yet."
    - `get_all_summaries()` — Returns all summaries as a list of `AgentSummary` objects.
    - `get_summary_json(summary=None)` — Returns a summary as a JSON string (defaults to latest summary).

- `agents/summarizer.py` — Summarization wrapper
  - Purpose: Contains the `summarize` function, which wraps calls to the LLM (via `llm.py`) and produces structured `AgentSummary` objects from input text.
  - Features:
    - Uses structured output with Pydantic models for consistent summaries
    - Enforces character limits (action: 100, reasoning: 200, findings: 150, next_steps: 100)
    - Optimized for medical/clinical reasoning with physician-focused prompts
    - Returns `AgentSummary` objects with fields: `agents`, `action`, `reasoning`, `findings`, `next_steps`

- `agents/base_agent.py` — Base agent interface
  - Purpose: Abstract base class (`BaseAgent`) defining the interface for agents that can be integrated with EXAID.
  - Key method: `act(input: str) -> str` — Abstract method that agents must implement.

- `buffer.py` — TraceBuffer
  - Purpose: Implements `TraceBuffer`, a buffer that accumulates traces per agent. Uses an LLM-based prompt to decide when to trigger summarization (event-driven, not just a static threshold).
  - Features:
    - LLM-powered trigger logic that evaluates trace content
    - Decides summarization based on completed thoughts, topic changes, or accumulated context
    - Tags traces with agent IDs for multi-agent tracking

- `llm.py` — LLM client configuration
  - Purpose: Holds the LLM client instance used for summarization and trigger decisions. Update this file with your model provider and credentials.
  - Currently configured for OpenAI-compatible API (using LangChain's `ChatOpenAI`).

- `demo.py` — Example usage
  - Purpose: Demonstrates how to use the EXAID system with multiple agents and traces in a medical scenario.
  - Shows a complete workflow: patient case processing through multiple specialized agents (InfectiousDiseaseAgent, HematologyAgent, OncologyAgent, etc.) with formatted summary output.

- `requirements.txt` — Python dependencies
  - Purpose: Lists the project's external Python dependencies (LangChain, Pydantic, etc.).

## Features

- **LLM-powered event-driven summarization:** The buffer uses an LLM to intelligently decide when to trigger summarization based on trace content, not just a static threshold. Summarization triggers when thoughts complete, topics change, or sufficient context accumulates.
- **Multi-agent support:** Traces are tagged by agent ID and summarized in context, allowing multiple specialized agents to contribute to a single reasoning workflow.
- **Structured summaries:** Summaries are generated as structured `AgentSummary` objects with fields optimized for medical reasoning:
  - `agents`: List of agent IDs involved
  - `action`: Brief action statement (max 100 chars)
  - `reasoning`: Concise reasoning explanation (max 200 chars)
  - `findings`: Key clinical findings or recommendations (max 150 chars, optional)
  - `next_steps`: Suggested next actions (max 100 chars, optional)
- **Character limit enforcement:** Automatic truncation ensures summaries remain concise and physician-friendly.
- **Medical/clinical focus:** Prompts and summaries are optimized for physician understanding of multi-agent clinical reasoning.
- **Simple async API:** Add traces and get summaries with a single async method call.
- **JSON export:** Export summaries as JSON for integration with other systems.

## Development Notes and Suggestions

- The project is a prototype. Expect to iterate on the summarization prompt and LLM configuration.
- Update `llm.py` to point at your LLM provider and set credentials securely. The current configuration uses an OpenAI-compatible API endpoint.
- The system uses async/await patterns throughout, so ensure you're running within an async context when calling methods.
- See `doc/feedback.md` for additional design suggestions:
  - Event subscription/pub-sub system for summary events (currently not implemented)
  - Further clarification on trace semantics and structure

## Project Structure

```
ExAID/
├── exaid.py                 # Main orchestrator class
├── buffer.py                # TraceBuffer with LLM-based trigger logic
├── llm.py                   # LLM client configuration
├── demo.py                  # Medical case demonstration
├── requirements.txt         # Python dependencies
├── agents/
│   ├── base_agent.py       # Abstract base class for agents
│   └── summarizer.py       # Summarization logic with structured output
└── doc/
    └── feedback.md         # Design feedback and suggestions
```

## Files Summary

- `exaid.py`: Orchestrator class that collects traces, buffers them, and records summaries. Provides methods for trace processing, summary retrieval, and JSON export.
- `agents/summarizer.py`: Summarization logic with structured output. Defines `AgentSummary` model and `summarize` function.
- `agents/base_agent.py`: Abstract base class (`BaseAgent`) defining the interface for agents that can integrate with EXAID.
- `buffer.py`: `TraceBuffer` implementation with LLM-based trigger logic for event-driven summarization.
- `llm.py`: LLM client configuration using LangChain's `ChatOpenAI` (configured for OpenAI-compatible endpoints).
- `demo.py`: Example usage demonstrating a complete medical case workflow with multiple specialized agents.
- `requirements.txt`: Project dependencies (LangChain, Pydantic, etc.).
- `doc/feedback.md`: Design feedback and suggestions for future improvements.

## License

MIT