# Deep Learning AI - LangGraph Essay Writer

A multi-agent LangGraph graph built as part of the DeepLearning.AI course. The agent autonomously plans, researches, writes, and iteratively revises a 5-paragraph essay using Tavily search and OpenAI.

## Project Structure

```
essay_writer/
├── essay_writer.py             # Main graph: nodes, edges, and entry point
├── agent_state.py              # AgentState TypedDict shared across all nodes
├── queries_model.py            # Pydantic model for structured search query output
├── pyproject.toml              # Project dependencies
├── .env                        # API keys (not committed)
└── README.md
```

## How It Works

The graph implements a plan → research → write → reflect → revise loop:

```
START → planner → research_plan → generate → (max revisions reached?) → END
                                      ↑               ↓ no
                                research_critique ← reflect
```

### Nodes

| Node | Responsibility |
|------|---------------|
| `planner` | Produces a high-level essay outline from the user's topic |
| `research_plan` | Generates up to 3 search queries and fetches Tavily results to inform writing |
| `generate` | Writes or rewrites the essay draft using the plan and accumulated research |
| `reflect` | Critiques the current draft (length, depth, style) |
| `research_critique` | Generates up to 3 search queries based on the critique and fetches more Tavily results |

### Revision Loop

After each draft, `should_continue` checks `revision_number` against `max_revisions`:

- If `revision_number > max_revisions` → route to `END`
- Otherwise → route to `reflect` for another revision cycle

### Structured Output

`research_plan` and `research_critique` use `model.with_structured_output(Queries)` to force the model to return a typed list of search query strings, then fan out to Tavily for each:

```python
class Queries(BaseModel):
    queries: list[str]
```

### AgentState

All nodes read from and write to a shared `AgentState` TypedDict:

```python
class AgentState(TypedDict):
    task: str            # The user's essay topic
    plan: str            # Outline from the planner
    draft: str           # Current essay draft
    critique: str        # Latest critique from reflect node
    content: list[str]   # Accumulated Tavily search results
    revision_number: int # Current revision count
    max_revisions: int   # Maximum allowed revisions
```

### Persistence (Checkpointing)

The graph uses an in-memory `SqliteSaver` checkpointer. Each run is identified by a `thread_id`:

```python
thread = {"configurable": {"thread_id": "1"}}
```

## Setup

### Prerequisites

- Python 3.12 (see note below)
- OpenAI API key
- Tavily API key
- LangSmith API key (optional, for tracing)

### Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install langgraph langchain-openai langchain-core tavily-python python-dotenv
```

### Configure environment

Create a `.env` file:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=lsv2_...        # optional
LANGSMITH_TRACING_V2=true         # optional
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=deep-learning-ai.lessons
```

## Usage

```bash
cd deep_learning.ai/essay_writer
uv run essay_writer.py
```

The default task is:

> *"what is the difference between langchain and langsmith"*

with `max_revisions=2`. Edit the `main()` call in `essay_writer.py` to change the topic or revision limit.

## Notes

- **Python 3.14 incompatibility**: `agent_state.py` and `queries_model.py` use `typing.List`, which was removed in Python 3.14. Use Python 3.12 or replace `List[str]` with `list[str]` and remove the `List` import.
- **`uv run` and `sys.path`**: Running `uv run essay_writer.py` inside a project with a `pyproject.toml` may not add the current directory to `sys.path`. If you get an `ImportError` for `agent_state`, use `PYTHONPATH=. uv run essay_writer.py` or `uv run --no-project essay_writer.py`.
- The model is `gpt-3.5-turbo` to minimise inference cost. Swap to `gpt-4o-mini` or higher for better quality.
- Tavily search returns up to 2 results per query. Content from all queries across all revision cycles is accumulated in `state['content']`.
- The in-memory SQLite store does not persist between process restarts.
