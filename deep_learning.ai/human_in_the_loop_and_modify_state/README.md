# Deep Learning AI - LangGraph Agent with Persistence & Streaming

A LangGraph-based ReAct agent built as part of the DeepLearning.AI course. The agent uses OpenAI's GPT models with Tavily web search, SQLite-backed persistence (checkpointing), and streaming support.

## Project Structure

```
persistence_streaming/
├── main.py          # Entry point, agent usage examples
├── my_agent.py      # Agent class with LangGraph graph definition
├── agent_state.py   # AgentState TypedDict definition
├── .env             # API keys (not committed)
└── README.md
```

## How It Works

The agent implements a ReAct (Reason + Act) loop using LangGraph:

1. **`llm` node** — Calls the OpenAI model with the current message history
2. **`action` node** — Executes any tool calls (Tavily search) returned by the model
3. The graph loops between `llm` and `action` until no more tool calls are made, then exits

```
START → llm → (tool calls?) → action → llm → ... → END
```

### Persistence (Checkpointing)

The agent uses `SqliteSaver` to persist conversation state across invocations. Each conversation is identified by a `thread_id`, passed via the `config` argument:

```python
thread = {"configurable": {"thread_id": "123"}}
result = abot.invoke({"messages": messages}, config=thread)
```

- The `configurable` key is the standard LangChain/LangGraph envelope for runtime config values.
- `thread_id` identifies the conversation thread — the checkpointer saves and loads state using this key.
- Using the same `thread_id` across multiple `.invoke()` calls gives the agent memory of prior messages.
- `SqliteSaver.from_conn_string(":memory:")` is a context manager — it must be used with `with` to get the actual saver instance.

```python
with SqliteSaver.from_conn_string(":memory:") as memory:
    abot = Agent(model, [tool], checkpointer=memory, system=prompt).graph
```

### Streaming

Instead of waiting for the full response, `.stream()` yields one chunk per node execution:

```python
for chunk in abot.stream({"messages": messages}, config=thread):
    print(chunk)
```

Each `chunk` is a dict shaped as `{ "node_name": { "messages": [...] } }`:

- `"llm"` node chunks contain an `AIMessage` (the model's response or tool call decision)
- `"action"` node chunks contain a `ToolMessage` (the search result)

For a tool-using query the sequence is:
```
chunk 1 → {"llm": ...}      # AI decides to call a tool
chunk 2 → {"action": ...}   # Tool runs, returns results
chunk 3 → {"llm": ...}      # AI reads results, gives final answer
```

To print only the AI's final text from each chunk:
```python
for chunk in abot.stream({"messages": messages}, config=thread):
    if "llm" in chunk:
        print(chunk["llm"]["messages"][-1].content)
```

## Setup

### Prerequisites

- Python 3.14+
- OpenAI API key (Tier 1+ for `gpt-4o-mini`)
- Tavily API key
- LangSmith API key (optional, for tracing)

### Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install langgraph langchain-openai langchain-community tavily-python python-dotenv
```

### Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=deep-learning-ai.lessons
```

## Usage

```bash
python main.py
```

The script runs four example queries using the same `thread_id`, so the agent accumulates conversation history across all of them:

1. Current weather in San Francisco
2. Current weather in SF and LA (parallel tool calls)
3. Multi-hop research: Super Bowl 2024 winner → team state → state GDP
4. Current weather in Atlanta (streamed output)

## Notes

- `gpt-4o-mini` is used throughout. Upgrading to `gpt-4o` or `gpt-4.1` requires an OpenAI account at Tier 2+ (minimum $50 spend).
- The Tavily search tool is configured to return up to 4 results per query.
- LangSmith tracing is enabled by default for observability.
- The in-memory SQLite store (`":memory:"`) does not persist between Python process restarts. Use a file path (e.g. `"checkpoints.sqlite"`) for durable persistence.
