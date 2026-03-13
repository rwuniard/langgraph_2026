# Deep Learning AI - LangGraph Agent with Human-in-the-Loop & State Modification

A LangGraph-based ReAct agent built as part of the DeepLearning.AI course. The agent extends the persistence/streaming pattern with **human-in-the-loop** interrupts and **runtime state modification**, allowing a user to inspect or modify pending tool calls before execution.

## Project Structure

```
human_in_the_loop_and_modify_state/
├── main_with_stream.py         # Human-in-the-loop demo with user confirmation prompt
├── main_modifystate_example.py # State modification demo: redirect a tool call mid-execution
├── my_agent.py                 # Agent class with interrupt_before=["action"]
├── agent_state.py              # AgentState TypedDict using add_messages reducer
├── .env                        # API keys (not committed)
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

### Human-in-the-Loop (Interrupts)

The graph is compiled with `interrupt_before=["action"]`, which pauses execution before every tool call:

```python
self.graph = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["action"]
)
```

After the initial `.stream()` call the graph halts at the interrupt. You can:

- **Inspect the pending state** with `abot.get_state(thread)`
- **Check what runs next** with `abot.get_state(thread).next` — this will be `('action',)` when paused
- **Resume execution** by streaming `None` as the input:

```python
for event in abot.stream(None, thread):
    for v in event.values():
        print(v)
```

`main_with_stream.py` demonstrates an interactive loop where the user is prompted (`y/n`) before each tool execution continues.

### State Modification

Because execution is paused at the interrupt, you can rewrite the queued tool call before resuming. `main_modifystate_example.py` shows this pattern:

```python
# 1. Inspect the interrupted state
current_state = abot.get_state(thread)
current_values = current_state.values["messages"][-1]
current_tool_id = current_values.tool_calls[0]["id"]

# 2. Replace the tool call (keep same id, change query)
current_values.tool_calls = [
    {
        "name": "tavily_search_results_json",
        "args": {"query": "weather in NYC"},   # was "weather in ATL"
        "id": current_tool_id,
        "type": "tool_call"
    }
]

# 3. Push the modified state back
abot.update_state(thread, current_state)

# 4. Resume — tool now runs with the new query
for event in abot.stream(None, thread):
    for v in event.values():
        print(v)
```

The tool call `id` must be preserved so LangGraph can correctly associate the `ToolMessage` response.

### AgentState and `add_messages`

`agent_state.py` uses the `add_messages` reducer instead of `operator.add`:

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` merges incoming messages by `id`, so updating a state message (e.g. replacing a tool call) overwrites the existing entry rather than appending a duplicate.

### Persistence (Checkpointing)

The agent uses `SqliteSaver` to persist conversation state across invocations. Each conversation is identified by a `thread_id`, passed via the `config` argument:

```python
thread = {"configurable": {"thread_id": "123"}}
```

```python
with SqliteSaver.from_conn_string(":memory:") as memory:
    abot = Agent(model, [tool], checkpointer=memory, system=prompt).graph
```

- `SqliteSaver.from_conn_string(":memory:")` is a context manager — it must be used with `with`.
- Using the same `thread_id` across calls gives the agent memory of prior messages.

### Streaming

`.stream()` yields one dict per node execution:

```python
for chunk in abot.stream({"messages": messages}, config=thread):
    for value in chunk.values():
        print(value)
```

Each chunk is shaped as `{ "node_name": { "messages": [...] } }`:

- `"llm"` chunks contain an `AIMessage` (response or tool call decision)
- `"action"` chunks contain a `ToolMessage` (search result)

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

Create a `.env` file with your keys:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=deep-learning-ai.lessons
```

## Usage

**Human-in-the-loop demo** — streams a weather query, pauses before tool execution, prompts the user to confirm each step:

```bash
python main_with_stream.py
```

**State modification demo** — streams a weather query for ATL, intercepts the tool call at the interrupt, rewrites the query to NYC, then resumes:

```bash
python main_modifystate_example.py
```

## Notes

- `gpt-4o-mini` is used throughout. Upgrading to `gpt-4o` or `gpt-4.1` requires an OpenAI account at Tier 2+ (minimum $50 spend).
- The Tavily search tool is configured to return up to 4 results per query.
- LangSmith tracing is enabled by default for observability.
- The in-memory SQLite store (`":memory:"`) does not persist between Python process restarts. Use a file path (e.g. `"checkpoints.sqlite"`) for durable persistence.
- The tool call `id` must be kept unchanged when modifying state — LangGraph uses it to match the `ToolMessage` response back to the original call.
