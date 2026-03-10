# Deep Learning AI - LangGraph Agent

A LangGraph-based ReAct agent built as part of the DeepLearning.AI course. The agent uses OpenAI's GPT models with Tavily web search to answer multi-step research questions.

## Project Structure

```
deep_learning.ai/
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

### Run as a script

```bash
python main.py
```

The script runs three example queries:
1. Current weather in San Francisco
2. Current weather in SF and LA (parallel tool calls)
3. Multi-hop research: Super Bowl 2024 winner → team state → state GDP

### Run with LangGraph dev server

```bash
uv run --active langgraph dev
```

This starts the LangGraph Studio UI locally, allowing you to inspect and interact with the graph visually.

The graph is configured in `langgraph.json`:

```json
{
  "graphs": {
    "agent": "./main.py:abot"
  }
}
```

`abot` is a module-level variable holding the compiled `StateGraph`. LangGraph's dev server imports the module and looks up the variable by name directly — it must be a `CompiledGraph` (or a factory function returning one) at the top level of the file.

## Notes

- `gpt-4o-mini` is used throughout. Upgrading to `gpt-4o` or `gpt-4.1` requires an OpenAI account at Tier 2+ (minimum $50 spend).
- The Tavily search tool is configured to return up to 4 results per query.
- LangSmith tracing is enabled by default for observability.
