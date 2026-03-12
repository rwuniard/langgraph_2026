from langgraph.graph import StateGraph, START, END
from agent_state import AgentState
from langchain_core.messages import SystemMessage, ToolMessage


class Agent:
    """ The agent class is a wrapper around the LangGraph StateGraph class.
    It is used to create a graph of nodes and edges that represent the agent's behavior.
    The agent class is initialized with a model, tools, and system.
    The model is a LangChain model that is used to generate responses.
    The tools are a list of tools that the agent can use to interact with the world.
    The system is a string that is used to set the system prompt for the agent.
    This will have a memory component that will be used to store the agent's history.

    Agent Constructor Parameters:
    - model: A LangChain model that is used to generate responses.
    - tools: A list of tools that the agent can use to interact with the world.
    - checkpointer: A checkpointer that is used to store the agent's history.
    - system: A string that is used to set the system prompt for the agent.
    """
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        # compile the graph with the checkpointer
        # interrupt_before=["action"] will allow the agent to interrupt 
        # before the action node is executed.
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]) # *****To interrupt the action node*****
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t['name'] not in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


