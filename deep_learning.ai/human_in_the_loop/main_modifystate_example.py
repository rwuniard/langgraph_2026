from langchain_openai import ChatOpenAI
from my_agent import Agent
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver


import os
from dotenv import load_dotenv
load_dotenv()

print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("TAVILY_API_KEY"))
print(os.getenv("LANGSMITH_API_KEY"))
print(os.getenv("LANGSMITH_TRACING_V2"))
print(os.getenv("LANGSMITH_ENDPOINT"))
print(os.getenv("LANGSMITH_PROJECT"))


tool = TavilySearchResults(max_results=4) #increased number of results
print(type(tool))
print(tool.name)

prompt = """You are a smart research assistant. Use the search engine to look up information.
    You are allowed to make multiple calls (either together or in sequence).
    Only look up information when you are sure of what you want.
    If you need to look up some information before asking a follow up question, you are allowed to do that!"""

model = ChatOpenAI(model="gpt-4o-mini")  #reduce inference cost


def main():
    print("Hello from deep-learning-ai!")

    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, [tool], checkpointer=memory, system=prompt).graph

        thread = {"configurable": {"thread_id": "123"}}
        # Using stream
        messages = [HumanMessage(content="What is the weather in ATL?")]
        print("Streaming...")
        for chunk in abot.stream({"messages": messages}, config=thread):
            for value in chunk.values():
                print(value)

        print("Get the state of the agent:")
        state = abot.get_state(thread)
        print(state)

        print("Get the next state:")
        next_state = abot.get_state(thread).next
        print(next_state)


        # Modifying the state and asking the weather in NYC instead of ATL
        print("Modify the state:")
        # Get the state
        current_state = abot.get_state(thread)
        # Get the current values
        current_values = current_state.values["messages"][-1]
        print("Current values:")
        print(current_values)

        current_tool_call = current_values.tool_calls[0]
        print("Current tool call:")
        print(current_tool_call)

        current_tool_id = current_tool_call["id"]
        print("Current tool id:")
        print(current_tool_id)

        # Modify the state of the toool call
        current_values.tool_calls = [
            {
                "name": "tavily_search_results_json", 
                "args": {"query": "weather in NYC"}, 
                "id": current_tool_id, 
                "type": "tool_call"
            }
        ]
        print("Modified tool call:")
        print(current_values.tool_calls)

        # Set the state
        abot.update_state(thread, current_state)
        print("Check the state after modifying the tool call:")
        print(abot.get_state(thread))

        # Execute the agent with the modified state
        print("Execute the agent with the modified state:")
        for event in abot.stream(None, thread):
            for v in event.values():
                print(v)
        
        

if __name__ == "__main__":
    main()
