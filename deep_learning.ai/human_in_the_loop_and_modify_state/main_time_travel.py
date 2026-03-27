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
        messages = [HumanMessage(content="What is the weather in SF?")]
        print("Streaming...")
        for event in abot.stream({"messages": messages}, config=thread):
            for value in event.values():
                print(value)

        # Execute the agent after the interrupt. It basically requires a new stream call
        # from the current state. This allows us to do something here to get user input.
        print("Execute the state")
        for event in abot.stream(None, thread):
            for v in event.values():
                print(v)


        # Second Call to the agent.
        print("--------------------------------")

        # Call with another HumanMessage
        messages = [HumanMessage(content="What is the weather in NYC?")]
        print("Streaming...")
        for event in abot.stream({"messages": messages}, config=thread):
            for value in event.values():
                print(value)

        # Execute the agent after the interrupt. It basically requires a new stream call
        # from the current state. This allows us to do something here to get user input.
        print("Execute the state")
        for event in abot.stream(None, thread):
            for v in event.values():
                print(v)

        # Now Let's do the time travel.
        print("--------------------------------\n\n")
        print("Time travel...")
        print("--------------------------------\n\n")
        # Get the state
        states = []
        for i, state in enumerate(abot.get_state_history(thread)):
            print(f"State {i}: {state}")
            print("\n")
            states.append(state)
        
        print(f"States size: {len(states)}")
        # state[0] is the newest
        # state[-1] is the oldest == states[9] if it is the size of 10.
        replay_state = states[3]
        print("Replay state:")
        print(replay_state)
        
        print("--------------------------------\n\n")
        print("Replaying the states...")
        print("--------------------------------\n\n")
        # Replay the states.
        for event in abot.stream(None, replay_state.config):
            for v in event.items():
                print(f"Values: {v}")
        
        

if __name__ == "__main__":
    main()
