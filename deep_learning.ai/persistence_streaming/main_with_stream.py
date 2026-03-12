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
            for node, values in chunk.items():
                print(f"{node}:")
                for msg in values['messages']:
                    msg.pretty_print()



if __name__ == "__main__":
    main()
