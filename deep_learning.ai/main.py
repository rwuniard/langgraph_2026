from langchain_openai import ChatOpenAI
from my_agent import Agent
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
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
abot = Agent(model, [tool], system=prompt).graph

def main():
    print("Hello from deep-learning-ai!")


    messages = [HumanMessage(content="What is the weather in sf?")]
    result = abot.invoke({"messages": messages})

    print(result)
    print(result['messages'][-1].content)

    messages = [HumanMessage(content="What is the weather in SF and LA?")]
    result = abot.invoke({"messages": messages})
    print(result)
    print(result['messages'][-1].content)

    # Note, the query was modified to produce more consistent results. 
    # Results may vary per run and over time as search information and models change.
    query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? What is the GDP of that state? Answer each question." 
    messages = [HumanMessage(content=query)]
    result = abot.invoke({"messages": messages})
    print(result)
    print(result['messages'][-1].content)




if __name__ == "__main__":
    main()
