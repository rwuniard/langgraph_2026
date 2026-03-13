from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # use add_messages instead of operator.add
    # This will allow messages with the same id to be replaced,
    # otherwise, the messages will be appended to the list.
    messages: Annotated[list[AnyMessage], add_messages]  # *****To reduce the messages*****