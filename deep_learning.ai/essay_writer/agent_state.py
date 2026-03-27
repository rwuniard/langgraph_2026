from typing import TypedDict, List


# Agent State
class AgentState(TypedDict, total=False):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int