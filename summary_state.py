from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class SummaryState(TypedDict):
    agent_name: str
    reasoning_trace: Annotated[
            list[str],
            "A list of reasoning steps taken by the agent so far."
        ]
    summary: str
    feedback: str