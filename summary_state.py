from typing import Annotated, TypedDict
from datetime import datetime, timezone
import uuid

class SummaryState():
    """
    A lightweight in-memory graph representation of reasoning state.
    Each agent has its own trace, summary, and feedback nodes.
    """

    def __init__(self):
        # Each key is agent_id; values are dicts of that agent's nodes
        self.state = {}
        self.summarizer_busy = False

    def add_agent(self, agent_id: str):
        if agent_id not in self.state:
            self.state[agent_id] = {
                "traces": [],
                "summaries": [],
                "feedback": []
            }
    def add_trace(self, agent_id: str, trace: str):
        trace_node ={
            "trace_id" : f"{uuid.uuid4()}|{agent_id}",
            "text": trace,
            "timestamp": datetime.now(timezone.utc)
        }
        self.state[agent_id]["traces"].append(trace_node)
    def add_summary(self, agent_id: str, summary: str):
        summary_node= {
            "summary_id": f"{uuid.uuid4()}|{agent_id}",
            "text": summary,
            "timestamp": datetime.now(timezone.utc)
        }
        self.state[agent_id]["summaries"].append(summary_node)
    def add_feedback(self, agent_id: str, feedback: str):
        feedback_node = {
            "feedback_id": f"{uuid.uuid4()}|{agent_id}",
            "text": feedback,
            "timestamp": datetime.now(timezone.utc)
        }
        self.state[agent_id]["feedback"].append(feedback_node)