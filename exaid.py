from summary_state import SummaryState

class EXAID:
    def __init__(self):
        self.name = "EXAID"
        self.graph = SummaryState()
        self.agents = {}
    def addAgent(self, agent, id: str):
        self.agents[id] = agent
    def addTrace(self, text: str, id: str):
        self.graph.add_trace(id, text)
