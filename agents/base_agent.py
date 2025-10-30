from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    @abstractmethod
    async def act(self, input: str) -> str:
        pass