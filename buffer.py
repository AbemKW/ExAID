class TraceBuffer:
    """A tiny buffer that stores chunks per-agent.

    When a given agent's accumulated chunks reach `chunk_threshold`, the
    `on_full_callback` is invoked with two arguments: (agent_id, combined_text),
    and that agent's buffer is reset.
    """

    def __init__(self, on_full_callback, chunk_threshold: int):
        # map agent_id -> list[str]
        self.buffer: dict[str, list[str]] = {}
        self.on_full_callback = on_full_callback
        self.chunk_threshold = chunk_threshold

    def addchunk(self, agent_id: str, chunk: str):
        # initialize list for agent if necessary
        if agent_id not in self.buffer:
            self.buffer[agent_id] = []

        self.buffer[agent_id].append(chunk)

        # if we've reached the threshold for this agent, combine and emit
        if len(self.buffer[agent_id]) >= self.chunk_threshold:
            combined = "\n".join(self.buffer[agent_id])
            # callback expected to accept (agent_id, combined_text)
            try:
                self.on_full_callback(agent_id, combined)
            finally:
                # reset only this agent's buffer
                self.buffer[agent_id] = []