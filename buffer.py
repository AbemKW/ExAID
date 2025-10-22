class TraceBuffer:
    def __init__(self, on_full_callback, chunk_threshold: int = 5):
        self.buffer = []
        self.on_full_callback = on_full_callback
        self.chunk_threshold = chunk_threshold

    def addchunk(self, chunk: str):
        if len(self.buffer) >= self.chunk_threshold:
            self.on_full_callback(self.buffer)
            self.buffer = []
        self.buffer.append(chunk)