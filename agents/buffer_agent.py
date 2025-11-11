from datetime import datetime, UTC
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from llm import llm, summarizer
import queue
import re
import json
from pydantic import BaseModel

class TraceData(BaseModel):
    trace_text: List[str]
    count: int

class BufferAgent:
    """A tiny buffer that stores chunks per-agent.

    When a given agent's accumulated chunks reach `chunk_threshold`, the
    `on_full_callback` is invoked with two arguments: (agent_id, combined_text),
    and that agent's buffer is reset.
    """

    def __init__(self):
        self.buffer: list[str] = []
        self.llm = summarizer
        self.flag_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are monitoring the reasoning streams of multiple AI agents. "
            "Your task is to decide if the new reasoning trace should trigger summarization. "
            "Reply with EXACTLY 'YES' (all caps) if ANY of these conditions are met:\n"
            "- The new trace completes a thought or reasoning step\n"
            "- The topic or focus has changed from previous traces\n"
            "- Enough context has accumulated to warrant a summary\n"
            "- The reasoning has reached a natural pause or conclusion\n\n"
            "Reply with EXACTLY 'NO' (all caps) if the new trace is just continuing the same line of reasoning without completion.\n"
            "Be decisive - prefer YES when in doubt, as summaries help track agent progress."),
            ("user", "Previous traces in buffer:\n{previous_trace}\n\nNew trace to evaluate:\n{new_trace}\n\nShould this trigger summarization? Reply with only 'YES' or 'NO'."),
        ])
        self.traces: dict[str, TraceData] = {}
        # Stream token accumulation per agent
        self.stream_buffers: dict[str, str] = {}
        # Sentence validation prompt
        self.sentence_validation_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are validating sentence boundaries in reasoning text. "
            "Given a list of potential sentences detected by regex, validate and refine them. "
            "Consider edge cases like abbreviations (e.g., 'Dr.', 'etc.'), quotes, and medical terminology. "
            "Return a JSON list of validated complete sentences. "
            "If a detected boundary is incorrect (e.g., abbreviation period), merge sentences. "
            "If paragraphs are detected (double newlines), keep them as separate units. "
            "Return ONLY a JSON array of strings, no other text."),
            ("user", "Detected potential sentences:\n{sentences}\n\nReturn validated sentences as JSON array."),
        ])

    async def addchunk(self, agent_id: str, chunk: str) :

        tagged_chunk = f"| {agent_id} | {chunk}"

        # Add the new chunk to the traces
        if agent_id not in self.traces:
            self.traces[agent_id] = TraceData(trace_text=[], count=0)
        self.traces[agent_id].count += 1
        trace_text = f"| Timestamp: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} | Trace Length: {len(chunk)} | {chunk}"
        self.traces[agent_id].trace_text.append(trace_text)
        
        # If this is the first trace, always trigger
        if not self.buffer:
            self.buffer.append(tagged_chunk)
            return True
        # Get previous traces (before adding the new one)
        previous_traces = "\n".join(self.buffer)
        
        # Add the new chunk to buffer
        self.buffer.append(tagged_chunk)

        flag_chain = self.flag_prompt | self.llm
        flag_response = await flag_chain.ainvoke({
            "previous_trace": previous_traces,
            "new_trace": tagged_chunk
        })
        # Extract text content from AIMessage object
        response_text = flag_response.content.strip().upper()

        return "YES" in response_text
    
    def peek(self) -> list[str]:
        """Get a copy of the current buffer without flushing it."""
        return self.buffer.copy()
    
    def flush(self) -> list[str]:
        flushed = self.buffer.copy()
        self.buffer.clear()
        return flushed
        
    def get_trace_count(self, agent_id: str) -> int:
        return self.traces.get(agent_id, TraceData(trace_text=[], count=0)).count
    
    def get_traces(self, agent_id: str) -> List[str]:
        return self.traces.get(agent_id, TraceData(trace_text=[], count=0)).trace_text
    
    def _accumulate_tokens(self, agent_id: str, token: str) -> str:
        """Accumulates tokens for an agent and returns the accumulated text."""
        if agent_id not in self.stream_buffers:
            self.stream_buffers[agent_id] = ""
        self.stream_buffers[agent_id] += token
        return self.stream_buffers[agent_id]
    
    def _detect_sentence_boundaries(self, text: str) -> tuple[list[str], str]:
        """Uses regex to detect potential sentence boundaries.
        
        Returns:
            tuple: (list of complete sentences/paragraphs, remaining incomplete text)
        """
        sentences = []
        remaining = text
        
        # First, check for paragraph breaks (double newlines or more)
        # Split on paragraph breaks while keeping the breaks
        parts = re.split(r'(\n\n+)', text)
        
        if len(parts) > 1:
            # We have paragraph breaks - process each paragraph separately
            # parts alternates: [text, break, text, break, ...]
            i = 0
            while i < len(parts):
                if re.match(r'\n\n+', parts[i]):
                    # Skip break markers
                    i += 1
                    continue
                
                para_text = parts[i]
                # Check if next part is a paragraph break (meaning this para is complete)
                if i + 1 < len(parts) and re.match(r'\n\n+', parts[i + 1]):
                    # Complete paragraph - process for sentences
                    para_sentences, para_remaining = self._detect_sentences_in_text(para_text)
                    sentences.extend(para_sentences)
                    # If there's remaining incomplete text in this para, it becomes remaining
                    if para_remaining.strip():
                        remaining = para_remaining
                    i += 2  # Move past text and break marker
                else:
                    # Last part (no break after) - might be incomplete
                    para_sentences, para_remaining = self._detect_sentences_in_text(para_text)
                    sentences.extend(para_sentences)
                    remaining = para_remaining
                    break
        else:
            # No paragraph breaks, just detect sentences
            sentences, remaining = self._detect_sentences_in_text(text)
        
        return sentences, remaining
    
    def _detect_sentences_in_text(self, text: str) -> tuple[list[str], str]:
        """Helper to detect sentences within a text block."""
        # Pattern: sentence ending punctuation followed by whitespace
        # But be careful with abbreviations - this is a simple version, LLM will refine
        sentence_pattern = r'([.!?]+)(\s+)'
        
        sentences = []
        matches = list(re.finditer(sentence_pattern, text))
        
        if not matches:
            return [], text
        
        last_end = 0
        for match in matches:
            start = last_end
            end = match.end()
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            last_end = end
        
        remaining = text[last_end:].strip()
        return sentences, remaining
    
    async def _validate_sentence_boundaries(self, sentences: list[str]) -> list[str]:
        """Uses LLM to validate and refine sentence boundaries."""
        if not sentences:
            return []
        
        try:
            validation_chain = self.sentence_validation_prompt | self.llm
            response = await validation_chain.ainvoke({
                "sentences": "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
            })
            
            # Try to extract JSON array from response
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            # Try to parse as JSON
            try:
                validated = json.loads(response_text)
                if isinstance(validated, list):
                    return [str(s).strip() for s in validated if s.strip()]
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract list from text
                # Look for array-like structure
                array_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
                if array_match:
                    # Try to extract strings from the array
                    content = array_match.group(1)
                    # Simple extraction - split by commas and clean
                    items = re.findall(r'"([^"]+)"', content)
                    if items:
                        return [s.strip() for s in items if s.strip()]
            
            # Fallback: return original sentences if validation fails
            return sentences
        except Exception as e:
            # On error, return original sentences
            return sentences
    
    async def _process_complete_units(self, agent_id: str, complete_units: list[str]) -> Optional[bool]:
        """Processes validated sentences/paragraphs through existing addchunk logic.
        
        Returns:
            Optional[bool]: True if summarization was triggered, False otherwise, None if no units to process
        """
        if not complete_units:
            return None
        
        # Process each complete unit through addchunk
        last_trigger = False
        for unit in complete_units:
            if unit.strip():
                trigger = await self.addchunk(agent_id, unit.strip())
                if trigger:
                    last_trigger = True
        
        return last_trigger
    
    async def add_streamed_tokens(self, agent_id: str, token: str) -> Optional[bool]:
        """Main entry point for streamed tokens.
        
        Accumulates tokens, detects sentence boundaries, validates with LLM,
        and processes complete units through addchunk.
        
        Returns:
            Optional[bool]: True if summarization was triggered, False otherwise, None if no complete units yet
        """
        # Accumulate tokens
        accumulated = self._accumulate_tokens(agent_id, token)
        
        # Detect sentence boundaries
        sentences, remaining = self._detect_sentence_boundaries(accumulated)
        
        if not sentences:
            # No complete sentences yet
            return None
        
        # Validate sentences with LLM
        validated_sentences = await self._validate_sentence_boundaries(sentences)
        
        if validated_sentences:
            # Update stream buffer to keep only remaining incomplete text
            self.stream_buffers[agent_id] = remaining
            
            # Process complete units
            trigger = await self._process_complete_units(agent_id, validated_sentences)
            return trigger
        
        return None
    
    def flush_stream_buffer(self, agent_id: str) -> Optional[str]:
        """Flushes any remaining incomplete content from stream buffer.
        
        Returns:
            Optional[str]: Remaining incomplete text, or None if buffer is empty
        """
        if agent_id in self.stream_buffers and self.stream_buffers[agent_id]:
            remaining = self.stream_buffers[agent_id]
            self.stream_buffers[agent_id] = ""
            return remaining.strip() if remaining.strip() else None
        return None