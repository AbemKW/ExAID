import unittest
from unittest.mock import patch, AsyncMock
import asyncio
from exaid import EXAID
from summary_state import SummaryState
from buffer import TraceBuffer


class TestSummaryState(unittest.TestCase):
    def setUp(self):
        self.state = SummaryState()

    def test_add_agent(self):
        agent_id = "agent_1"
        self.state.add_agent(agent_id)
        self.assertIn(agent_id, self.state.state)
        self.assertEqual(self.state.state[agent_id]["traces"], [])
        self.assertEqual(self.state.state[agent_id]["summaries"], [])
        self.assertEqual(self.state.state[agent_id]["feedback"], [])

    def test_add_trace(self):
        agent_id = "agent_1"
        self.state.add_agent(agent_id)
        trace_text = "This is a test trace"
        self.state.add_trace(agent_id, trace_text)

        traces = self.state.state[agent_id]["traces"]
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["text"], trace_text)
        self.assertIn("trace_id", traces[0])
        self.assertIn("timestamp", traces[0])

    def test_add_summary(self):
        agent_id = "agent_1"
        self.state.add_agent(agent_id)
        summary_text = "This is a test summary"
        self.state.add_summary(agent_id, summary_text)

        summaries = self.state.state[agent_id]["summaries"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["text"], summary_text)
        self.assertIn("summary_id", summaries[0])
        self.assertIn("timestamp", summaries[0])

    def test_add_feedback(self):
        agent_id = "agent_1"
        self.state.add_agent(agent_id)
        feedback_text = "This is test feedback"
        self.state.add_feedback(agent_id, feedback_text)

        feedback = self.state.state[agent_id]["feedback"]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["text"], feedback_text)
        self.assertIn("feedback_id", feedback[0])
        self.assertIn("timestamp", feedback[0])


class TestTraceBuffer(unittest.TestCase):
    def setUp(self):
        self.callback_called = False
        self.callback_args = None

        def mock_callback(agent_id, combined_text):
            self.callback_called = True
            self.callback_args = (agent_id, combined_text)

        self.buffer = TraceBuffer(mock_callback, chunk_threshold=3)

    def test_add_chunk_below_threshold(self):
        agent_id = "agent_1"
        self.buffer.addchunk(agent_id, "chunk 1")
        self.buffer.addchunk(agent_id, "chunk 2")

        self.assertFalse(self.callback_called)
        self.assertEqual(len(self.buffer.buffer[agent_id]), 2)

    def test_add_chunk_at_threshold(self):
        agent_id = "agent_1"
        self.buffer.addchunk(agent_id, "chunk 1")
        self.buffer.addchunk(agent_id, "chunk 2")
        self.buffer.addchunk(agent_id, "chunk 3")

        self.assertTrue(self.callback_called)
        self.assertEqual(self.callback_args[0], agent_id)
        self.assertEqual(self.callback_args[1], "chunk 1\nchunk 2\nchunk 3")
        # Buffer should be reset
        self.assertEqual(len(self.buffer.buffer[agent_id]), 0)

    def test_multiple_agents(self):
        agent1 = "agent_1"
        agent2 = "agent_2"

        self.buffer.addchunk(agent1, "agent1 chunk 1")
        self.buffer.addchunk(agent2, "agent2 chunk 1")
        self.buffer.addchunk(agent1, "agent1 chunk 2")
        self.buffer.addchunk(agent1, "agent1 chunk 3")  # Should trigger for agent1

        self.assertTrue(self.callback_called)
        self.assertEqual(self.callback_args[0], agent1)
        self.assertEqual(self.callback_args[1], "agent1 chunk 1\nagent1 chunk 2\nagent1 chunk 3")

        # agent2 buffer should still have 1 chunk
        self.assertEqual(len(self.buffer.buffer[agent2]), 1)


class TestEXAID(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.exaid = EXAID(chunk_threshold=2)  # Lower threshold for easier testing

    @patch('summarizer_agent.summarize')
    def test_add_agent(self, mock_summarize):
        agent_name = "test_agent"
        agent_id = "agent_1"

        result = self.exaid.addAgent(agent_name, agent_id)
        self.assertIsNone(result)  # addAgent doesn't return anything

        self.assertIn(agent_id, self.exaid.agents)
        self.assertEqual(self.exaid.agents[agent_id], agent_name)
        self.assertIn(agent_id, self.exaid.graph.state)

    @patch('exaid.summarize')
    async def test_add_trace_at_threshold(self, mock_summarize):
        agent_id = "agent_1"
        self.exaid.addAgent("agent", agent_id)

        await self.exaid.addTrace(agent_id, "trace 1")

        # Should not have triggered summarization yet
        self.assertEqual(len(self.exaid.graph.state[agent_id]["summaries"]), 0)

    @patch('exaid.summarize')
    async def test_add_trace_at_threshold(self, mock_summarize):
        mock_summarize.return_value = "Mocked summary"

        agent_id = "agent_1"
        self.exaid.addAgent("agent", agent_id)

        await self.exaid.addTrace(agent_id, "trace 1")
        await self.exaid.addTrace(agent_id, "trace 2")  # Should trigger summarization

        # Should have called summarize
        mock_summarize.assert_called_once_with(agent_id, "trace 1\ntrace 2")

        # Should have added trace and summary
        self.assertEqual(len(self.exaid.graph.state[agent_id]["traces"]), 1)
        self.assertEqual(len(self.exaid.graph.state[agent_id]["summaries"]), 1)
        self.assertEqual(self.exaid.graph.state[agent_id]["summaries"][0]["text"], "Mocked summary")

    async def test_getsummary_no_summaries(self):
        agent_id = "agent_1"
        self.exaid.addAgent("agent", agent_id)

        summary = await self.exaid.getsummary(agent_id)
        self.assertEqual(summary, "")

    @patch('exaid.summarize')
    async def test_getsummary_with_summaries(self, mock_summarize):
        mock_summarize.return_value = "Latest summary"

        agent_id = "agent_1"
        self.exaid.addAgent("agent", agent_id)

        # Add traces to trigger summarization
        await self.exaid.addTrace(agent_id, "trace 1")
        await self.exaid.addTrace(agent_id, "trace 2")

        summary = await self.exaid.getsummary(agent_id)
        self.assertEqual(summary, "Latest summary")

    @patch('exaid.summarize')
    async def test_getfullsummary(self, mock_summarize):
        mock_summarize.return_value = "Summary 1"

        agent_id = "agent_1"
        self.exaid.addAgent("agent", agent_id)

        # Add traces to trigger summarization
        await self.exaid.addTrace(agent_id, "trace 1")
        await self.exaid.addTrace(agent_id, "trace 2")

        full_summaries = await self.exaid.getfullsummary(agent_id)
        self.assertEqual(len(full_summaries), 1)
        self.assertEqual(full_summaries[0]["text"], "Summary 1")


if __name__ == '__main__':
    unittest.main()