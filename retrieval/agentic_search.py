"""
Agentic Retrieval — uses query decomposition to solve multi-hop questions.
Instead of a single search, the agent breaks the query into sub-questions,
retrieves context for each, and synthesizes a final answer.
"""

from __future__ import annotations

from typing import List, Optional

import structlog
from llama_index.core import QueryBundle, Settings, VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import get_settings
from config.qdrant_client import get_qdrant_client

logger = structlog.get_logger(__name__)


class AgenticSearcher:
    """Agentic searcher for multi-hop reasoning."""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_qdrant_client()

        # Initialize Vector Store & Index
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=self.settings.qdrant_collection)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        # Create a base query engine
        self.base_query_engine = self.index.as_query_engine(similarity_top_k=int(self.settings.retrieval_top_k))

        # Define tools for the agent
        query_engine_tools = [
            QueryEngineTool(
                query_engine=self.base_query_engine,
                metadata=ToolMetadata(
                    name="knowledge_base",
                    description="Provides information from the ingested documents.",
                ),
            ),
        ]

        # Initialize the SubQuestionQueryEngine with explicit LLM question generator
        self.agent = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),
            use_async=True,
        )
        logger.info("agentic_searcher_initialized")

    async def search(self, query: str) -> str:
        """Run agentic search with a robust fallback to standard retrieval."""
        print(f"!!! AGENT_SEARCH_START !!! query='{query}'")
        logger.info("agentic_search_start", query=query)

        try:
            # 1. Try Agentic Search (Multi-hop)
            print(f"DEBUGLOG: Attempting SubQuestionQueryEngine for query: {query}")
            response = await self.agent.aquery(query)

            # 2. Validate Agentic Response (Check for both nulls and "Empty Response" string)
            answer = ""
            if response and hasattr(response, "response"):
                answer = response.response.strip()

            print(f"DEBUGLOG: Raw agent response: '{answer}'")

            if answer and answer != "" and "Empty Response" not in answer:
                # Post-process: Ensure double newlines before headers for proper Markdown rendering
                import re

                answer = re.sub(r"([^\n])\n?##", r"\1\n\n##", answer)

                print(f"DEBUGLOG: Agentic search successful. length: {len(answer)}")
                return answer

            # 3. Fallback to Standard Search if Agentic is empty or says 'Empty Response'
            print("DEBUGLOG: Agentic response was inconclusive. Falling back to base_query_engine...")
            fallback_response = self.base_query_engine.query(query)

            if fallback_response and hasattr(fallback_response, "response") and fallback_response.response:
                return f"*(Agentic reasoning fallback to standard search)* \n\n {fallback_response.response}"

            return "I searched the knowledge base but couldn't find a detailed specific answer. Try rephrasing your question."

        except Exception as e:
            print(f"!!! AGENT_SEARCH_EXCEPTION !!! {str(e)}")
            logger.error("agentic_search_failed", error=str(e))

            # Emergency Fallback on Exception
            try:
                print("DEBUGLOG: Exception caught, trying emergency base search...")
                emergency_resp = self.base_query_engine.query(query)
                return str(emergency_resp)
            except:
                return f"Agent Error: {str(e)}"
