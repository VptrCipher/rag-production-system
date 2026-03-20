"""
Query Router for intelligent query processing.

Determines if a query requires RAG retrieval or if it's a simple conversational
greeting/question that the LLM can handle directly.
"""

from typing import Literal

import structlog

from config import get_settings

logger = structlog.get_logger(__name__)

class QueryRouter:
    """Routes queries to either the RAG pipeline or direct LLM."""

    def __init__(self):
        self.settings = get_settings()
        
        # Use Groq or OpenAI for the router LLM
        if self.settings.groq_api_key:
            from groq import Groq
            self.client = Groq(api_key=self.settings.groq_api_key)
            self.model = "llama-3.1-8b-instant" # Fast model for routing
            self._backend = "groq"
        elif self.settings.openai_api_key:
            import openai
            self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
            self.model = "gpt-4o-mini"
            self._backend = "openai"
        else:
            self.client = None
            self._backend = "echo"

    def route_query(self, query: str) -> Literal["RAG", "CONVERSATIONAL"]:
        """Determine if a query needs retrieval.
        
        Very simple implementation utilizing a fast LLM call.
        """
        if self._backend == "echo":
            # Default to RAG if no LLM configured
            return "RAG"

        prompt = f"""You are a specialized routing assistant. Your job is to classify the user's query into one of two categories:
1. "CONVERSATIONAL": The query is a simple greeting, pleasantry, or a general question that does not require searching an external knowledge base (e.g., "Hello", "How are you?").
2. "RAG": The query is asking for factual information, requires domain knowledge, or asks a specific question that should be looked up in a database (e.g., "What is RAG?", "How does the system work?").

Query: "{query}"

Respond with EXACTLY one word: either CONVERSATIONAL or RAG.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            decision = response.choices[0].message.content.strip().upper()
            if "RAG" in decision:
                return "RAG"
            elif "CONVERSATIONAL" in decision:
                return "CONVERSATIONAL"
            return "RAG" # Fallback
        except Exception as e:
            logger.error("query_router_error", error=str(e))
            return "RAG" # Fallback to retrieval on error
