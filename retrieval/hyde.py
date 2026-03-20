"""Hypothetical Document Embeddings (HyDE) generator."""
from config import get_settings
import structlog

logger = structlog.get_logger(__name__)

class HyDEGenerator:
    """Generates a hypothetical document/answer to improve similarity search."""
    def __init__(self):
        self.settings = get_settings()
        if self.settings.groq_api_key:
            from groq import Groq
            self.client = Groq(api_key=self.settings.groq_api_key)
            self.model = "llama-3.1-8b-instant"
            self._backend = "groq"
        elif self.settings.openai_api_key:
            import openai
            self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
            self.model = "gpt-4o-mini"
            self._backend = "openai"
        else:
            self.client = None
            self._backend = "none"

    def generate(self, query: str) -> str:
        """Generate a hypothetical answer to the query."""
        if not self.client:
            return query
            
        prompt = f"""You are a knowledgeable expert. Please write a short, factual answer to the following question. Do not include any conversational filler.
Question: {query}
Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            hypothetical_doc = response.choices[0].message.content.strip()
            logger.info("hyde_generated", query=query, doc_length=len(hypothetical_doc))
            return hypothetical_doc
        except Exception as e:
            logger.error("hyde_generation_error", error=str(e))
            return query
