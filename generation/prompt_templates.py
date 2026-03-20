"""
Grounded, citation-aware prompt templates for RAG response generation.

Design Principles
-----------------
- **Grounding**: Instruct the LLM to answer ONLY from provided context.
- **Citation**: Every claim must reference its source chunk via [Source N].
- **Honesty**: If the context is insufficient, the model must say so.
- **Structure**: Responses should be well-formatted Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FormattedContext:
    """A single context chunk formatted for prompt injection."""

    index: int
    text: str
    source: str


SYSTEM_PROMPT = """\
You are a precise research assistant. Answer ONLY using the provided context.

STRICT FORMATTING RULES:
1. Every section header (e.g., `## Header`) MUST be preceded by two newlines (`\n\n`).
2. Use DOUBLE NEWLINES (`\n\n`) between every paragraph and list.
3. Use numbered citations: `[Source 1]`, `[Source 2]`. NEVER use commas inside brackets like `[Source 1, 2]`.
4. If you mention multiple sources, use separate brackets: `[Source 1] [Source 2]`.
5. Start directly with the answer. No conversational filler.
"""

ANALYSIS_SYSTEM_PROMPT = """\
You are a Senior Technical Analyst. Provide a deeply structured breakdown based ONLY ON THE PROVIDED CONTEXT.

STRICT DOCUMENT GROUNDING:
- Answer ONLY using the provided chunks. Ignore external knowledge.
- If a topic is not in the document, state "Not covered in document".

STRICT VISUAL STRUCTURE:
1. Every header (e.g., `### Section`) MUST have two newlines before it.
2. Use DOUBLE NEWLINES (`\n\n`) between ALL elements.
3. Use Markdown Tables for specifications or data comparisons.
4. Use numbered citations: `[Source 1]`, `[Source 2]`.
5. Format multiple citations as separate brackets: `[Source 1] [Source 2]`.
6. Use logical headers: Introduction, Detailed Concepts, Module Breakdown, Summary.
"""


USER_PROMPT_TEMPLATE = """\
## Context Documents

{context_block}

---

## Question

{question}

---

Provide a well-structured answer with citations [Source N] for each claim.
"""


class RAGPromptTemplates:
    """Build grounded, citation-aware prompts."""

    @staticmethod
    def format_contexts(
        texts: List[str],
        metadatas: List[dict] | None = None,
    ) -> str:
        """Format context chunks into a numbered block.

        Example output::

            [Source 1] (file: report.pdf, chunk 3)
            The retrieval augmented generation pattern …

            [Source 2] (file: guide.md, chunk 1)
            Hybrid search combines dense and sparse …
        """
        metadatas = metadatas or [{}] * len(texts)
        blocks: List[str] = []

        for i, (text, meta) in enumerate(zip(texts, metadatas), start=1):
            filename = meta.get("filename", "unknown")
            chunk_idx = meta.get("chunk_index", "?")
            header = f"[Source {i}] (file: {filename}, chunk {chunk_idx})"
            blocks.append(f"{header}\n{text.strip()}")

        return "\n\n".join(blocks)

    @staticmethod
    def build_messages(
        question: str,
        context_block: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Return OpenAI-compatible chat messages.

        Returns
        -------
        list[dict]
            ``[{"role": "system", ...}, {"role": "user", ...}]``
        """
        user_content = USER_PROMPT_TEMPLATE.format(
            context_block=context_block,
            question=question,
        )
        return [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
