"""
Evaluation dataset construction utilities.

Builds evaluation datasets in the format expected by RAGAS:

    {
        "question":        str,
        "answer":          str,   # generated answer
        "contexts":        list[str],
        "ground_truth":    str,   # human-verified reference answer
    }

Two modes of dataset creation:
1. **Manual** — human-curated QA pairs with ground truth.
2. **Synthetic** — auto-generated from ingested documents using LLM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from config import get_settings

logger = structlog.get_logger(__name__)


class EvaluationDatasetBuilder:
    """Build and manage evaluation datasets.

    Usage::

        builder = EvaluationDatasetBuilder()
        builder.add_sample(
            question="What is RAG?",
            answer="RAG combines retrieval with generation...",
            contexts=["RAG is an architecture...", "The pattern uses..."],
            ground_truth="Retrieval-Augmented Generation combines...",
        )
        builder.save("eval_dataset.json")
    """

    def __init__(self):
        self.samples: List[Dict[str, Any]] = []

    def add_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> None:
        """Add a single evaluation sample."""
        self.samples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    def add_from_pipeline(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> None:
        """Add a sample directly from pipeline output (ground_truth optional)."""
        self.samples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth or "",
            }
        )

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(self.samples),
            "samples": self.samples,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info("dataset_saved", path=str(path), samples=len(self.samples))

    def load(self, path: str | Path) -> None:
        """Load dataset from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        self.samples = data.get("samples", [])
        logger.info("dataset_loaded", path=str(path), samples=len(self.samples))

    def to_ragas_dataset(self) -> dict:
        """Convert to RAGAS-compatible dict-of-lists format."""
        return {
            "question": [s["question"] for s in self.samples],
            "answer": [s["answer"] for s in self.samples],
            "contexts": [s["contexts"] for s in self.samples],
            "ground_truth": [s["ground_truth"] for s in self.samples],
        }

    def generate_synthetic_questions(
        self,
        texts: List[str],
        num_questions: int = 10,
    ) -> List[Dict[str, str]]:
        """Generate synthetic QA pairs from document texts using OpenAI.

        This creates an evaluation dataset without manual annotation.
        """
        import openai

        settings = get_settings()
        client = openai.OpenAI(api_key=settings.openai_api_key)

        synthetic_pairs: List[Dict[str, str]] = []

        for text in texts[:num_questions]:
            prompt = (
                "Given the following text, generate exactly ONE question that "
                "can be answered from it, and provide the ground truth answer.\n\n"
                f"Text:\n{text[:2000]}\n\n"
                'Respond in JSON: {"question": "...", "ground_truth": "..."}'
            )
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            try:
                content = response.choices[0].message.content or ""
                # Parse JSON from response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    pair = json.loads(content[start:end])
                    synthetic_pairs.append(pair)
            except (json.JSONDecodeError, IndexError):
                logger.warning("synthetic_parse_error", text_preview=text[:100])

        logger.info("synthetic_generation", pairs=len(synthetic_pairs))
        return synthetic_pairs
