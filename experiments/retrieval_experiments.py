"""
Experimentation framework — compare retrieval strategies, chunk sizes,
embedding models, and rerankers in a structured way.

Each experiment run produces a JSON report with metrics and configuration
so results are reproducible and comparable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from config import get_settings
from evaluation.dataset_builder import EvaluationDatasetBuilder
from evaluation.ragas_evaluator import RAGASEvaluator
from generation.response_generator import ResponseGenerator
from reranking.cohere_rerank import CohereReranker
from retrieval.hybrid_search import HybridSearcher

logger = structlog.get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "text-embedding-3-small"
    retrieval_top_k: int = 30
    rerank_top_n: int = 5
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    llm_model: str = "gpt-4o"
    description: str = ""


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    config: ExperimentConfig
    scores: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    total_latency_s: float = 0.0
    avg_latency_ms: float = 0.0
    timestamp: str = ""


class RetrievalExperimentRunner:
    """Run comparative experiments across RAG configurations.

    Usage::

        runner = RetrievalExperimentRunner()

        # Define experiments
        configs = [
            ExperimentConfig(name="chunk_256", chunk_size=256),
            ExperimentConfig(name="chunk_512", chunk_size=512),
            ExperimentConfig(name="chunk_1024", chunk_size=1024),
        ]

        results = runner.run_experiments(configs, eval_dataset_path="eval_data.json")
        runner.save_comparison(results, "experiment_results.json")
    """

    def __init__(self):
        self.settings = get_settings()
        self.evaluator = RAGASEvaluator()

    def run_single(
        self,
        config: ExperimentConfig,
        questions: List[str],
        ground_truths: List[str],
    ) -> ExperimentResult:
        """Run a single experiment configuration.

        Parameters
        ----------
        config : ExperimentConfig
            Parameters for this experiment.
        questions : list[str]
            Evaluation questions.
        ground_truths : list[str]
            Reference answers for evaluation.
        """
        logger.info("experiment_start", name=config.name)

        hybrid = HybridSearcher()
        hybrid.bm25.build_index()
        reranker = CohereReranker()
        generator = ResponseGenerator(model=config.llm_model)

        answers: List[str] = []
        all_contexts: List[List[str]] = []
        total_latency = 0.0

        for question in questions:
            start = time.perf_counter()

            # Retrieve
            candidates = hybrid.search(
                query=question,
                top_k=config.retrieval_top_k,
                vector_weight=config.vector_weight,
                bm25_weight=config.bm25_weight,
            )

            # Rerank
            reranked = reranker.rerank(question, candidates, top_n=config.rerank_top_n)

            # Generate
            result = generator.generate(query=question, contexts=reranked)

            elapsed = time.perf_counter() - start
            total_latency += elapsed

            answers.append(result.answer)
            all_contexts.append([c.text for c in reranked])

        # Evaluate with RAGAS
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": all_contexts,
            "ground_truth": ground_truths,
        }
        eval_report = self.evaluator.evaluate(dataset_dict)

        return ExperimentResult(
            config=config,
            scores=eval_report["scores"],
            num_samples=len(questions),
            total_latency_s=round(total_latency, 2),
            avg_latency_ms=round((total_latency / len(questions)) * 1000, 2) if questions else 0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def run_experiments(
        self,
        configs: List[ExperimentConfig],
        eval_dataset_path: str,
    ) -> List[ExperimentResult]:
        """Run multiple experiments and return all results."""
        builder = EvaluationDatasetBuilder()
        builder.load(eval_dataset_path)
        data = builder.to_ragas_dataset()

        results: List[ExperimentResult] = []
        for config in configs:
            result = self.run_single(
                config=config,
                questions=data["question"],
                ground_truths=data["ground_truth"],
            )
            results.append(result)
            logger.info(
                "experiment_complete",
                name=config.name,
                scores=result.scores,
            )

        return results

    @staticmethod
    def save_comparison(
        results: List[ExperimentResult],
        path: str | Path,
    ) -> None:
        """Save all experiment results for comparison."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiments": [
                {
                    "name": r.config.name,
                    "config": {
                        "chunk_size": r.config.chunk_size,
                        "chunk_overlap": r.config.chunk_overlap,
                        "embedding_model": r.config.embedding_model,
                        "retrieval_top_k": r.config.retrieval_top_k,
                        "rerank_top_n": r.config.rerank_top_n,
                        "vector_weight": r.config.vector_weight,
                        "bm25_weight": r.config.bm25_weight,
                        "llm_model": r.config.llm_model,
                    },
                    "scores": r.scores,
                    "num_samples": r.num_samples,
                    "avg_latency_ms": r.avg_latency_ms,
                    "timestamp": r.timestamp,
                }
                for r in results
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("comparison_saved", path=str(path), experiments=len(results))


# ── Convenience: pre-defined experiment sets ──────────────
def chunk_size_experiments() -> List[ExperimentConfig]:
    """Compare chunk sizes: 256, 512, 1024."""
    return [
        ExperimentConfig(name="chunk_256", chunk_size=256, description="Small chunks"),
        ExperimentConfig(name="chunk_512", chunk_size=512, description="Medium chunks (default)"),
        ExperimentConfig(name="chunk_1024", chunk_size=1024, description="Large chunks"),
    ]


def retrieval_strategy_experiments() -> List[ExperimentConfig]:
    """Compare retrieval strategies: vector-only, bm25-only, hybrid."""
    return [
        ExperimentConfig(name="vector_only", vector_weight=1.0, bm25_weight=0.0),
        ExperimentConfig(name="bm25_only", vector_weight=0.0, bm25_weight=1.0),
        ExperimentConfig(name="hybrid_70_30", vector_weight=0.7, bm25_weight=0.3),
        ExperimentConfig(name="hybrid_50_50", vector_weight=0.5, bm25_weight=0.5),
    ]


def reranker_experiments() -> List[ExperimentConfig]:
    """Compare with and without reranking (different top_n)."""
    return [
        ExperimentConfig(name="no_rerank", rerank_top_n=30),  # effectively no reranking
        ExperimentConfig(name="rerank_top_3", rerank_top_n=3),
        ExperimentConfig(name="rerank_top_5", rerank_top_n=5),
        ExperimentConfig(name="rerank_top_10", rerank_top_n=10),
    ]
