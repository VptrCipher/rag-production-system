"""
RAGAS evaluation — automated RAG quality measurement.

Metrics
-------
- **Faithfulness**: Is the answer grounded in the provided context?
- **Answer Correctness**: Does the answer match the ground truth?
- **Context Recall**: Were all relevant pieces of context retrieved?
- **Context Precision**: Are retrieved contexts relevant to the question?

Evaluation Dataset Design
-------------------------
Each sample requires:
  - question (str)
  - answer (str)          — the generated answer
  - contexts (list[str])  — retrieved context chunks
  - ground_truth (str)    — human-verified reference answer

Automated Testing Pipeline
---------------------------
1. Build evaluation dataset (manual or synthetic).
2. Run the full RAG pipeline for each question.
3. Evaluate with RAGAS metrics.
4. Log results to JSON + stdout.
5. Fail CI if any metric drops below threshold.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)

from config import get_settings

logger = structlog.get_logger(__name__)

# ── Default thresholds ────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.7,
    "answer_correctness": 0.6,
    "context_recall": 0.7,
    "context_precision": 0.7,
}


class RAGASEvaluator:
    """Run RAGAS evaluation on RAG pipeline outputs.

    Usage::

        evaluator = RAGASEvaluator()
        results = evaluator.evaluate(dataset_dict)
        evaluator.save_results(results, "eval_results.json")

        # Check CI thresholds
        evaluator.assert_thresholds(results)
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.settings = get_settings()
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.metrics = [
            faithfulness,
            answer_correctness,
            context_recall,
            context_precision,
        ]

    def evaluate(
        self,
        dataset_dict: Dict[str, List],
    ) -> Dict[str, Any]:
        """Run RAGAS evaluation.

        Parameters
        ----------
        dataset_dict : dict
            Must contain keys: question, answer, contexts, ground_truth.
            Each value is a list of equal length.

        Returns
        -------
        dict
            Metric scores and per-sample details.
        """
        logger.info(
            "evaluation_start",
            samples=len(dataset_dict.get("question", [])),
        )

        start = time.perf_counter()

        dataset = Dataset.from_dict(dataset_dict)

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )

        elapsed_s = round(time.perf_counter() - start, 2)

        scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_correctness": float(result["answer_correctness"]),
            "context_recall": float(result["context_recall"]),
            "context_precision": float(result["context_precision"]),
        }

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(dataset_dict["question"]),
            "scores": scores,
            "evaluation_time_s": elapsed_s,
            "thresholds": self.thresholds,
            "passed": all(scores[m] >= t for m, t in self.thresholds.items()),
        }

        logger.info("evaluation_complete", **scores, elapsed_s=elapsed_s)
        return report

    def assert_thresholds(self, report: Dict[str, Any]) -> None:
        """Raise if any metric is below its threshold (for CI)."""
        scores = report["scores"]
        failures: List[str] = []
        for metric, threshold in self.thresholds.items():
            if scores.get(metric, 0) < threshold:
                failures.append(f"{metric}: {scores[metric]:.3f} < {threshold:.3f}")
        if failures:
            msg = "RAGAS threshold check FAILED:\n" + "\n".join(failures)
            logger.error("threshold_check_failed", failures=failures)
            raise AssertionError(msg)
        logger.info("threshold_check_passed")

    @staticmethod
    def save_results(report: Dict[str, Any], path: str | Path) -> None:
        """Persist evaluation results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("results_saved", path=str(path))
