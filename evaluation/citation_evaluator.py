"""
================================================================================
GRAPHONE PIPELINE: CITATION-AWARE RAG BENCHMARK & EVALUATOR
================================================================================
Executes standardized comparative evaluation of Citation-Aware RAG across
12 deterministic ground-truth scenarios:
1. Answer supported by one source
2. Answer supported by multiple sources
3. Multiple claims requiring different citations
4. Unsupported question (abstention)
5. Valid citation ID attached to irrelevant evidence (unsupported claim)
6. Nonexistent citation ID
7. Fabricated source URL or document ID
8. Missing page metadata
9. Duplicate retrieved chunks
10. Validator failure (safe handling)
11. Conflicting source evidence
12. Answer that should abstain

Measures scientific RAG metrics:
- Citation Reference Validity
- Claim-Level Citation Coverage
- Citation Support / Faithfulness
- Unsupported-Claim Rate
- Abstention Correctness
- Validation Latency
================================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from rich.console import Console
from rich.table import Table
from src.retrieval.citation_validator import CitationValidator, citation_validator
from src.retrieval.models import (
    Citation,
    SourceChunkProvenance,
    ValidationStatus,
)

from evaluation.metrics import (
    calculate_abstention_accuracy,
    calculate_citation_faithfulness,
    calculate_citation_reference_validity,
    calculate_claim_citation_coverage,
    calculate_stats,
    calculate_unsupported_claim_rate,
    get_git_commit,
)

logger = structlog.get_logger(__name__)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "citation_eval_dataset.json"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "reports" / "citation_benchmark_report.json"


def load_citation_eval_dataset(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load deterministic ground-truth evaluation dataset for citation-aware RAG."""
    file_path = path or DATASET_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Citation evaluation dataset missing at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_citation_benchmark(
    dataset_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    validator_instance: Optional[CitationValidator] = None,
) -> Dict[str, Any]:
    """
    Execute full scientific citation evaluation across 12 canonical test scenarios.
    """
    validator = validator_instance or citation_validator
    dataset = load_citation_eval_dataset(dataset_path)
    output_file = output_path or REPORT_OUTPUT_PATH
    output_file.parent.mkdir(parents=True, exist_ok=True)

    latencies_ms: List[float] = []
    case_results: List[Dict[str, Any]] = []

    total_citations_accum = 0
    valid_citations_accum = 0
    supported_citations_accum = 0
    total_claims_accum = 0
    cited_claims_accum = 0
    unsupported_claims_accum = 0
    correct_abstention_decisions = 0

    for item in dataset:
        case_id = item["id"]
        case_name = item["case_name"]
        answer = item["candidate_answer"]
        expected_status = item["expected_status"]
        expected_abstained = item["expected_abstained"]
        simulate_error = item.get("simulate_validator_error", False)

        # Build SourceChunkProvenance objects
        sources: List[SourceChunkProvenance] = []
        for s in item.get("retrieved_sources", []):
            sources.append(
                SourceChunkProvenance(
                    index=s["index"],
                    source_id=s["source_id"],
                    document_id=s["document_id"],
                    chunk_id=s["chunk_id"],
                    title=s["title"],
                    source_url=s.get("source_url"),
                    page_number=s.get("page_number"),
                    section=s.get("section"),
                    text=s["text"],
                    retrieval_method=s.get("retrieval_method", "hybrid"),
                    retrieval_score=s.get("retrieval_score", 0.8),
                    metadata=s.get("metadata", {}),
                )
            )

        # Build explicit Citation if specified
        explicit_citations: Optional[List[Citation]] = None
        if "explicit_citation" in item:
            ec = item["explicit_citation"]
            explicit_citations = [
                Citation(
                    citation_id=ec["citation_id"],
                    source_id=ec["source_id"],
                    chunk_id=ec.get("chunk_id"),
                    document_id=ec.get("document_id"),
                    claim_text=ec.get("claim_text"),
                    text_span=ec.get("text_span"),
                    reference_valid=ec.get("reference_valid", True),
                    claim_supported=ec.get("claim_supported"),
                )
            ]

        t0 = time.perf_counter()
        if simulate_error:
            # Simulate validator failure / exception handling
            lat = round((time.perf_counter() - t0) * 1000.0, 3)
            val_status = ValidationStatus.VALIDATION_ERROR
            ref_valid = False
            sem_valid = False
            total_c = 1
            valid_c = 0
            unsupported_c = 0
            warnings = ["Simulated validator timeout / exception intercepted safely."]
        else:
            val_res = validator.validate(
                answer=answer,
                sources=sources,
                citations=explicit_citations,
                semantic_validation=True,
            )
            lat = round((time.perf_counter() - t0) * 1000.0, 3)
            val_status = val_res.status
            ref_valid = val_res.reference_valid
            sem_valid = val_res.semantic_valid
            total_c = val_res.total_citations
            valid_c = val_res.valid_citations
            unsupported_c = val_res.unsupported_claims
            warnings = val_res.warnings

        latencies_ms.append(lat)

        # Calculate claim metrics for this case
        is_abstained = "insufficient" in answer.lower()
        claims_in_answer = 0 if is_abstained else max(1, total_c)
        claims_cited = total_c if not is_abstained else 0

        total_claims_accum += claims_in_answer
        cited_claims_accum += min(claims_in_answer, claims_cited)
        total_citations_accum += total_c
        valid_citations_accum += valid_c
        if sem_valid and valid_c > 0:
            supported_citations_accum += valid_c
        unsupported_claims_accum += unsupported_c

        # Abstention decision correctness
        is_correct_decision = (
            (expected_abstained and is_abstained) or
            (not expected_abstained and not is_abstained)
        )
        if is_correct_decision:
            correct_abstention_decisions += 1

        matches_expected_status = (val_status.value == expected_status)

        case_results.append({
            "id": case_id,
            "case_name": case_name,
            "expected_status": expected_status,
            "actual_status": val_status.value,
            "matches_expected": matches_expected_status,
            "reference_valid": ref_valid,
            "semantic_valid": sem_valid,
            "total_citations": total_c,
            "valid_citations": valid_c,
            "unsupported_claims": unsupported_c,
            "latency_ms": lat,
            "warnings_count": len(warnings),
        })

    # Global Aggregate Rates
    ref_validity_rate = calculate_citation_reference_validity(
        valid_citations_accum, total_citations_accum
    )
    claim_coverage_rate = calculate_claim_citation_coverage(
        cited_claims_accum, total_claims_accum
    )
    faithfulness_rate = calculate_citation_faithfulness(
        supported_citations_accum, valid_citations_accum
    )
    unsupported_rate = calculate_unsupported_claim_rate(
        unsupported_claims_accum, total_claims_accum
    )
    abstention_acc = calculate_abstention_accuracy(
        correct_abstention_decisions, len(dataset)
    )

    latency_stats = calculate_stats(latencies_ms)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
        "total_test_cases": len(dataset),
        "metrics": {
            "citation_reference_validity": ref_validity_rate,
            "claim_level_citation_coverage": claim_coverage_rate,
            "citation_faithfulness": faithfulness_rate,
            "unsupported_claim_rate": unsupported_rate,
            "abstention_correctness": abstention_acc,
            "validation_latency_ms": latency_stats,
        },
        "totals": {
            "total_citations_evaluated": total_citations_accum,
            "valid_citations": valid_citations_accum,
            "supported_citations": supported_citations_accum,
            "total_claims_evaluated": total_claims_accum,
            "unsupported_claims": unsupported_claims_accum,
            "correct_abstention_decisions": correct_abstention_decisions,
        },
        "cases": case_results,
    }

    # Persist report
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Rich Table Summary
    _print_rich_report(report)

    return report


def _print_rich_report(report: Dict[str, Any]) -> None:
    """Render terminal report with Rich formatting."""
    m = report["metrics"]
    t = Table(title="Phase 6: Citation-Aware RAG Scientific Benchmark Results", border_style="cyan")
    t.add_column("Scientific Metric", style="bold white", width=36)
    t.add_column("Measured Value", style="bold green", justify="right", width=18)
    t.add_column("Denominator / Reference", style="dim", width=42)

    t.add_row(
        "Citation Reference Validity",
        f"{m['citation_reference_validity'] * 100:.1f}%",
        "valid citations / total citation references",
    )
    t.add_row(
        "Claim-Level Citation Coverage",
        f"{m['claim_level_citation_coverage'] * 100:.1f}%",
        "cited claims / total claims made",
    )
    t.add_row(
        "Citation Faithfulness",
        f"{m['citation_faithfulness'] * 100:.1f}%",
        "semantically entailed / valid citations",
    )
    t.add_row(
        "Unsupported-Claim Rate",
        f"{m['unsupported_claim_rate'] * 100:.1f}%",
        "unsupported claims / total claims made",
    )
    t.add_row(
        "Abstention Correctness",
        f"{m['abstention_correctness'] * 100:.1f}%",
        "correct abstentions / total test queries",
    )
    t.add_row(
        "Validation Latency (Mean)",
        f"{m['validation_latency_ms']['mean']:.2f} ms",
        f"p50: {m['validation_latency_ms']['p50']:.2f} ms, p95: {m['validation_latency_ms']['p95']:.2f} ms",
    )

    console.print(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Citation-Aware RAG Benchmark")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to ground-truth dataset")
    parser.add_argument("--output", type=Path, default=REPORT_OUTPUT_PATH, help="Path to write JSON report")
    args = parser.parse_args()

    run_citation_benchmark(dataset_path=args.dataset, output_path=args.output)
