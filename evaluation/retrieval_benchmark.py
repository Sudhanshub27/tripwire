"""
================================================================================
GRAPHONE PIPELINE: HYBRID RETRIEVAL BENCHMARK & COMPARATIVE EVALUATION
================================================================================
Executes standardized comparative retrieval evaluation comparing:
1. Vector-Only Retrieval (Dense semantic similarity baseline)
2. Graph-Only Retrieval (Direct Knowledge Graph traversal baseline)
3. Hybrid Retrieval (Integrated Vector + Graph with deterministic scoring)

Measures and reports scientific IR metrics:
- Recall@K (K=1, 3, 5, 10)
- Precision@K (K=1, 3, 5, 10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Average Precision (MAP)
- Execution Latency (mean, p50, p95)
================================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.retrieval.models import HybridRetrievalRequest, RankingStrategy
from src.retrieval.service import HybridRetrievalService, hybrid_retrieval_service

from evaluation.metrics import (
    calculate_map,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_stats,
    get_git_commit,
)

logger = structlog.get_logger(__name__)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "datasets" / "retrieval_eval_dataset.json"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "reports" / "retrieval_benchmark_report.json"


def load_retrieval_eval_dataset(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the deterministic retrieval evaluation ground-truth dataset."""
    file_path = path or DATASET_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Retrieval evaluation dataset missing at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_retrieval_benchmark(
    dataset_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    service: Optional[HybridRetrievalService] = None,
) -> Dict[str, Any]:
    """
    Execute full comparative evaluation across Vector-only, Graph-only, and Hybrid retrieval.
    Computes Precision@K, Recall@K, MRR, NDCG@K, MAP, and latency distributions.
    """
    retrieval_svc = service or hybrid_retrieval_service
    retrieval_svc._ensure_initialized()

    dataset = load_retrieval_eval_dataset(dataset_path)
    output_file = output_path or REPORT_OUTPUT_PATH
    output_file.parent.mkdir(parents=True, exist_ok=True)

    k_values = [1, 3, 5, 10]

    # Metrics accumulators for each paradigm
    modes = ["vector_only", "graph_only", "hybrid"]
    metrics_by_mode: Dict[str, Dict[str, List[float]]] = {
        m: {
            "latencies_ms": [],
            "mrr": [],
            "map": [],
            **{f"precision@{k}": [] for k in k_values},
            **{f"recall@{k}": [] for k in k_values},
            **{f"ndcg@{k}": [] for k in k_values},
        }
        for m in modes
    }

    per_query_details: List[Dict[str, Any]] = []

    for item in dataset:
        qid = item["query_id"]
        qtext = item["query"]
        ctype = item.get("candidate_type", "all")
        seed_entity = item.get("seed_entity", "")
        expected_ids: Set[str] = set(item.get("expected_relevant_ids", []))
        graded_rel: Dict[str, float] = item.get("graded_relevance", {})

        query_record: Dict[str, Any] = {
            "query_id": qid,
            "query": qtext,
            "candidate_type": ctype,
            "expected_count": len(expected_ids),
            "results": {},
        }

        # ----------------------------------------------------------------------
        # 1. Evaluate Vector-Only Retrieval
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        vec_cands = retrieval_svc.search_vector_only(query=qtext, candidate_type=ctype, limit=10)
        t_vec_ms = (time.perf_counter() - t0) * 1000.0
        vec_ids = [c.id for c in vec_cands]

        metrics_by_mode["vector_only"]["latencies_ms"].append(t_vec_ms)
        metrics_by_mode["vector_only"]["mrr"].append(calculate_mrr(vec_ids, expected_ids))
        metrics_by_mode["vector_only"]["map"].append(calculate_map(vec_ids, expected_ids, k=5))
        for k in k_values:
            metrics_by_mode["vector_only"][f"precision@{k}"].append(calculate_precision_at_k(vec_ids, expected_ids, k))
            metrics_by_mode["vector_only"][f"recall@{k}"].append(calculate_recall_at_k(vec_ids, expected_ids, k))
            metrics_by_mode["vector_only"][f"ndcg@{k}"].append(calculate_ndcg_at_k(vec_ids, graded_rel, k))

        query_record["results"]["vector_only"] = {
            "retrieved_ids": vec_ids,
            "latency_ms": round(t_vec_ms, 3),
            "recall@5": round(calculate_recall_at_k(vec_ids, expected_ids, 5), 4),
            "precision@5": round(calculate_precision_at_k(vec_ids, expected_ids, 5), 4),
            "mrr": round(calculate_mrr(vec_ids, expected_ids), 4),
            "ndcg@5": round(calculate_ndcg_at_k(vec_ids, graded_rel, 5), 4),
        }

        # ----------------------------------------------------------------------
        # 2. Evaluate Graph-Only Retrieval
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        graph_cands = retrieval_svc.search_graph_only(
            seed_query=seed_entity if seed_entity else qtext,
            depth=2,
            limit=10,
        )
        t_graph_ms = (time.perf_counter() - t0) * 1000.0
        graph_ids = [c.id for c in graph_cands]

        metrics_by_mode["graph_only"]["latencies_ms"].append(t_graph_ms)
        metrics_by_mode["graph_only"]["mrr"].append(calculate_mrr(graph_ids, expected_ids))
        metrics_by_mode["graph_only"]["map"].append(calculate_map(graph_ids, expected_ids, k=5))
        for k in k_values:
            metrics_by_mode["graph_only"][f"precision@{k}"].append(calculate_precision_at_k(graph_ids, expected_ids, k))
            metrics_by_mode["graph_only"][f"recall@{k}"].append(calculate_recall_at_k(graph_ids, expected_ids, k))
            metrics_by_mode["graph_only"][f"ndcg@{k}"].append(calculate_ndcg_at_k(graph_ids, graded_rel, k))

        query_record["results"]["graph_only"] = {
            "retrieved_ids": graph_ids,
            "latency_ms": round(t_graph_ms, 3),
            "recall@5": round(calculate_recall_at_k(graph_ids, expected_ids, 5), 4),
            "precision@5": round(calculate_precision_at_k(graph_ids, expected_ids, 5), 4),
            "mrr": round(calculate_mrr(graph_ids, expected_ids), 4),
            "ndcg@5": round(calculate_ndcg_at_k(graph_ids, graded_rel, 5), 4),
        }

        # ----------------------------------------------------------------------
        # 3. Evaluate Hybrid Retrieval (Vector + Graph)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        req = HybridRetrievalRequest(
            query=qtext,
            candidate_type=ctype,
            vector_limit=10,
            graph_depth=2,
            max_graph_candidates=20,
            vector_weight=0.6,
            graph_weight=0.4,
            ranking_strategy=RankingStrategy.WEIGHTED,
            limit=10,
        )
        hybrid_resp = retrieval_svc.search_hybrid(req)
        t_hybrid_ms = (time.perf_counter() - t0) * 1000.0
        hybrid_ids = [c.id for c in hybrid_resp.results]

        metrics_by_mode["hybrid"]["latencies_ms"].append(t_hybrid_ms)
        metrics_by_mode["hybrid"]["mrr"].append(calculate_mrr(hybrid_ids, expected_ids))
        metrics_by_mode["hybrid"]["map"].append(calculate_map(hybrid_ids, expected_ids, k=5))
        for k in k_values:
            metrics_by_mode["hybrid"][f"precision@{k}"].append(calculate_precision_at_k(hybrid_ids, expected_ids, k))
            metrics_by_mode["hybrid"][f"recall@{k}"].append(calculate_recall_at_k(hybrid_ids, expected_ids, k))
            metrics_by_mode["hybrid"][f"ndcg@{k}"].append(calculate_ndcg_at_k(hybrid_ids, graded_rel, k))

        query_record["results"]["hybrid"] = {
            "retrieved_ids": hybrid_ids,
            "latency_ms": round(t_hybrid_ms, 3),
            "recall@5": round(calculate_recall_at_k(hybrid_ids, expected_ids, 5), 4),
            "precision@5": round(calculate_precision_at_k(hybrid_ids, expected_ids, 5), 4),
            "mrr": round(calculate_mrr(hybrid_ids, expected_ids), 4),
            "ndcg@5": round(calculate_ndcg_at_k(hybrid_ids, graded_rel, 5), 4),
            "vector_count": hybrid_resp.vector_candidates_count,
            "graph_count": hybrid_resp.graph_candidates_count,
            "merged_count": hybrid_resp.merged_candidates_count,
        }

        per_query_details.append(query_record)

    # Aggregate metric summaries across queries
    summary_report: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": get_git_commit(),
            "query_count": len(dataset),
            "evaluation_dataset": str(DATASET_PATH),
        },
        "metrics_summary": {},
        "per_query_details": per_query_details,
    }

    for mode in modes:
        mode_data = metrics_by_mode[mode]
        lat_stats = calculate_stats(mode_data["latencies_ms"])

        summary_report["metrics_summary"][mode] = {
            "latency_stats_ms": lat_stats,
            "mrr": round(sum(mode_data["mrr"]) / len(mode_data["mrr"]), 4),
            "map@5": round(sum(mode_data["map"]) / len(mode_data["map"]), 4),
            **{
                f"recall@{k}": round(sum(mode_data[f"recall@{k}"]) / len(mode_data[f"recall@{k}"]), 4)
                for k in k_values
            },
            **{
                f"precision@{k}": round(sum(mode_data[f"precision@{k}"]) / len(mode_data[f"precision@{k}"]), 4)
                for k in k_values
            },
            **{
                f"ndcg@{k}": round(sum(mode_data[f"ndcg@{k}"]) / len(mode_data[f"ndcg@{k}"]), 4)
                for k in k_values
            },
        }

    # Save structured JSON benchmark report
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info("Retrieval benchmark complete", output=str(output_file))
    print_benchmark_summary_tables(summary_report)

    return summary_report


def print_benchmark_summary_tables(report: Dict[str, Any]) -> None:
    """Render terminal report comparing Vector, Graph, and Hybrid retrieval metrics."""
    console.print("\n")
    console.print(
        Panel(
            "[bold cyan]PHASE 5: GRAPH + VECTOR HYBRID RETRIEVAL BENCHMARK RESULTS[/bold cyan]\n"
            f"Queries Evaluated: {report['metadata']['query_count']} | Git Commit: {report['metadata']['git_commit']}",
            border_style="cyan",
        )
    )

    summary = report["metrics_summary"]

    table = Table(title="Retrieval Quality Comparison (Precision, Recall, MRR, NDCG)", border_style="blue")
    table.add_column("Retrieval Approach", style="bold white", justify="left")
    table.add_column("Recall@1", justify="center")
    table.add_column("Recall@5", justify="center")
    table.add_column("Precision@5", justify="center")
    table.add_column("MRR", justify="center")
    table.add_column("NDCG@5", justify="center")
    table.add_column("MAP@5", justify="center")

    for mode_key, label in [
        ("vector_only", "Vector-Only (Dense Embedding)"),
        ("graph_only", "Graph-Only (BFS Traversal)"),
        ("hybrid", "Hybrid (Vector + Graph Fusion)"),
    ]:
        m = summary[mode_key]
        is_hybrid = mode_key == "hybrid"
        color = "bold green" if is_hybrid else "white"
        table.add_row(
            f"[{color}]{label}[/{color}]",
            f"[{color}]{m['recall@1'] * 100:.1f}%[/{color}]",
            f"[{color}]{m['recall@5'] * 100:.1f}%[/{color}]",
            f"[{color}]{m['precision@5'] * 100:.1f}%[/{color}]",
            f"[{color}]{m['mrr']:.4f}[/{color}]",
            f"[{color}]{m['ndcg@5']:.4f}[/{color}]",
            f"[{color}]{m['map@5']:.4f}[/{color}]",
        )

    console.print(table)

    # Latency comparison table
    lat_table = Table(title="Execution Latency Performance (Milliseconds)", border_style="green")
    lat_table.add_column("Approach", style="bold white")
    lat_table.add_column("Mean (ms)", justify="right")
    lat_table.add_column("p50 (ms)", justify="right")
    lat_table.add_column("p95 (ms)", justify="right")
    lat_table.add_column("Min (ms)", justify="right")
    lat_table.add_column("Max (ms)", justify="right")

    for mode_key, label in [
        ("vector_only", "Vector-Only"),
        ("graph_only", "Graph-Only"),
        ("hybrid", "Hybrid (Vector + Graph)"),
    ]:
        stats = summary[mode_key]["latency_stats_ms"]
        lat_table.add_row(
            label,
            f"{stats['mean']:.3f}",
            f"{stats['p50']:.3f}",
            f"{stats['p95']:.3f}",
            f"{stats['min']:.3f}",
            f"{stats['max']:.3f}",
        )

    console.print(lat_table)
    console.print(f"[dim]Detailed benchmark report written to: {REPORT_OUTPUT_PATH}[/dim]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Graph + Vector Hybrid Retrieval Benchmark")
    parser.add_argument("--dataset", type=Path, default=None, help="Custom evaluation dataset path")
    parser.add_argument("--output", type=Path, default=None, help="Output benchmark report JSON path")
    args = parser.parse_args()

    run_retrieval_benchmark(dataset_path=args.dataset, output_path=args.output)
