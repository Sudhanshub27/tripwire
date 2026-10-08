"""
================================================================================
TRIPWIRE PIPELINE REPRODUCIBLE BENCHMARK RUNNER
================================================================================

Executes warm-up handling, multi-iteration testing, cold-start vs steady-state
separation, and exports structured scientific benchmark JSON reports.

Usage:
  python -m evaluation.benchmark [--mode mock|live] [--records N] [--iterations I]
                                 [--warmup-records W] [--output PATH]

Example:
  python -m evaluation.benchmark --mode mock --records 20 --iterations 3 --warmup-records 2 --output evaluation/reports/benchmark_report.json
================================================================================
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import structlog
from config.settings import settings
from src.llm.fallback_chain import FallbackChain, MockProvider
from src.resolution.entity_resolver import EntityResolver
from src.schemas.base import BaseRecord
from src.schemas.job import Job
from src.schemas.news import News
from src.schemas.product import Product
from src.schemas.research_paper import ResearchPaper
from src.schemas.startup import Startup
from src.vector.vector_store import VectorStoreManager, compute_dense_text_embedding

from evaluation.metrics import BenchmarkMetricsCollector

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "evaluation" / "datasets" / "sample_eval_dataset.json"
DEFAULT_OUTPUT_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "benchmark_report.json"

SCHEMA_MAP: Dict[str, Type[BaseRecord]] = {
    "Startup": Startup,
    "Product": Product,
    "ResearchPaper": ResearchPaper,
    "Job": Job,
    "News": News,
}


def load_evaluation_dataset(target_count: int) -> List[Dict[str, Any]]:
    """Load evaluation dataset from datasets/, cycling if target_count > dataset length."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Evaluation dataset file missing at {DATASET_FILE}")

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Evaluation dataset is empty")

    items: List[Dict[str, Any]] = []
    while len(items) < target_count:
        for item in data:
            items.append(item)
            if len(items) >= target_count:
                break
    return items


async def run_benchmark(
    mode: str = "mock",
    record_count: int = 20,
    iterations: int = 3,
    warmup_records: int = 2,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute reproducible scientific benchmark suite with warm-up and multi-iteration aggregation."""
    output_file = Path(output_path) if output_path else DEFAULT_OUTPUT_REPORT
    output_file.parent.mkdir(parents=True, exist_ok=True)

    collector = BenchmarkMetricsCollector(
        mode=mode,
        requested_records=record_count,
        iterations=iterations,
        warmup_records=warmup_records,
    )

    # Initialize components
    resolver = EntityResolver()
    vector_mgr = VectorStoreManager()

    if mode == "mock":
        fallback_chain = FallbackChain(providers=[MockProvider()])
    else:
        fallback_chain = FallbackChain()

    orig_mock_mode = getattr(settings, "MOCK_MODE", True)
    if mode == "mock":
        settings.MOCK_MODE = True

    try:
        # --- PHASE 1: Cold-Start Warm-up Handling ---
        logger.info("Initiating benchmark warm-up phase", warmup_records=warmup_records)
        t0_cold = time.perf_counter()

        if warmup_records > 0:
            warmup_dataset = load_evaluation_dataset(warmup_records)
            for w_item in warmup_dataset:
                w_schema = SCHEMA_MAP.get(w_item.get("target_schema", "Startup"), Startup)
                # Perform dummy extraction, resolution, and vector embedding calculation
                extracted, _ = await fallback_chain.extract_with_fallback(
                    text=w_item.get("html", "<div>Warmup</div>"),
                    schema=w_schema,
                )
                raw_name = w_item.get("raw_name", "Warmup Entity")
                _ = resolver.resolve(raw_name)
                _ = compute_dense_text_embedding(raw_name)

        cold_start_time_seconds = time.perf_counter() - t0_cold
        logger.info("Completed benchmark warm-up phase", cold_start_time_seconds=round(cold_start_time_seconds, 4))

        # --- PHASE 2: Steady-State Multi-Iteration Processing ---
        dataset = load_evaluation_dataset(record_count)
        t0_steady = time.perf_counter()
        processed_entities: List[Dict[str, Any]] = []

        for it in range(iterations):
            logger.info("Executing benchmark iteration", iteration=it + 1, total_iterations=iterations)

            for idx, item in enumerate(dataset):
                schema_name = item.get("target_schema", "Startup")
                schema_cls = SCHEMA_MAP.get(schema_name, Startup)
                html_payload = item.get("html", "<div>Sample HTML Entity Payload</div>")

                # 1. Scraper Ingestion Stage
                t0_scr = time.perf_counter()
                _ = len(html_payload)
                scr_latency_ms = round((time.perf_counter() - t0_scr) * 1000, 4)
                collector.record_scrape(latency_ms=scr_latency_ms, success=True)

                # 2. Multi-Tier LLM Extraction Stage
                t0_llm = time.perf_counter()
                extracted_dict, winning_provider = await fallback_chain.extract_with_fallback(
                    text=html_payload,
                    schema=schema_cls,
                )
                llm_latency_ms = round((time.perf_counter() - t0_llm) * 1000, 2)

                if extracted_dict and winning_provider != "FAILED_ALL_PROVIDERS":
                    if mode == "mock":
                        # Offline deterministic mock benchmark: MockProvider is the primary execution engine, not a failure fallback
                        is_fallback = False
                    else:
                        is_fallback = "rule" in winning_provider.lower() or winning_provider not in ("GeminiProvider", "Gemini", "Gemini-Flash")

                    collector.record_llm_call(
                        provider=winning_provider,
                        latency_ms=llm_latency_ms,
                        success=True,
                        is_fallback=is_fallback,
                    )

                    try:
                        validated = schema_cls.model_validate(extracted_dict)
                        collector.record_extraction(success=True, schema_valid=True, missing_fields=0)
                        record_dict = validated.model_dump(mode="json")
                    except Exception:
                        collector.record_extraction(success=False, schema_valid=False, missing_fields=1)
                        record_dict = extracted_dict
                else:
                    collector.record_llm_call(
                        provider=winning_provider,
                        latency_ms=llm_latency_ms,
                        success=False,
                        is_fallback=True,
                    )
                    collector.record_extraction(success=False, schema_valid=False, missing_fields=2)
                    record_dict = {"name": item.get("raw_name", f"Entity {idx}"), "recordType": "base"}

                # 3. Entity Resolution Stage
                raw_entity_name = item.get("raw_name") or record_dict.get("name") or record_dict.get("maker_company") or "Unknown Entity"
                canonical, method, conf = resolver.resolve(raw_entity_name)
                collector.record_resolution(method=method)

                if canonical:
                    record_dict["canonical_name"] = canonical
                    if "name" in record_dict:
                        record_dict["name"] = canonical

                if it == 0:
                    processed_entities.append(record_dict)

        steady_state_time_seconds = time.perf_counter() - t0_steady
        total_execution_time_seconds = cold_start_time_seconds + steady_state_time_seconds

        # --- PHASE 3: Vector Storage Indexing & Hybrid Search ---
        indexed_items = []
        for p_idx, entity in enumerate(processed_entities):
            title = entity.get("canonical_name") or entity.get("name") or entity.get("title") or f"Entity {p_idx}"
            rec_type = entity.get("recordType", "startup")
            text_body = f"{title} {entity.get('description', '')} {entity.get('stage', '')}".strip()
            indexed_items.append({
                "id": f"eval-{rec_type}-{p_idx}",
                "record_type": rec_type,
                "title": title,
                "text": text_body,
                "vector": compute_dense_text_embedding(text_body),
                "payload": entity,
            })

        vector_mgr.in_memory_index = indexed_items
        vector_mgr._is_indexed = True
        collector.set_vector_indexed_count(len(indexed_items))

        # Vector search queries
        test_queries = [
            "AI infrastructure and developer tools",
            "Low-latency voice agents",
            "Speech transcription batching",
            "Distributed Systems Engineer remote",
            "Reasoning open-source LLM model",
        ]

        for q in test_queries:
            t0_vec = time.perf_counter()
            _ = vector_mgr.search(query=q, limit=5)
            vec_latency_ms = round((time.perf_counter() - t0_vec) * 1000, 2)
            collector.record_vector_search(latency_ms=vec_latency_ms)

        # Record timings and generate final report
        collector.set_timings(
            cold_start_seconds=cold_start_time_seconds,
            steady_state_seconds=steady_state_time_seconds,
            total_seconds=total_execution_time_seconds,
        )

        report_data = collector.to_dict()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info("Saved reproducible benchmark JSON report", path=str(output_file))
        collector.print_terminal_summary()

        return report_data
    finally:
        settings.MOCK_MODE = orig_mock_mode


def main():
    """CLI entry point for python -m evaluation.benchmark."""
    parser = argparse.ArgumentParser(
        description="Tripwire Pipeline Scientific Reproducible Benchmark Harness"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Benchmark mode: 'mock' (offline zero-cost deterministic) or 'live'. Default: mock",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=20,
        help="Number of records per iteration (default: 20)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations (default: 3)",
    )
    parser.add_argument(
        "--warmup-records",
        type=int,
        default=2,
        help="Number of warm-up records to process before steady-state measurement (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_REPORT),
        help=f"Target output JSON report path (default: {DEFAULT_OUTPUT_REPORT})",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Also run the Phase 2 deterministic incremental benchmark suite",
    )

    args = parser.parse_args()

    if args.incremental:
        from evaluation.incremental_benchmark import run_incremental_benchmark

        inc_report_path = Path(args.output).parent / "incremental_benchmark.json"
        asyncio.run(
            run_incremental_benchmark(
                record_count=100,
                output_path=inc_report_path,
                mode=args.mode,
            )
        )

    asyncio.run(
        run_benchmark(
            mode=args.mode,
            record_count=args.records,
            iterations=args.iterations,
            warmup_records=args.warmup_records,
            output_path=Path(args.output),
        )
    )


if __name__ == "__main__":
    main()
