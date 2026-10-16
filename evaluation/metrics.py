"""
================================================================================
TRIPWIRE PIPELINE EVALUATION METRICS MODULE
================================================================================

Provides scientific metric calculations, statistical percentiles (mean, median,
p50, p95, min, max), rates (success, failure, fallback, validation, deduplication),
reproducibility metadata (git commit, settings, environment), and Rich reporting.
================================================================================
"""

import math
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from config.settings import settings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.observability.metrics import metrics_collector

logger = structlog.get_logger(__name__)
console = Console()


def get_git_commit() -> str:
    """Safely retrieve short git commit hash if repository context is available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive aggregate statistics over a list of numeric values:
    mean, median, p50, p95, minimum, and maximum.
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    # Calculate median / p50
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    # Calculate p95 percentile with linear interpolation
    if n == 1:
        p95_val = sorted_vals[0]
    else:
        k = (n - 1) * 0.95
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            p95_val = sorted_vals[int(k)]
        else:
            p95_val = sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

    return {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "p50": round(median_val, 2),
        "p95": round(p95_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
    }


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: Any, k: int) -> float:
    """Calculate Precision@K: fraction of top-K retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    hits = sum(1 for item in retrieved_ids[:k] if item in rel_set)
    return round(hits / k, 4)


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Any, k: int) -> float:
    """Calculate Recall@K: fraction of relevant items retrieved in top-K."""
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0
    if k <= 0:
        return 0.0
    hits = sum(1 for item in retrieved_ids[:k] if item in rel_set)
    return round(hits / len(rel_set), 4)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: Any) -> float:
    """Calculate Mean Reciprocal Rank (MRR): reciprocal of the 1-based rank of the first relevant hit."""
    rel_set = set(relevant_ids)
    for idx, item in enumerate(retrieved_ids, start=1):
        if item in rel_set:
            return round(1.0 / idx, 4)
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (NDCG@K) supporting graded relevance.
    Formula: DCG@K / IDCG@K where DCG = sum((2^rel - 1) / log2(i + 2)).
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        rel = float(relevance_scores.get(item_id, 0.0))
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)

    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(ideal_rels) if rel > 0)
    if idcg <= 0.0:
        return 0.0

    return round(dcg / idcg, 4)


def calculate_map(retrieved_ids: List[str], relevant_ids: Any, k: int) -> float:
    """Calculate Mean Average Precision at K (MAP@K)."""
    rel_set = set(relevant_ids)
    if not rel_set or k <= 0:
        return 0.0

    precisions: List[float] = []
    hits = 0
    for idx, item in enumerate(retrieved_ids[:k], start=1):
        if item in rel_set:
            hits += 1
            precisions.append(hits / idx)

    if not precisions:
        return 0.0

    return round(sum(precisions) / min(len(rel_set), k), 4)


# ==============================================================================
# PHASE 6: CITATION-AWARE RAG EVALUATION METRICS
# ==============================================================================


def calculate_citation_reference_validity(valid_citations: int, total_citations: int) -> float:
    """
    Calculate Citation Reference Validity rate: fraction of citation markers that resolve
    to actual retrieved document chunks.
    Denominator: total citation markers in response.
    Returns 1.0 if total_citations == 0 (e.g. valid abstention).
    """
    if total_citations <= 0:
        return 1.0
    return round(min(1.0, max(0.0, valid_citations / total_citations)), 4)


def calculate_claim_citation_coverage(cited_claims: int, total_claims: int) -> float:
    """
    Calculate Claim-Level Citation Coverage: fraction of factual claims supported by at least one citation.
    Denominator: total factual claims in response.
    Returns 1.0 if total_claims == 0 (e.g. valid abstention).
    """
    if total_claims <= 0:
        return 1.0
    return round(min(1.0, max(0.0, cited_claims / total_claims)), 4)


def calculate_citation_faithfulness(supported_citations: int, valid_citations: int) -> float:
    """
    Calculate Citation Faithfulness / Evidence Support: fraction of valid citations whose
    cited chunk semantically entails and supports the claim.
    Denominator: total valid citation references evaluated.
    Returns 1.0 if valid_citations == 0 (e.g. valid abstention).
    """
    if valid_citations <= 0:
        return 1.0
    return round(min(1.0, max(0.0, supported_citations / valid_citations)), 4)


def calculate_unsupported_claim_rate(unsupported_claims: int, total_claims: int) -> float:
    """
    Calculate Unsupported Claim Rate: fraction of factual claims lacking verifiable source support.
    Denominator: total factual claims evaluated.
    Returns 0.0 if total_claims == 0 (e.g. valid abstention).
    """
    if total_claims <= 0:
        return 0.0
    return round(min(1.0, max(0.0, unsupported_claims / total_claims)), 4)


def calculate_abstention_accuracy(correct_decisions: int, total_queries: int) -> float:
    """
    Calculate Abstention Accuracy: fraction of queries correctly answered when evidence is sufficient
    or correctly abstained from when evidence is insufficient.
    Denominator: total evaluated queries (supported + unsupported).
    """
    if total_queries <= 0:
        return 1.0
    return round(min(1.0, max(0.0, correct_decisions / total_queries)), 4)


class BenchmarkMetricsCollector:
    """
    Scientific Evaluation Metrics Collector tracking iterations, warm-up,
    cold-start time, steady-state runtime, aggregate stats, and key rates.
    """

    def __init__(
        self,
        mode: str = "mock",
        requested_records: int = 20,
        iterations: int = 1,
        warmup_records: int = 2,
    ):
        self.mode = mode
        self.requested_records = requested_records
        self.iterations = iterations
        self.warmup_records = warmup_records
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.git_commit = get_git_commit()
        self.app_mock_mode = getattr(settings, "MOCK_MODE", False)

        # Environmental & Concurrency Configuration state
        # In offline mock mode, the benchmark harness operates in mock mode (MOCK_MODE=True)
        # while application_mock_mode accurately records the underlying host/app settings.MOCK_MODE.
        effective_mock_mode = True if mode == "mock" else self.app_mock_mode
        self.configuration = {
            "MAX_CONCURRENT_SCRAPES": getattr(settings, "MAX_CONCURRENT_SCRAPES", 5),
            "SCRAPER_TIMEOUT_SECONDS": getattr(settings, "SCRAPER_TIMEOUT_SECONDS", 15.0),
            "RATE_LIMIT_TOKENS_PER_SEC": getattr(settings, "RATE_LIMIT_TOKENS_PER_SEC", 10.0),
            "MOCK_MODE": effective_mock_mode,
            "APPLICATION_MOCK_MODE": self.app_mock_mode,
            "BENCHMARK_MODE": mode,
            "DATA_PROCESSED_DIR": str(settings.DATA_PROCESSED_DIR),
        }

        # Timings (seconds)
        self.cold_start_time_seconds: float = 0.0
        self.steady_state_time_seconds: float = 0.0
        self.total_execution_time_seconds: float = 0.0

        # Scraper latencies & counts (ms)
        self.scraper_latencies_ms: List[float] = []
        self.scraper_success_count: int = 0
        self.scraper_fail_count: int = 0

        # LLM latencies & counts (ms)
        self.llm_latencies_ms: List[float] = []
        self.llm_success_count: int = 0
        self.llm_fail_count: int = 0
        self.llm_fallback_count: int = 0
        if self.mode == "mock":
            self.provider_counts: Dict[str, int] = {"Mock": 0}
        else:
            self.provider_counts: Dict[str, int] = {
                "Gemini": 0,
                "Groq": 0,
                "DeepSeek": 0,
                "RuleBased": 0,
            }

        # Extraction Quality counts
        self.extraction_success_count: int = 0
        self.schema_validation_failures: int = 0
        self.missing_invalid_records: int = 0

        # Entity Resolution counts
        self.er_total_evaluated: int = 0
        self.er_exact_matches: int = 0
        self.er_normalized_matches: int = 0
        self.er_fuzzy_matches: int = 0
        self.er_unresolved: int = 0

        # Vector Search latencies & index count (ms)
        self.vector_search_latencies_ms: List[float] = []
        self.indexed_records_count: int = 0

    def record_scrape(self, latency_ms: float, success: bool = True) -> None:
        """Record scraper execution latency and outcome."""
        self.scraper_latencies_ms.append(latency_ms)
        if success:
            self.scraper_success_count += 1
        else:
            self.scraper_fail_count += 1

    def record_llm_call(
        self,
        provider: str,
        latency_ms: float,
        success: bool = True,
        is_fallback: bool = False,
        token_count: Optional[int] = None,
    ) -> None:
        """Record LLM call telemetry, provider tier, and fallback status."""
        self.llm_latencies_ms.append(latency_ms)
        if success:
            self.llm_success_count += 1
        else:
            self.llm_fail_count += 1

        if is_fallback:
            self.llm_fallback_count += 1

        if "mock" in provider.lower():
            p_clean = "Mock"
        elif "rule" in provider.lower():
            p_clean = "RuleBased"
        elif "gemini" in provider.lower():
            p_clean = "Gemini"
        elif "groq" in provider.lower():
            p_clean = "Groq"
        elif "deepseek" in provider.lower():
            p_clean = "DeepSeek"
        else:
            p_clean = provider

        if p_clean not in self.provider_counts:
            self.provider_counts[p_clean] = 0
        self.provider_counts[p_clean] += 1

        if token_count is not None:
            metrics_collector.record_llm_call(
                provider_family=p_clean,
                latency_seconds=latency_ms / 1000.0,
                token_count=token_count,
            )

    def record_extraction(
        self,
        success: bool,
        schema_valid: bool = True,
        missing_fields: int = 0,
    ) -> None:
        """Record schema extraction outcome and Pydantic validation status."""
        if success and schema_valid:
            self.extraction_success_count += 1
        else:
            if not schema_valid:
                self.schema_validation_failures += 1
            if missing_fields > 0 or not success:
                self.missing_invalid_records += 1

    def record_resolution(self, method: str) -> None:
        """Record entity resolution match method."""
        self.er_total_evaluated += 1
        m = method.lower()
        if m == "exact":
            self.er_exact_matches += 1
        elif m == "normalized":
            self.er_normalized_matches += 1
        elif m == "fuzzy":
            self.er_fuzzy_matches += 1
        else:
            self.er_unresolved += 1

        metrics_collector.record_er_merge(method=m)

    def record_vector_search(self, latency_ms: float) -> None:
        """Record vector search query latency."""
        self.vector_search_latencies_ms.append(latency_ms)

    def set_vector_indexed_count(self, count: int) -> None:
        """Set indexed record count in LanceDB vector index."""
        self.indexed_records_count = count

    def set_timings(
        self,
        cold_start_seconds: float,
        steady_state_seconds: float,
        total_seconds: float,
    ) -> None:
        """Set cold-start, steady-state, and total execution wall-clock timings."""
        self.cold_start_time_seconds = round(cold_start_seconds, 4)
        self.steady_state_time_seconds = round(steady_state_seconds, 4)
        self.total_execution_time_seconds = round(total_seconds, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Generate scientific machine-readable benchmark JSON dictionary."""
        # Baseline counts
        records_processed = self.extraction_success_count + self.missing_invalid_records
        successful_records = self.extraction_success_count
        failed_records = self.missing_invalid_records

        # Calculate Rates (0.0 to 1.0 float ranges & percentages)
        eff_time = self.steady_state_time_seconds if self.steady_state_time_seconds > 0 else (self.total_execution_time_seconds or 0.001)
        rec_per_sec = round(records_processed / eff_time, 2) if eff_time > 0 else 0.0

        success_rate = round(successful_records / records_processed, 4) if records_processed > 0 else 0.0
        failure_rate = round(failed_records / records_processed, 4) if records_processed > 0 else 0.0

        tot_llm_calls = self.llm_success_count + self.llm_fail_count
        llm_fallback_rate = round(self.llm_fallback_count / tot_llm_calls, 4) if tot_llm_calls > 0 else 0.0
        schema_valid_rate = round(self.extraction_success_count / records_processed, 4) if records_processed > 0 else 0.0

        merged_count = self.er_exact_matches + self.er_normalized_matches + self.er_fuzzy_matches
        dup_detection_rate = round(merged_count / self.er_total_evaluated, 4) if self.er_total_evaluated > 0 else 0.0

        # Calculate provider percentage breakdown
        prov_breakdown = {}
        for prov, cnt in self.provider_counts.items():
            if self.mode == "mock" and cnt == 0 and len(self.provider_counts) > 1:
                continue
            pct = round((cnt / tot_llm_calls) * 100, 1) if tot_llm_calls > 0 else 0.0
            prov_breakdown[prov] = {"count": cnt, "percentage": pct}

        # Calculate aggregate statistics
        scr_stats = calculate_stats(self.scraper_latencies_ms)
        llm_stats = calculate_stats(self.llm_latencies_ms)
        vec_stats = calculate_stats(self.vector_search_latencies_ms)

        return {
            "metadata": {
                "timestamp": self.timestamp,
                "git_commit": self.git_commit,
                "mode": self.mode,
                "benchmark_mode": self.mode,
                "application_mock_mode": self.app_mock_mode,
                "iterations": self.iterations,
                "records": self.requested_records,
                "requested_records": self.requested_records,
                "warmup": self.warmup_records,
                "warmup_records": self.warmup_records,
                "configuration": self.configuration,
            },
            "pipeline": {
                "records_processed": records_processed,
                "successful_records": successful_records,
                "failed_records": failed_records,
                "records_per_second": rec_per_sec,
                "cold_start_time_seconds": self.cold_start_time_seconds,
                "steady_state_time_seconds": self.steady_state_time_seconds,
                "total_execution_time_seconds": self.total_execution_time_seconds,
            },
            "rates": {
                "success_rate": success_rate,
                "failure_rate": failure_rate,
                "llm_fallback_rate": llm_fallback_rate,
                "schema_validation_success_rate": schema_valid_rate,
                "duplicate_detection_rate": dup_detection_rate,
            },
            "scraper": {
                "scrape_count": self.scraper_success_count + self.scraper_fail_count,
                "successful_scrapes": self.scraper_success_count,
                "failed_scrapes": self.scraper_fail_count,
                "stats_ms": scr_stats,
            },
            "llm": {
                "total_calls": tot_llm_calls,
                "successful_calls": self.llm_success_count,
                "failed_calls": self.llm_fail_count,
                "fallback_calls": self.llm_fallback_count,
                "fallback_rate": llm_fallback_rate,
                "stats_ms": llm_stats,
                "provider_breakdown": prov_breakdown,
            },
            "extraction_quality": {
                "successfully_parsed": self.extraction_success_count,
                "schema_validation_failures": self.schema_validation_failures,
                "missing_invalid_records": self.missing_invalid_records,
                "validation_success_rate": schema_valid_rate,
            },
            "resolution": {
                "total_evaluated": self.er_total_evaluated,
                "exact_matches": self.er_exact_matches,
                "normalized_matches": self.er_normalized_matches,
                "fuzzy_matches": self.er_fuzzy_matches,
                "unresolved": self.er_unresolved,
                "duplicate_rate": dup_detection_rate,
            },
            "vector_search": {
                "stats_ms": vec_stats,
                "indexed_records": self.indexed_records_count,
            },
        }

    def print_terminal_summary(self) -> None:
        """Render formatted Rich terminal report table."""
        data = self.to_dict()
        p = data["pipeline"]
        rates_data = data["rates"]
        scraper_stats = data["scraper"]["stats_ms"]
        llm_stats = data["llm"]["stats_ms"]
        vector_stats = data["vector_search"]["stats_ms"]

        title = f"[bold cyan]TRIPWIRE SCIENTIFIC EVALUATION BENCHMARK[/bold cyan] (Git: [yellow]{self.git_commit}[/yellow] | Mode: [green]{self.mode.upper()}[/green])"
        console.print("\n")
        console.print(
            Panel(
                f"Iterations: [bold white]{self.iterations}[/bold white] | Warm-up: [bold white]{self.warmup_records} rec[/bold white] | "
                f"Cold Start: [bold yellow]{p['cold_start_time_seconds']}s[/bold yellow] | "
                f"Steady State: [bold green]{p['steady_state_time_seconds']}s[/bold green] | "
                f"Throughput: [bold green]{p['records_per_second']} rec/sec[/bold green]",
                title=title,
                border_style="cyan",
            )
        )

        table = Table(title="Scientific Benchmark Metrics Summary", show_header=True, header_style="bold magenta")
        table.add_column("Stage / Dimension", style="cyan", width=24)
        table.add_column("Metric Name", style="white")
        table.add_column("Value / Aggregate Stats", style="bold green", justify="right")

        # System Rates
        table.add_row("1. Pipeline Rates", "Overall Record Success Rate", f"{round(rates_data['success_rate'] * 100, 2)}%")
        table.add_row("", "Pipeline Failure Rate", f"{round(rates_data['failure_rate'] * 100, 2)}%")
        table.add_row("", "LLM Fallback Rate", f"{round(rates_data['llm_fallback_rate'] * 100, 2)}%")
        table.add_row("", "Schema Validation Success Rate", f"{round(rates_data['schema_validation_success_rate'] * 100, 2)}%")
        table.add_row("", "Duplicate Detection Rate", f"{round(rates_data['duplicate_detection_rate'] * 100, 2)}%")

        # Scraper Stats
        table.add_row("2. Scraper Latency (ms)", "Mean / Median / p95", f"{scraper_stats['mean']}ms / {scraper_stats['median']}ms / {scraper_stats['p95']}ms")
        table.add_row("", "Min / Max Range", f"[{scraper_stats['min']}ms, {scraper_stats['max']}ms]")

        # LLM Stats
        table.add_row("3. LLM Latency (ms)", "Mean / Median / p95", f"{llm_stats['mean']}ms / {llm_stats['median']}ms / {llm_stats['p95']}ms")
        table.add_row("", "Min / Max Range", f"[{llm_stats['min']}ms, {llm_stats['max']}ms]")
        prov_summary = ", ".join(f"{k}: {v['count']} ({v['percentage']}%)" for k, v in data["llm"]["provider_breakdown"].items())
        table.add_row("", "Provider Breakdown", prov_summary)

        # Vector Stats
        table.add_row("4. Vector Latency (ms)", "Mean / Median / p95", f"{vector_stats['mean']}ms / {vector_stats['median']}ms / {vector_stats['p95']}ms")
        table.add_row("", "Indexed Record Count", str(data["vector_search"]["indexed_records"]))

        console.print(table)
        console.print("\n")
