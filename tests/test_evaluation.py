"""
================================================================================
TEST SUITE FOR SCIENTIFIC EVALUATION & BENCHMARKING FRAMEWORK
================================================================================
Tests statistical aggregations (mean, median, p50, p95, min, max), rates,
git commit metadata extraction, warm-up handling, multi-iteration execution,
and baseline vs new benchmark comparison.
================================================================================
"""

import json
from pathlib import Path

import pytest
from config.settings import settings
from evaluation.benchmark import run_benchmark
from evaluation.compare import compare_reports
from evaluation.incremental_benchmark import run_incremental_benchmark
from evaluation.metrics import (
    BenchmarkMetricsCollector,
    calculate_stats,
    get_git_commit,
)


def test_calculate_stats_empty_and_single():
    """Test calculate_stats with empty list and single item."""
    res_empty = calculate_stats([])
    assert res_empty["mean"] == 0.0
    assert res_empty["median"] == 0.0
    assert res_empty["p50"] == 0.0
    assert res_empty["p95"] == 0.0
    assert res_empty["min"] == 0.0
    assert res_empty["max"] == 0.0

    res_single = calculate_stats([42.5])
    assert res_single["mean"] == 42.5
    assert res_single["median"] == 42.5
    assert res_single["min"] == 42.5
    assert res_single["max"] == 42.5


def test_calculate_stats_distribution():
    """Test calculate_stats with numeric range [10..100]."""
    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    res = calculate_stats(vals)
    assert res["mean"] == 55.0
    assert res["median"] == 55.0
    assert res["min"] == 10.0
    assert res["max"] == 100.0
    assert res["p95"] >= 90.0


def test_get_git_commit():
    """Test get_git_commit helper returns a valid string."""
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_benchmark_metrics_collector_rates_and_stats():
    """Test BenchmarkMetricsCollector calculates rates and aggregate stats correctly."""
    collector = BenchmarkMetricsCollector(
        mode="mock",
        requested_records=10,
        iterations=2,
        warmup_records=2,
    )

    collector.record_scrape(latency_ms=10.0, success=True)
    collector.record_scrape(latency_ms=30.0, success=True)

    collector.record_llm_call(provider="RuleBased", latency_ms=5.0, success=True, is_fallback=True)
    collector.record_extraction(success=True, schema_valid=True, missing_fields=0)

    collector.record_resolution("exact")
    collector.record_resolution("normalized")
    collector.record_resolution("fuzzy")
    collector.record_resolution("unresolved")

    collector.record_vector_search(latency_ms=2.0)
    collector.set_vector_indexed_count(10)
    collector.set_timings(cold_start_seconds=0.1, steady_state_seconds=0.4, total_seconds=0.5)

    data = collector.to_dict()

    assert data["metadata"]["git_commit"] == collector.git_commit
    assert data["metadata"]["iterations"] == 2
    assert data["metadata"]["warmup_records"] == 2
    assert data["pipeline"]["cold_start_time_seconds"] == 0.1
    assert data["pipeline"]["steady_state_time_seconds"] == 0.4

    # Rates assertions
    rates = data["rates"]
    assert rates["success_rate"] == 1.0
    assert rates["failure_rate"] == 0.0
    assert rates["llm_fallback_rate"] == 1.0
    assert rates["duplicate_detection_rate"] == 0.75

    # Aggregate stats assertions
    scr_stats = data["scraper"]["stats_ms"]
    assert scr_stats["mean"] == 20.0
    assert scr_stats["min"] == 10.0
    assert scr_stats["max"] == 30.0


@pytest.mark.asyncio
async def test_run_benchmark_mock_iterations(tmp_path: Path):
    """Test multi-iteration benchmark run in mock mode."""
    target_report = tmp_path / "benchmark_scientific_report.json"
    report = await run_benchmark(
        mode="mock",
        record_count=4,
        iterations=2,
        warmup_records=2,
        output_path=target_report,
    )

    assert target_report.exists()
    assert report["metadata"]["mode"] == "mock"
    assert report["metadata"]["iterations"] == 2
    assert report["metadata"]["warmup_records"] == 2
    assert "cold_start_time_seconds" in report["pipeline"]
    assert "steady_state_time_seconds" in report["pipeline"]
    assert "rates" in report
    assert "stats_ms" in report["scraper"]
    assert "stats_ms" in report["llm"]
    assert "stats_ms" in report["vector_search"]


def test_compare_reports_cli(tmp_path: Path):
    """Test comparing baseline vs new benchmark JSON reports."""
    base_file = tmp_path / "baseline.json"
    new_file = tmp_path / "new.json"
    diff_file = tmp_path / "diff.json"

    base_data = {
        "metadata": {"timestamp": "2026-09-04T00:00:00Z", "git_commit": "abc1234"},
        "pipeline": {"records_per_second": 10.0, "cold_start_time_seconds": 0.5, "steady_state_time_seconds": 2.0},
        "rates": {
            "success_rate": 0.90,
            "failure_rate": 0.10,
            "llm_fallback_rate": 0.20,
            "schema_validation_success_rate": 0.90,
            "duplicate_detection_rate": 0.80,
        },
        "scraper": {"stats_ms": {"mean": 20.0, "p95": 30.0}},
        "llm": {"stats_ms": {"mean": 50.0, "p95": 80.0}},
        "vector_search": {"stats_ms": {"mean": 3.0, "p95": 5.0}},
    }

    new_data = {
        "metadata": {"timestamp": "2026-09-04T00:10:00Z", "git_commit": "def5678"},
        "pipeline": {"records_per_second": 15.0, "cold_start_time_seconds": 0.4, "steady_state_time_seconds": 1.5},
        "rates": {
            "success_rate": 0.95,
            "failure_rate": 0.05,
            "llm_fallback_rate": 0.10,
            "schema_validation_success_rate": 0.95,
            "duplicate_detection_rate": 0.85,
        },
        "scraper": {"stats_ms": {"mean": 15.0, "p95": 25.0}},
        "llm": {"stats_ms": {"mean": 40.0, "p95": 60.0}},
        "vector_search": {"stats_ms": {"mean": 2.5, "p95": 4.0}},
    }

    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(base_data, f)
    with open(new_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f)

    comp = compare_reports(base_file, new_file, output_path=diff_file)

    assert diff_file.exists()
    deltas = comp["deltas"]
    assert deltas["throughput_rec_per_sec"]["baseline"] == 10.0
    assert deltas["throughput_rec_per_sec"]["new"] == 15.0
    assert deltas["throughput_rec_per_sec"]["delta"] == 5.0
    assert deltas["throughput_rec_per_sec"]["pct_change"] == 50.0

    assert deltas["success_rate"]["delta"] == 0.05
    assert deltas["llm_fallback_rate"]["delta"] == -0.10


@pytest.mark.asyncio
async def test_mock_metadata_consistency(tmp_path: Path):
    """Test that mock benchmark metadata consistently records modes, counts, and configuration."""
    output_report = tmp_path / "mock_meta_report.json"
    report = await run_benchmark(
        mode="mock",
        record_count=10,
        iterations=2,
        warmup_records=2,
        output_path=output_report,
    )

    meta = report["metadata"]
    assert meta["mode"] == "mock"
    assert meta["benchmark_mode"] == "mock"
    assert meta["application_mock_mode"] == getattr(settings, "MOCK_MODE", False)
    assert meta["records"] == 10
    assert meta["requested_records"] == 10
    assert meta["iterations"] == 2
    assert meta["warmup"] == 2
    assert meta["warmup_records"] == 2
    assert isinstance(meta["git_commit"], str) and len(meta["git_commit"]) > 0
    assert isinstance(meta["timestamp"], str) and "T" in meta["timestamp"]

    # Configuration assertions: mock benchmark must not claim production/live MOCK_MODE = false
    config = meta["configuration"]
    assert config["MOCK_MODE"] is True
    assert config["APPLICATION_MOCK_MODE"] == getattr(settings, "MOCK_MODE", False)
    assert config["BENCHMARK_MODE"] == "mock"
    assert "MAX_CONCURRENT_SCRAPES" in config
    assert "DATA_PROCESSED_DIR" in config


@pytest.mark.asyncio
async def test_provider_accounting(tmp_path: Path):
    """Test that provider accounting in mock mode is accurate and does not claim fallbacks or ghost providers."""
    output_report = tmp_path / "provider_accounting_report.json"
    report = await run_benchmark(
        mode="mock",
        record_count=6,
        iterations=2,
        warmup_records=1,
        output_path=output_report,
    )

    llm_section = report["llm"]
    assert llm_section["total_calls"] == 12
    assert llm_section["successful_calls"] == 12
    assert llm_section["failed_calls"] == 0
    assert llm_section["fallback_calls"] == 0
    assert llm_section["fallback_rate"] == 0.0
    assert report["rates"]["llm_fallback_rate"] == 0.0

    # Provider breakdown: must only contain Mock, no fabricated 0 counts for uncalled providers
    breakdown = llm_section["provider_breakdown"]
    assert "Mock" in breakdown
    assert breakdown["Mock"]["count"] == 12
    assert breakdown["Mock"]["percentage"] == 100.0
    assert "Gemini" not in breakdown
    assert "Groq" not in breakdown
    assert "DeepSeek" not in breakdown

    # Live mode provider accounting test using collector
    live_collector = BenchmarkMetricsCollector(mode="live", requested_records=5, iterations=1)
    assert "Gemini" in live_collector.provider_counts
    assert "Groq" in live_collector.provider_counts
    assert "DeepSeek" in live_collector.provider_counts
    assert "RuleBased" in live_collector.provider_counts

    live_collector.record_llm_call(provider="Gemini", latency_ms=10.0, success=True, is_fallback=False)
    live_collector.record_llm_call(provider="Groq", latency_ms=20.0, success=True, is_fallback=True)
    live_data = live_collector.to_dict()

    live_breakdown = live_data["llm"]["provider_breakdown"]
    assert live_breakdown["Gemini"]["count"] == 1
    assert live_breakdown["Groq"]["count"] == 1
    assert live_data["llm"]["fallback_calls"] == 1
    assert live_data["llm"]["fallback_rate"] == 0.5


@pytest.mark.asyncio
async def test_report_schema(tmp_path: Path):
    """Test that generated benchmark report adheres strictly to the expected schema and constraints."""
    output_report = tmp_path / "schema_check_report.json"
    report = await run_benchmark(
        mode="mock",
        record_count=5,
        iterations=1,
        warmup_records=1,
        output_path=output_report,
    )

    expected_top_keys = {
        "metadata",
        "pipeline",
        "rates",
        "scraper",
        "llm",
        "extraction_quality",
        "resolution",
        "vector_search",
    }
    assert set(report.keys()) == expected_top_keys

    # Rates bounded in [0.0, 1.0]
    rates = report["rates"]
    for rate_key in [
        "success_rate",
        "failure_rate",
        "llm_fallback_rate",
        "schema_validation_success_rate",
        "duplicate_detection_rate",
    ]:
        assert 0.0 <= rates[rate_key] <= 1.0, f"Rate {rate_key} out of bounds: {rates[rate_key]}"
    assert round(rates["success_rate"] + rates["failure_rate"], 2) == 1.0

    # Latency statistics structures
    for section_key in ["scraper", "llm", "vector_search"]:
        stats = report[section_key]["stats_ms"]
        for stat_field in ["mean", "median", "p50", "p95", "min", "max"]:
            assert stat_field in stats
            assert isinstance(stats[stat_field], (int, float))
            assert stats[stat_field] >= 0.0
        assert stats["min"] <= stats["max"]
        assert stats["min"] <= stats["p50"] <= stats["max"]


@pytest.mark.asyncio
async def test_repeated_mock_benchmark_reproducibility(tmp_path: Path):
    """Test that two consecutive mock benchmark runs produce identical deterministic metrics."""
    run1_file = tmp_path / "run1.json"
    run2_file = tmp_path / "run2.json"

    run1 = await run_benchmark(
        mode="mock",
        record_count=10,
        iterations=2,
        warmup_records=2,
        output_path=run1_file,
    )

    run2 = await run_benchmark(
        mode="mock",
        record_count=10,
        iterations=2,
        warmup_records=2,
        output_path=run2_file,
    )

    # 1. Pipeline deterministic numbers
    assert run1["pipeline"]["records_processed"] == run2["pipeline"]["records_processed"] == 20
    assert run1["pipeline"]["successful_records"] == run2["pipeline"]["successful_records"] == 20
    assert run1["pipeline"]["failed_records"] == run2["pipeline"]["failed_records"] == 0

    # 2. Key rates deterministic match
    assert run1["rates"] == run2["rates"]

    # 3. LLM call counts and provider breakdown deterministic match
    assert run1["llm"]["total_calls"] == run2["llm"]["total_calls"] == 20
    assert run1["llm"]["successful_calls"] == run2["llm"]["successful_calls"] == 20
    assert run1["llm"]["failed_calls"] == run2["llm"]["failed_calls"] == 0
    assert run1["llm"]["fallback_calls"] == run2["llm"]["fallback_calls"] == 0
    assert run1["llm"]["fallback_rate"] == run2["llm"]["fallback_rate"] == 0.0
    assert run1["llm"]["provider_breakdown"] == run2["llm"]["provider_breakdown"]

    # 4. Extraction quality deterministic match
    assert run1["extraction_quality"] == run2["extraction_quality"]

    # 5. Entity resolution counts deterministic match
    assert run1["resolution"] == run2["resolution"]

    # 6. Vector search indexed count deterministic match
    assert run1["vector_search"]["indexed_records"] == run2["vector_search"]["indexed_records"] == 10


@pytest.mark.asyncio
async def test_incremental_benchmark_execution(tmp_path: Path):
    """Test that run_incremental_benchmark completes and generates a valid report structure."""
    out_file = tmp_path / "incremental_report.json"
    report = await run_incremental_benchmark(
        record_count=10,
        output_path=out_file,
        mode="mock",
    )

    assert out_file.exists()
    assert report["metadata"]["record_count"] == 10
    assert report["baseline_full_processing"]["records_processed"] == 10
    assert report["baseline_full_processing"]["llm_calls"] == 10
    assert report["baseline_full_processing"]["embeddings_generated"] == 10

    # Incremental unchanged checks
    assert report["incremental_processing_unchanged"]["records_evaluated"] == 10
    assert report["incremental_processing_unchanged"]["documents_skipped"] == 10
    assert report["incremental_processing_unchanged"]["skip_percentage"] == 100.0
    assert report["incremental_processing_unchanged"]["llm_calls"] == 0
    assert report["incremental_processing_unchanged"]["llm_calls_saved"] == 10
    assert report["incremental_processing_unchanged"]["embeddings_saved"] == 10

    # Incremental selective checks
    assert report["incremental_processing_selective_50pct"]["documents_skipped"] == 5
    assert report["incremental_processing_selective_50pct"]["llm_calls"] == 5
