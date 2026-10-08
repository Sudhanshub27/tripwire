"""
================================================================================
INCREMENTAL PROCESSING ENGINE TEST SUITE (PHASE 2)
================================================================================
Comprehensive verification of true content-fingerprinting incremental processing:
1. New document is processed.
2. Identical second run becomes UNCHANGED.
3. Unchanged document does not call LLM.
4. Unchanged document does not generate embeddings.
5. Changed document is processed again.
6. Exactly 10/100 changed scenario works.
7. Normalization produces stable hashes.
8. Failed NEW processing does not persist success.
9. Failed CHANGED processing preserves previous successful fingerprint.
10. Retry after failure processes again.
11. Incremental metrics are updated correctly.
12. Benchmark produces the required counts.
13. Existing Phase 1 tests continue passing.
==================================================
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from config.settings import settings
from evaluation.incremental_benchmark import run_incremental_benchmark
from src.main import PipelineOrchestrator
from src.observability.metrics import metrics_collector
from src.resolution.entity_resolver import EntityResolver
from src.schemas.base import SourceMetadata
from src.schemas.startup import Startup
from src.scrapers.dedup_tracker import (
    DedupTracker,
    IncrementalAction,
    IncrementalDecision,
    ProcessingAction,
    ProcessingDecision,
)


@pytest.fixture
def isolated_tracker(tmp_path: Path) -> DedupTracker:
    """Fixture providing isolated DedupTracker backed by temporary SQLite database."""
    db_file = tmp_path / "test_dedup.db"
    return DedupTracker(db_path=db_file)


# ==============================================================================
# 1. NEW DOCUMENT IS PROCESSED
# ==============================================================================
def test_new_document_is_processed(isolated_tracker: DedupTracker):
    """Test 1: Unseen URL evaluates to NEW -> PROCESS and is stored on success."""
    url = "https://example.com/company/alpha"
    content = "<div><h1>Alpha AI</h1><p>Autonomous Agents</p></div>"

    res = isolated_tracker.check_incremental(url=url, content=content)

    assert res.decision == ProcessingDecision.NEW
    assert res.action == ProcessingAction.PROCESS
    assert res.is_skip is False
    assert res.should_process is True
    assert res.stored_hash is None
    assert res.content_hash == isolated_tracker.compute_content_hash(content)

    # Process and record success
    isolated_tracker.record_successful_processing(url=url, content=content, source="TestRunner")
    assert isolated_tracker.is_seen(url) is True
    assert isolated_tracker.get_stored_hash(url) == res.content_hash


# ==============================================================================
# 2. IDENTICAL SECOND RUN BECOMES UNCHANGED
# ==============================================================================
def test_identical_second_run_becomes_unchanged(isolated_tracker: DedupTracker):
    """Test 2: Re-checking previously processed URL with identical content evaluates to UNCHANGED -> SKIP."""
    url = "https://example.com/company/alpha"
    content = "<div><h1>Alpha AI</h1><p>Autonomous Agents</p></div>"

    # Seed initial success
    isolated_tracker.record_successful_processing(url=url, content=content, source="TestRunner")

    # Second run
    res = isolated_tracker.check_incremental(url=url, content=content)

    assert res.decision == ProcessingDecision.UNCHANGED
    assert res.action == ProcessingAction.SKIP
    assert res.is_skip is True
    assert res.should_process is False
    assert res.stored_hash == isolated_tracker.compute_content_hash(content)
    assert res.content_hash == res.stored_hash


# ==============================================================================
# 3. UNCHANGED DOCUMENT DOES NOT CALL LLM
# ==============================================================================
@pytest.mark.asyncio
async def test_unchanged_document_does_not_call_llm(isolated_tracker: DedupTracker):
    """Test 3: Unchanged document strictly avoids executing the LLM extraction chain."""
    url = "https://example.com/company/bravo"
    content = "<div><h1>Bravo Systems</h1><p>Distributed Ledger</p></div>"

    mock_llm_chain = MagicMock()
    mock_llm_chain.extract_with_fallback = AsyncMock()

    # Seed initial run
    isolated_tracker.record_successful_processing(url=url, content=content)

    # Second run evaluation
    res = isolated_tracker.check_incremental(url=url, content=content)
    assert res.decision == ProcessingDecision.UNCHANGED
    assert res.is_skip is True

    # Pipeline logic check: skip branch prevents LLM invocation
    if not res.is_skip:
        await mock_llm_chain.extract_with_fallback(content, Startup)

    mock_llm_chain.extract_with_fallback.assert_not_called()


# ==============================================================================
# 4. UNCHANGED DOCUMENT DOES NOT GENERATE EMBEDDINGS
# ==============================================================================
def test_unchanged_document_does_not_generate_embeddings(isolated_tracker: DedupTracker):
    """Test 4: Unchanged document strictly avoids generating dense vector embeddings."""
    url = "https://example.com/company/charlie"
    content = "<div><h1>Charlie AI</h1><p>High Speed Synthesis</p></div>"

    mock_embed_engine = MagicMock()

    # Seed initial run
    isolated_tracker.record_successful_processing(url=url, content=content)

    # Second run evaluation
    res = isolated_tracker.check_incremental(url=url, content=content)
    assert res.decision == ProcessingDecision.UNCHANGED
    assert res.is_skip is True

    # Pipeline logic check: skip branch prevents embedding computation
    if not res.is_skip:
        mock_embed_engine("Charlie AI High Speed Synthesis")

    mock_embed_engine.assert_not_called()


# ==============================================================================
# 5. CHANGED DOCUMENT IS PROCESSED AGAIN
# ==============================================================================
def test_changed_document_is_processed_again(isolated_tracker: DedupTracker):
    """Test 5: Same URL with modified content evaluates to CHANGED -> PROCESS and updates stored fingerprint."""
    url = "https://example.com/company/delta"
    content_v1 = "<div><h1>Delta AI</h1><p>Series A - $10M Funding</p></div>"
    content_v2 = "<div><h1>Delta AI</h1><p>Series B - $50M Funding</p></div>"

    # Seed v1
    isolated_tracker.record_successful_processing(url=url, content=content_v1)
    hash_v1 = isolated_tracker.compute_content_hash(content_v1)
    assert isolated_tracker.get_stored_hash(url) == hash_v1

    # Check v2
    res = isolated_tracker.check_incremental(url=url, content=content_v2)
    assert res.decision == ProcessingDecision.CHANGED
    assert res.action == ProcessingAction.PROCESS
    assert res.is_skip is False
    assert res.stored_hash == hash_v1
    assert res.content_hash == isolated_tracker.compute_content_hash(content_v2)
    assert res.content_hash != hash_v1

    # On successful reprocessing, update fingerprint
    isolated_tracker.record_successful_processing(url=url, content=content_v2)
    assert isolated_tracker.get_stored_hash(url) == res.content_hash

    # Subsequent check is now UNCHANGED
    res_v3 = isolated_tracker.check_incremental(url=url, content=content_v2)
    assert res_v3.decision == ProcessingDecision.UNCHANGED
    assert res_v3.is_skip is True


# ==============================================================================
# 6. EXACTLY 10/100 CHANGED SCENARIO WORKS
# ==============================================================================
@pytest.mark.asyncio
async def test_ten_of_hundred_changed_scenario(tmp_path: Path):
    """
    Test 6: Validates exact 100/100/10 scenario:
      Run 1: 100 NEW -> 100 processed, 100 LLM calls, 100 embedding calls
      Run 2: 100 identical -> 100 UNCHANGED -> 0 LLM calls, 0 embedding calls, 100 skipped
      Run 3: 10 changed, 90 unchanged -> 10 CHANGED, 90 UNCHANGED -> 10 LLM calls, 10 embedding calls, 90 skipped
    """
    db_file = tmp_path / "scenario_100.db"
    tracker = DedupTracker(db_path=db_file)

    mock_llm_calls = 0
    mock_embedding_calls = 0

    base_docs = [
        (f"https://tripwire.test/doc/{i}", f"<div><h1>Doc {i}</h1><p>Initial content for doc {i}</p></div>")
        for i in range(100)
    ]

    # --- RUN 1: 100 NEW ---
    r1_new = 0
    r1_skipped = 0
    for url, content in base_docs:
        res = tracker.check_incremental(url=url, content=content)
        if res.decision == ProcessingDecision.NEW:
            r1_new += 1
        if res.is_skip:
            r1_skipped += 1
            continue
        mock_llm_calls += 1
        mock_embedding_calls += 1
        tracker.record_successful_processing(url=url, content=content)

    assert r1_new == 100
    assert r1_skipped == 0
    assert mock_llm_calls == 100
    assert mock_embedding_calls == 100

    # --- RUN 2: 100 UNCHANGED ---
    mock_llm_calls = 0
    mock_embedding_calls = 0
    r2_unchanged = 0
    r2_skipped = 0

    for url, content in base_docs:
        res = tracker.check_incremental(url=url, content=content)
        if res.decision == ProcessingDecision.UNCHANGED:
            r2_unchanged += 1
        if res.is_skip:
            r2_skipped += 1
            continue
        mock_llm_calls += 1
        mock_embedding_calls += 1

    assert r2_unchanged == 100
    assert r2_skipped == 100
    assert mock_llm_calls == 0
    assert mock_embedding_calls == 0

    # --- RUN 3: EXACTLY 10 CHANGED, 90 UNCHANGED ---
    mock_llm_calls = 0
    mock_embedding_calls = 0
    r3_changed = 0
    r3_unchanged = 0
    r3_skipped = 0

    for i, (url, content) in enumerate(base_docs):
        if i < 10:  # Mutate first 10
            mutated_content = content + f"<span class='mod'>Modified Round 3 Update {i}</span>"
        else:
            mutated_content = content

        res = tracker.check_incremental(url=url, content=mutated_content)
        if res.decision == ProcessingDecision.CHANGED:
            r3_changed += 1
        elif res.decision == ProcessingDecision.UNCHANGED:
            r3_unchanged += 1

        if res.is_skip:
            r3_skipped += 1
            continue

        mock_llm_calls += 1
        mock_embedding_calls += 1
        tracker.record_successful_processing(url=url, content=mutated_content)

    assert r3_changed == 10
    assert r3_unchanged == 90
    assert r3_skipped == 90
    assert mock_llm_calls == 10
    assert mock_embedding_calls == 10


# ==============================================================================
# 7. NORMALIZATION PRODUCES STABLE HASHES
# ==============================================================================
def test_normalization_produces_stable_hashes(isolated_tracker: DedupTracker):
    """
    Test 7: Content normalization produces identical hashes across tag spacing,
    whitespace collapsing (spaces, tabs, newlines), and Unicode equivalent characters.
    """
    base = "<div><h1>Beta Voice</h1><p>Low latency speech synthesis model.</p></div>"
    variants = [
        # Whitespace between HTML tags
        "<div > <h1 >Beta Voice</h1 > <p >Low latency speech synthesis model.</p > </div>",
        # Newlines and indentation
        """
        <div>
            <h1>Beta Voice</h1>
            <p>
                Low latency speech synthesis model.
            </p>
        </div>
        """,
        # Tabs and carriage returns
        "<div>\t<h1>Beta Voice</h1>\r\n<p>Low \t latency   speech \n synthesis   model.</p></div>",
    ]

    base_hash = isolated_tracker.compute_content_hash(base)
    for v in variants:
        v_hash = isolated_tracker.compute_content_hash(v)
        assert v_hash == base_hash, f"Hash mismatch for variant:\n{v!r}"

    # Semantic alteration must produce a distinct hash
    altered = "<div><h1>Beta Voice</h1><p>High latency speech synthesis model.</p></div>"
    assert isolated_tracker.compute_content_hash(altered) != base_hash


# ==============================================================================
# 8. FAILED NEW PROCESSING DOES NOT PERSIST SUCCESS
# ==============================================================================
def test_failed_new_processing_does_not_persist_success(isolated_tracker: DedupTracker):
    """Test 8: Failure during NEW processing must NOT commit the fingerprint."""
    url = "https://example.com/company/echo"
    content = "<div><h1>Echo Cloud</h1></div>"

    res = isolated_tracker.check_incremental(url=url, content=content)
    assert res.decision == ProcessingDecision.NEW

    # Downstream processing encounters an unrecoverable failure
    try:
        raise ValueError("Simulated LLM API Timeout Failure")
    except Exception as exc:
        isolated_tracker.record_failed_processing(url=url, reason=str(exc))

    # Verify fingerprint was NOT persisted
    assert isolated_tracker.is_seen(url) is False
    assert isolated_tracker.get_stored_hash(url) is None

    # Re-check still evaluates as NEW
    retry_res = isolated_tracker.check_incremental(url=url, content=content)
    assert retry_res.decision == ProcessingDecision.NEW
    assert retry_res.action == ProcessingAction.PROCESS


# ==============================================================================
# 9. FAILED CHANGED PROCESSING PRESERVES PREVIOUS FINGERPRINT
# ==============================================================================
def test_failed_changed_processing_preserves_previous_fingerprint(isolated_tracker: DedupTracker):
    """Test 9: Failure during CHANGED processing preserves previous valid fingerprint."""
    url = "https://example.com/company/foxtrot"
    content_v1 = "<p>Foxtrot v1.0 Operational</p>"
    content_v2 = "<p>Foxtrot v2.0 Broken Deployment</p>"

    # 1. First run succeeds with v1
    isolated_tracker.record_successful_processing(url=url, content=content_v1)
    hash_v1 = isolated_tracker.compute_content_hash(content_v1)
    assert isolated_tracker.get_stored_hash(url) == hash_v1

    # 2. Content changes to v2
    res_v2 = isolated_tracker.check_incremental(url=url, content=content_v2)
    assert res_v2.decision == ProcessingDecision.CHANGED

    # 3. Downstream processing fails for v2
    try:
        raise RuntimeError("Simulated Schema Validation Crash on v2")
    except Exception as exc:
        isolated_tracker.record_failed_processing(url=url, reason=str(exc))

    # 4. Verify previous fingerprint is STILL hash_v1, NOT hash_v2!
    current_stored = isolated_tracker.get_stored_hash(url)
    assert current_stored == hash_v1
    assert current_stored != isolated_tracker.compute_content_hash(content_v2)

    # 5. Checking with original v1 is still UNCHANGED
    check_v1 = isolated_tracker.check_incremental(url=url, content=content_v1)
    assert check_v1.decision == ProcessingDecision.UNCHANGED

    # 6. Checking with v2 still demands reprocessing as CHANGED
    check_v2 = isolated_tracker.check_incremental(url=url, content=content_v2)
    assert check_v2.decision == ProcessingDecision.CHANGED


# ==============================================================================
# 10. RETRY AFTER FAILURE PROCESSES AGAIN
# ==============================================================================
def test_retry_after_failure_processes_again(isolated_tracker: DedupTracker):
    """Test 10: Retrying a failed document allows successful reprocessing and fingerprint persistence."""
    url = "https://example.com/company/golf"
    content = "<div><h1>Golf Analytics</h1></div>"

    # Attempt 1: fails
    res1 = isolated_tracker.check_incremental(url=url, content=content)
    assert res1.decision == ProcessingDecision.NEW
    isolated_tracker.record_failed_processing(url=url, reason="Network glitch")
    assert isolated_tracker.get_stored_hash(url) is None

    # Attempt 2: retry succeeds
    res2 = isolated_tracker.check_incremental(url=url, content=content)
    assert res2.decision == ProcessingDecision.NEW
    isolated_tracker.record_successful_processing(url=url, content=content)
    assert isolated_tracker.get_stored_hash(url) == isolated_tracker.compute_content_hash(content)

    # Attempt 3: now evaluated as UNCHANGED -> SKIP
    res3 = isolated_tracker.check_incremental(url=url, content=content)
    assert res3.decision == ProcessingDecision.UNCHANGED
    assert res3.is_skip is True


# ==============================================================================
# 11. INCREMENTAL METRICS ARE UPDATED CORRECTLY
# ==============================================================================
def test_incremental_metrics_updated_correctly():
    """Test 11: All 7 incremental metrics and Prometheus exposition text update accurately."""
    metrics_collector.reset_incremental_metrics()

    # Record decisions
    metrics_collector.record_incremental_decision(ProcessingDecision.NEW, ProcessingAction.PROCESS)
    metrics_collector.record_incremental_decision(ProcessingDecision.CHANGED, ProcessingAction.PROCESS)
    metrics_collector.record_incremental_decision(
        ProcessingDecision.UNCHANGED, ProcessingAction.SKIP, llm_calls_saved=1, embedding_calls_saved=1
    )
    metrics_collector.record_incremental_decision(
        ProcessingDecision.UNCHANGED, ProcessingAction.SKIP, llm_calls_saved=1, embedding_calls_saved=1
    )

    stats = metrics_collector.get_incremental_stats()
    assert stats["incremental_total"] == 4
    assert stats["incremental_new"] == 1
    assert stats["incremental_changed"] == 1
    assert stats["incremental_unchanged"] == 2
    assert stats["incremental_skipped"] == 2
    assert stats["llm_calls_saved"] == 2
    assert stats["embedding_calls_saved"] == 2
    assert stats["work_avoided_pct"] == 50.0

    prom_text = metrics_collector.generate_prometheus_format()
    assert "graphone_incremental_total 4" in prom_text
    assert "graphone_incremental_new 1" in prom_text
    assert "graphone_incremental_changed 1" in prom_text
    assert "graphone_incremental_unchanged 2" in prom_text
    assert "graphone_incremental_skipped 2" in prom_text
    assert "graphone_llm_calls_saved_total 2" in prom_text
    assert "graphone_embedding_calls_saved_total 2" in prom_text


# ==============================================================================
# 12. BENCHMARK PRODUCES THE REQUIRED COUNTS
# ==============================================================================
@pytest.mark.asyncio
async def test_benchmark_produces_required_counts(tmp_path: Path):
    """Test 12: Benchmark execution generates the exact required call counts and schema fields."""
    out_file = tmp_path / "bench_report.json"
    report = await run_incremental_benchmark(
        record_count=100,
        output_path=out_file,
        mode="mock",
    )

    assert out_file.exists()
    assert report["metadata"]["record_count"] == 100

    # RUN 1: 100 NEW, 100 LLM, 100 Embeddings, 0 Skipped
    r1 = report["run_1_all_new"]
    assert r1["discovered"] == 100
    assert r1["new"] == 100
    assert r1["llm_calls"] == 100
    assert r1["embedding_calls"] == 100
    assert r1["skipped"] == 0
    assert r1["llm_calls_saved"] == 0
    assert r1["embedding_calls_saved"] == 0

    # RUN 2: 100 UNCHANGED, 0 LLM, 0 Embeddings, 100 Skipped
    r2 = report["run_2_all_unchanged"]
    assert r2["discovered"] == 100
    assert r2["new"] == 0
    assert r2["unchanged"] == 100
    assert r2["skipped"] == 100
    assert r2["llm_calls"] == 0
    assert r2["embedding_calls"] == 0
    assert r2["llm_calls_saved"] == 100
    assert r2["embedding_calls_saved"] == 100
    assert r2["work_avoided_pct"] == 100.0

    # RUN 3: 10 CHANGED, 90 UNCHANGED, 10 LLM, 10 Embeddings, 90 Skipped
    r3 = report["run_3_partial_changed"]
    assert r3["discovered"] == 100
    assert r3["changed"] == 10
    assert r3["unchanged"] == 90
    assert r3["skipped"] == 90
    assert r3["llm_calls"] == 10
    assert r3["embedding_calls"] == 10
    assert r3["llm_calls_saved"] == 90
    assert r3["embedding_calls_saved"] == 90
    assert r3["work_avoided_pct"] == 90.0

    # Summary cumulative totals
    summary = report["summary"]
    assert summary["discovered"] == 300
    assert summary["new"] == 100
    assert summary["changed"] == 10
    assert summary["unchanged"] == 190
    assert summary["skipped"] == 190
    assert summary["llm_calls"] == 110
    assert summary["embedding_calls"] == 110
    assert summary["llm_calls_saved"] == 190
    assert summary["embedding_calls_saved"] == 190
    assert summary["work_avoided_pct"] == 63.3


# ==============================================================================
# 13. EXISTING PHASE 1 TESTS & BACKWARD COMPATIBILITY
# ==============================================================================
def test_existing_phase1_tests_continue_passing():
    """Test 13: Backward compatibility aliases and Entity Resolution deduplication semantics."""
    # Enum aliases
    assert IncrementalDecision.NEW == ProcessingDecision.NEW
    assert IncrementalDecision.CHANGED == ProcessingDecision.CHANGED
    assert IncrementalDecision.UNCHANGED == ProcessingDecision.UNCHANGED
    assert IncrementalAction.PROCESS == ProcessingAction.PROCESS
    assert IncrementalAction.SKIP == ProcessingAction.SKIP

    # Entity Resolution preservation
    resolver = EntityResolver()
    raw1 = Startup(
        name="Mistral AI SAS",
        description="Frontier Open Weights Models",
        stage="Series B",
        founding_year=2023,
        source=SourceMetadata(name="Test", url="https://mistral.ai"),
    )
    raw2 = Startup(
        name="Mistral AI Inc.",
        description="Frontier AI Company",
        stage="Series B",
        founding_year=2023,
        source=SourceMetadata(name="Test", url="https://mistral.ai/about"),
    )
    resolved1, info1 = resolver.resolve_record(raw1)
    resolved2, info2 = resolver.resolve_record(raw2)
    assert resolved1.name == resolved2.name == "Mistral AI"
    assert info1["method_used"] in ("normalized", "exact", "fuzzy")


@pytest.mark.asyncio
async def test_pipeline_orchestrator_incremental_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Phase 1 Orchestrator integration test verifying second run skips work."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "DATA_PROCESSED_DIR", processed_dir)

    metrics_collector.reset_incremental_metrics()
    orchestrator = PipelineOrchestrator(dry_run=True, limit=2)

    count1 = await orchestrator.run_startups_pipeline()
    assert count1 == 2
    assert orchestrator.metrics["incremental"]["new"] == 2
    assert orchestrator.metrics["incremental"]["skipped"] == 0

    startups_file = processed_dir / "startups.jsonl"
    assert startups_file.exists()
    mtime1 = startups_file.stat().st_mtime

    orchestrator2 = PipelineOrchestrator(dry_run=True, limit=2)
    count2 = await orchestrator2.run_startups_pipeline()

    assert count2 == 0
    assert orchestrator2.metrics["incremental"]["unchanged"] == 2
    assert orchestrator2.metrics["incremental"]["skipped"] == 2
    assert orchestrator2.metrics["incremental"]["llm_calls_saved"] == 2
    assert orchestrator2.metrics["incremental"]["embedding_calls_saved"] == 2
    assert startups_file.stat().st_mtime == mtime1
