# Tripwire Scientific Evaluation & Benchmarking Framework

The `evaluation` package provides a scientific, reproducible benchmarking and comparison harness to quantitatively measure and track Tripwire Data Platform performance changes across code revisions.

---

## Key Features & Scientific Design

1. **Warm-Up Phase Handling**:
   * Evaluates `--warmup-records` to initialize memory caches, vector embeddings, and LLM fallbacks prior to steady-state measurement.
2. **Multi-Iteration Execution**:
   * Runs `--iterations I` benchmark loops to calculate stable aggregate statistics.
3. **Cold-Start vs. Steady-State Timing**:
   * Distinguishes cold-start setup time (`cold_start_time_seconds`) from steady-state pipeline runtime (`steady_state_time_seconds`).
4. **Comprehensive Aggregate Statistics**:
   * Measures `mean`, `median`, `p50`, `p95`, `min`, and `max` distributions for Scraper, LLM, and Vector Search latencies (ms).
5. **Operational Key Rates**:
   * `success_rate`: Ratio of validated records to total records processed
   * `failure_rate`: Ratio of failed records to total records processed
   * `llm_fallback_rate`: Ratio of extractions requiring tier > 1 fallback providers (0.0 in offline mock benchmark mode, as `MockProvider` is the primary deterministic execution engine rather than a failure fallback)
   * `schema_validation_success_rate`: Ratio of records matching Pydantic schema
   * `duplicate_detection_rate`: Ratio of merged entities (exact + normalized + fuzzy) to total evaluated
6. **Reproducibility & Environment Metadata**:
   * Git commit hash (`git_commit`), ISO timestamp (`timestamp`), benchmark mode (`benchmark_mode`, `mode`), application mock mode (`application_mock_mode`), dataset sizes (`records`, `iterations`, `warmup`), and active operational configuration (`MAX_CONCURRENT_SCRAPES`, `SCRAPER_TIMEOUT_SECONDS`, `RATE_LIMIT_TOKENS_PER_SEC`, `MOCK_MODE`, `APPLICATION_MOCK_MODE`, `BENCHMARK_MODE`, `DATA_PROCESSED_DIR`).
   * In offline mock mode (`--mode mock`), operational configuration `MOCK_MODE` is explicitly `true` so mock benchmarks never claim production/live execution, while `application_mock_mode` accurately records the host environment setting.
   * `provider_breakdown` accurately describes mock execution (`Mock`), omitting unconfigured live frontier providers (Gemini, Groq, DeepSeek).
7. **Baseline vs. New Comparison Harness**:
   * CLI tool `python -m evaluation.compare` computes metric deltas, percentage changes, and color-coded terminal diffs between baseline and new reports.

---

## Executing the Benchmark

### 1. Offline Deterministic Mode (Recommended)

```bash
python -m evaluation.benchmark --mode mock --records 20 --iterations 3 --warmup-records 2 --output evaluation/reports/baseline.json
```

### 2. Live API Mode

```bash
python -m evaluation.benchmark --mode live --records 50 --iterations 3 --warmup-records 5 --output evaluation/reports/live_report.json
```

### 3. Incremental Processing Benchmark

```bash
python -m evaluation.incremental_benchmark --records 20 --output evaluation/reports/incremental_benchmark.json
```

---

## Comparing Benchmark Reports

Compare two benchmark JSON report runs to evaluate performance changes, regressions, or optimization impact:

```bash
python -m evaluation.compare evaluation/reports/baseline.json evaluation/reports/new.json --output evaluation/reports/comparison_diff.json
```

### Output Comparison Metrics:
* **Throughput Change**: Records/sec delta and percentage change
* **Latency Changes**: Scraper, LLM, and Vector Search mean/p95 deltas
* **Rate Changes**: Success rate, failure rate, LLM fallback rate, schema validation rate, and duplicate detection rate deltas
* **Timing Deltas**: Cold-start and steady-state processing time changes

---

## CLI Options Reference

### Benchmark Runner (`python -m evaluation.benchmark`)

| Argument | Type / Default | Description |
| :--- | :--- | :--- |
| `--mode` | `mock` / `live` (`mock`) | Operational mode: `mock` (offline zero-cost) or `live` |
| `--records` | Integer (`20`) | Target record count per iteration |
| `--iterations` | Integer (`3`) | Number of benchmark iterations to execute |
| `--warmup-records` | Integer (`2`) | Warm-up records to process before timing steady-state |
| `--output` | Path (`evaluation/reports/benchmark_report.json`) | Target report JSON destination path |

### Comparison Tool (`python -m evaluation.compare`)

| Argument | Type | Description |
| :--- | :--- | :--- |
| `baseline` | Path (Positional) | Baseline benchmark report JSON path |
| `new` | Path (Positional) | New benchmark report JSON path |
| `--output` | Path (Optional) | Optional output JSON path for comparison diff |

---

## Unit & Integration Tests

```bash
pytest tests/test_evaluation.py -v
```
