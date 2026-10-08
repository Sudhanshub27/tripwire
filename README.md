# Tripwire Pipeline

[![CI](https://github.com/Sudhanshub27/graphone-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Sudhanshub27/graphone-pipeline/actions/workflows/ci.yml)

Tripwire Pipeline is an enterprise-grade asynchronous data ingestion, LLM entity extraction, entity resolution, knowledge graph generation, and hybrid vector search framework built in Python 3.11+.

It combines multi-source web scraping (supporting static HTTP requests via aiohttp and client-rendered JavaScript via Playwright), resilient LLM extraction fallback chains (Google Gemini, Groq, DeepSeek), fuzzy entity deduplication, automated Knowledge Graph relational triple generation with Cypher export, LanceDB vector embeddings, Prometheus observability metrics, and a React + FastAPI management dashboard.

---

## Architecture Overview

```
                                  TRIPWIRE PIPELINE ARCHITECTURE

+---------------------------------------------------------------------------------------------------+
|                                     Async Scraper Network Layer                                   |
|                          (aiohttp / httpx / Playwright / TokenBucket Rate Limiter)                |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+-------------------------------------------------+
|                                       Raw Cache Layer                                             |
|                                         (data/raw/)                                               |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+-------------------------------------------------+
|                                 Multi-Tier LLM Extraction Chain                                   |
|                       (Gemini 2.5 -> Groq -> DeepSeek -> Dense Summarizer)                        |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+-------------------------------------------------+
|                                Pydantic v2 Schema Validation                                      |
|                  (Startup, Product, ResearchPaper, Job, News, BaseRecord)                         |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+-------------------------------------------------+
|                                 Entity Resolution Engine                                          |
|                (RapidFuzz token_sort_ratio + Legal Suffix Normalization)                          |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
       +------------------------------------------+------------------------------------------+
       |                                          |                                          |
       v                                          v                                          v
+-----------------------------+    +-----------------------------+    +-----------------------------+
|    Knowledge Graph Engine   |    |    LanceDB Vector Search    |    |    Prometheus Telemetry     |
|   (Triples & Cypher Export) |    |  (128D Dense Embeddings)    |    |  (/metrics & LLM Telemetry) |
+--------------+--------------+    +--------------+--------------+    +--------------+--------------+
               |                                  |                                  |
               +----------------------------------+----------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+-------------------------------------------------+
|                             FastAPI Backend & React Control Center                                |
|                        (Overview, Browser, Resolution, Graph, Vector, Logs)                       |
+---------------------------------------------------------------------------------------------------+
```

---

## Core Technical Features

### 1. Asynchronous Multi-Source Web Ingestion
* **Hybrid Scraping**: Supports static HTML parsing (BeautifulSoup4 / Trafilatura) and headless browser automation (Playwright Chromium) for dynamic single-page applications.
* **Concurrency & Rate Control**: TokenBucket rate limiters protect upstream endpoints, enforcing configurable domain-level request quotas and concurrency caps.

### 2. Multi-Tier LLM Orchestration & Fallback Chain
* **Tiered Failover Execution**: Routes extraction requests sequentially through primary and fallback provider tiers (Google Gemini -> Groq -> DeepSeek -> Rule-Based Extractor).
* **Token Optimization & Chunking**: Dynamically estimates document token counts using `tiktoken`. Large documents exceeding safety thresholds are semantically chunked or processed via a dense summarization layer to prevent HTTP 413 or context-window overflow errors.
* **Structured Output Validation**: Enforces strict Pydantic v2 schema compliance with automatic type coercion and error trapping.

### 3. Entity Resolution & Deduplication
* **Canonical Matching**: Combines fast-path exact alias matching with normalized legal suffix stripping (stripping "Inc.", "LLC", "PBC", "Corp").
* **Fuzzy Normalization**: Employs RapidFuzz token-sort similarity scoring to detect duplicate records across disparate data sources before persisting to `data/processed/`.

### 4. Knowledge Graph & Relational Topology Engine
* **Relational Triple Extraction**: Automatically identifies topological relationships between entities (e.g., `Startup -[PRODUCES]-> Product`, `Job -[POSTED_BY]-> Startup`).
* **Cypher Export Engine**: Generates ready-to-run Neo4j Cypher import scripts (`MERGE` statements) for seamless downstream graph database synchronization.

### 5. LanceDB Hybrid Vector Search Engine
* **128-Dimensional Embeddings**: Computes normalized dense text feature vectors for all processed entity records.
* **Hybrid Similarity Ranking**: Combines vector cosine similarity scoring with keyword BM25 text match boosting.
* **Command Palette Integration**: Search vector embeddings directly via the React Command Palette (`Cmd + K`) or through the dedicated Vector Search workbench interface.

### 6. Prometheus Telemetry & System Observability
* **Real-time Metrics**: Exposes operational counters and histograms at `/metrics` following standard Prometheus format (`scrapes_total`, `scrape_errors_total`, `llm_calls_total`, `llm_call_latency_seconds`).
* **Execution Telemetry Logs**: Records structured LLM call execution details (model name, tier, input tokens, output tokens, latency, cost estimate) to `data/processed/llm_calls_log.jsonl`.

### 7. React + FastAPI Control Center
* **Single-Command Orchestration**: `./run.sh` script automates dependency checks, environment verification, React asset bundling, and FastAPI server execution.
* **Real-Time Dashboards**: Includes overview statistics, interactive data browser, deduplication logs viewer, Knowledge Graph explorer, LanceDB vector search workbench, and live log tailing interface.

---

## Directory Structure

```
tripwire-pipeline/
├── src/
│   ├── scrapers/       # Site-specific async scrapers (aiohttp, Playwright, rate limiters)
│   ├── llm/            # Multi-provider fallback chain (Gemini, Groq, DeepSeek, chunking)
│   ├── resolution/     # Entity resolution, normalization, and deduplication logic
│   ├── schemas/        # Pydantic v2 data models (Startup, Product, Paper, Job, News)
│   ├── vector/         # LanceDB vector embedding storage and hybrid search engine
│   ├── observability/  # Prometheus MetricsCollector singleton and telemetry logger
│   ├── export/         # Google Sheets export integrations
│   └── dashboard/      # FastAPI backend endpoints and React web management dashboard
├── config/             # Centralized settings and logging configurations
├── data/
│   ├── raw/            # Scraped raw HTML and JSON payload cache
│   └── processed/      # Validated entity JSONL logs and LanceDB vector index files
├── tests/              # Pytest unit and integration test suite
├── .env.example        # Environment configuration template
├── pyproject.toml      # Linter and project configuration file
├── requirements.txt    # Python dependencies specification
├── run.sh              # Master build and startup shell script
└── README.md           # Technical documentation
```

---

## Installation & Setup

### Prerequisites
* **Python**: `3.11` or higher
* **Node.js**: `18.0.0` or higher (for dashboard frontend compilation)

### Quick Start

1. **Clone Repository & Set Up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to configure your API keys (`GEMINI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`).

4. **Launch Application Dashboard**:
   ```bash
   ./run.sh
   ```
   Access the dashboard at `http://localhost:8000`.

---

## Configuration Reference

| Environment Variable | Description | Default Value |
| :--- | :--- | :--- |
| `MOCK_MODE` | Enable synthetic log generation mode | `false` |
| `GEMINI_API_KEY` | Primary LLM Provider API key | `None` |
| `GROQ_API_KEY` | Secondary LLM Provider API key | `None` |
| `DEEPSEEK_API_KEY` | Fallback LLM Provider API key | `None` |
| `MAX_CONCURRENT_SCRAPES` | Maximum parallel scraper worker tasks | `5` |
| `MAX_CONCURRENT_LLM_CALLS` | Maximum parallel LLM API execution tasks | `3` |
| `RATE_LIMIT_PER_MINUTE` | Global domain rate limit per minute | `60` |
| `HTTP_TIMEOUT_SECONDS` | HTTP request timeout limit (seconds) | `30` |
| `LOG_LEVEL` | Global logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

---

## Data Schemas

All record schemas inherit from `BaseRecord` to enforce uniform data provenance tracking:

* **Provenance Metadata**: `schemaVersion`, `recordType`, `source` (`name`, `url`), `collectedAt`

### Supported Schemas:
1. **`Startup`**: Company name, description, website, founding year, founders, funding stage, total funding, location, categories/tags, employee count.
2. **`Product`**: Product name, tagline, description, URL, maker company, launch date, categories/tags, pricing model, upvotes count.
3. **`ResearchPaper`**: Paper title, authors, abstract, published date, PDF URL, journal/conference, DOI, topics, citations count.
4. **`Job`**: Role title, hiring company, location, job type, salary range, description, requirements list, posted date, application URL.
5. **`News`**: Headline title, summary, full body content, author, publication timestamp, category tags, sentiment score.

---

## API Endpoints Reference

### Dashboard & Analytics API
* `GET /api/stats`: Fetch real-time system metrics, total records processed, and LLM orchestration breakdown.
* `GET /api/config`: Return active server configuration state (`mockMode`, `logLevel`).
* `POST /api/run`: Trigger manual asynchronous pipeline execution run.

### Data & Resolution API
* `GET /api/records/{category}`: Retrieve processed JSONL records for specified category (`startup`, `product`, `research_paper`, `job`, `news`).
* `GET /api/entity-log`: Fetch entity deduplication resolution log entries.

### Knowledge Graph & Vector Search API
* `GET /api/graph`: Fetch graph node topologies and relational edge triples.
* `GET /api/graph/export`: Export formatted Neo4j Cypher import script.
* `GET /api/search?q={query}&type={type}`: Execute hybrid vector semantic search across LanceDB index.
* `POST /api/search/reindex`: Bulk re-index processed entity records into LanceDB.

### Observability & Logging API
* `GET /metrics`: Prometheus operational metrics endpoint.
* `GET /api/logs?source={file}`: Fetch live log file entries (`scrape.log`, `llm_extraction.log`, `entity_resolution.log`, `llm_calls_log.jsonl`).

---

## Testing & Benchmarking

### Unit & Integration Test Suite

Execute the test suite using `pytest`:

```bash
# Run full unit and integration test suite
pytest tests/ -v

# Run isolated end-to-end pipeline integration test suite
pytest tests/test_e2e_pipeline.py -v
```

To run test coverage analysis:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Reproducible Pipeline Benchmarking Framework

Run quantitative, zero-cost pipeline benchmarks across ingestion throughput, scraper latency, LLM tier fallback, Pydantic validation quality, entity resolution duplicate rates, and LanceDB vector search query latencies:

```bash
# Run offline scientific benchmark (zero-cost, mock mode with warm-up & iterations)
python -m evaluation.benchmark --mode mock --records 20 --iterations 3 --warmup-records 2 --output evaluation/reports/baseline.json

# Compare baseline and new benchmark reports
python -m evaluation.compare evaluation/reports/baseline.json evaluation/reports/new.json --output evaluation/reports/diff.json

# Run LLM extraction quality evaluation across ground-truth dataset
python -m evaluation.llm.evaluator --provider chain --output evaluation/reports/llm_chain_eval.json

# Run Entity Resolution engine evaluation & threshold sweep
python -m evaluation.resolution.evaluator --threshold 85.0 --output evaluation/reports/resolution_eval.json

# Run Incremental Processing Benchmark (Baseline vs Incremental vs 50% Updates)
python -m evaluation.incremental_benchmark --records 20 --output evaluation/reports/incremental_benchmark.json
```

### Empirical Benchmark: Baseline vs Incremental Processing

Tripwire implements content-fingerprint incremental processing on top of `DedupTracker` (`data/processed/dedup_tracker.db`). If a source URL's whitespace-normalized content hash has not changed, LLM extraction and vector embedding generation are bypassed completely (`UNCHANGED` → `SKIP`), and processed data is preserved without unnecessary writes. When content changes, it is reprocessed and the stored SHA-256 fingerprint is updated (`CHANGED` → `PROCESS`).

The benchmark suite (`evaluation/incremental_benchmark.py`) empirically measures resource consumption and latency across 20 evaluation records:

| Metric | Baseline (Full Processing) | Incremental (Unchanged Content) | Incremental (50% Updates) | Measured Efficiency Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Records Evaluated** | 20 | 20 | 20 | — |
| **Documents Skipped** | 0 (0.0%) | **20 (100.0%)** | 10 (50.0%) | **100% skip rate on unchanged** |
| **LLM Calls Executed** | 20 | **0 (20 saved)** | 10 (10 saved) | **100% LLM cost avoidance** |
| **Embeddings Generated** | 20 | **0 (20 saved)** | 10 (10 saved) | **100% embedding calls saved** |
| **Wall Execution Time** | 0.1259s | **0.0080s** | 0.0278s | **15.75x speedup (93.65% latency reduction)** |
| **Throughput (rec/sec)** | 158.8 rec/s | **2501.8 rec/s** | 718.3 rec/s | **+1475% throughput increase** |

Prometheus metrics exposed at `/metrics`:
* `graphone_incremental_total`: Total documents evaluated by the incremental decision engine.
* `graphone_incremental_new`: Count of newly discovered URLs.
* `graphone_incremental_changed`: Count of existing URLs with updated content fingerprints.
* `graphone_incremental_unchanged`: Count of existing URLs with identical content fingerprints.
* `graphone_incremental_skipped`: Total document reprocessings skipped.
* `graphone_llm_calls_saved_total`: Cumulative LLM extraction invocations saved.
* `graphone_embedding_calls_saved_total`: Cumulative dense vector embeddings saved.

For detailed metrics descriptions, scientific methodologies, and report specifications, refer to:
* [`evaluation/README.md`](evaluation/README.md)
* [`evaluation/llm/README.md`](evaluation/llm/README.md)
* [`evaluation/resolution/README.md`](evaluation/resolution/README.md)
