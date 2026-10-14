# Graphone Pipeline Architecture & Scaling Specification

## 1. System Architecture & Scale Strategy (Demo to 500,000+ Records)

### Current Architecture Overview
The current implementation utilizes a modular, asynchronous architecture built on Python 3.11+ `asyncio`, Pydantic v2, and `aiohttp`. The pipeline decouples data acquisition, structured LLM extraction, entity resolution, validation, and storage.

```
                                  [ Async Ingestion Engine ]
                                               │
           ┌───────────────────────────────────┼───────────────────────────────────┐
           ▼                                   ▼                                   ▼
    [ ArXiv / PWC API ]               [ RSS / News Crawler ]             [ Directory Scraper ]
           │                                   │                                   │
           └───────────────────────────────────┼───────────────────────────────────┘
                                               ▼
                                 [ LLM Multi-Tier Fallback Chain ]
                                 (Gemini ➔ Groq ➔ DeepSeek ➔ Rule)
                                               │
                                               ▼
                                 [ Entity Resolution Engine ]
                              (Exact ➔ Normalized ➔ Fuzzy Threshold)
                                               │
                                               ▼
                                  [ Data Persistence & Sync ]
                              (JSONL ➔ SQLite / GSheets Exporter)
```

### Zero-Code Scaling Strategy (500k+ Records)
The architecture is designed to scale horizontally from sample datasets ($10^2$ records) to production scale ($500,000+$ records) without requiring structural code modifications:

1. **Stateless Scrapers & Extractors**:
   - `AsyncScraper` and `LLMProvider` instances maintain no internal state between items.
   - Work units are defined as immutable serialized JSON payloads containing `source_url`, `raw_payload`, and `target_schema`.

2. **Decoupled Task Queue Upgrade Path (`asyncio.Queue` ➔ `Redis` / `Celery`)**:
   - **Demo Scale**: In-process `asyncio.Queue` manages producer-consumer concurrency across local coroutines.
   - **Production Scale**: Swap the in-process queue for a distributed message broker (Redis Streams or RabbitMQ) and task runners (Celery or ARQ). Stateless worker containers run `src.main` worker tasks across distributed Kubernetes pods without worker-to-worker state coordination.

3. **Config-Driven Concurrency Decoupling**:
   - Throughput is managed via externalized environment configuration (`settings.py`):
     - `MAX_CONCURRENT_SCRAPES` controls network worker concurrency.
     - `MAX_CONCURRENT_LLM_CALLS` manages provider API quota limits.
     - `RATE_LIMIT_PER_MINUTE` controls domain-level token bucket refill rates.

---

## 2. Rate Limit & Payload Management (Handling 413s & 429s)

### HTTP 413 (Payload Too Large) & LLM Context Optimization
Large HTML DOMs or long text documents exceeding model context limits are managed via a multi-stage chunking pipeline:

1. **Semantic Boundary Splitting**:
   - Text is not split arbitrarily by byte length. The parser splits payloads at semantic HTML boundaries (`<article>`, `<section>`, `<div>`, `<h1>`-`<h6>`) or paragraph breaks (`\n\n`) to preserve contextual coherence for LLM extraction.
2. **Pre-flight Token Estimation & Safety Margins**:
   - Document chunks undergo pre-flight token estimation using `tiktoken` (`cl100k_base`).
   - Hard input caps are set at **80% of max provider window sizes** (e.g., 6,400 tokens for an 8k context window). The remaining 20% margin is reserved for system instructions, Pydantic JSON schema constraints, and model output generation.
3. **Recursive Chunk Summarization**:
   - If a document exceeds the safety threshold, chunks are extracted independently and merged via a map-reduce schema aggregation pass.

### HTTP 429 (Rate Limits) & Concurrency Control
To operate reliably across thousands of concurrent extractions without triggering IP bans or API key revocations:

```
    [ Incoming Request ]
             │
             ▼
   [ Per-Domain Token Bucket ]  ──(Quota Depleted?)──► [ Exponential Backoff + Jitter ]
             │                                                  │
             │ (Tokens Available)                               │ (Wait Expiry)
             ▼                                                  │
   [ HTTP Request Dispatch ] ◄──────────────────────────────────┘
             │
             ▼
   [ Response Inspection ]
             │
     ┌───────┴────────────────────────┐
     ▼                                ▼
[ HTTP 200 OK ]               [ HTTP 429 / 403 ]
     │                                │
     ▼                                ▼
[ Parse Data ]               [ Read Retry-After Header ]
                                      │
                                      ▼
                             [ Pause Worker Loop ]
```

1. **Per-Provider & Per-Domain Token Buckets**:
   - `AsyncScraper` enforces domain-level token bucket rate limiters, refilling tokens at `RATE_LIMIT_PER_MINUTE`.
2. **Exponential Backoff with Full Jitter**:
   - Transient failures (500, 502, 503, 504) and unhandled 429s trigger `tenacity`-backed retries:
     $$T_{\text{wait}} = \min\left(T_{\text{max}}, T_{\text{base}} \times 2^{\text{attempt}}\right) \pm \text{jitter}$$
   - Random jitter prevents the "thundering herd" problem when multiple concurrent workers retry simultaneously.
3. **Header-Driven Reset Suspension (`Retry-After` / `X-RateLimit-Reset`)**:
   - When 429 responses return explicit rate limit headers (`Retry-After` or `X-RateLimit-Reset`), backoff loops suspend execution precisely until the reset timestamp rather than guessing backoff intervals.
   - For unauthenticated external APIs (e.g., GitHub REST API), if reset duration exceeds 5.0 seconds, the worker logs an audit warning, sets fallback metadata, and continues execution without blocking the event loop.

---

## 3. Distributed Freshness & Deduplication Architecture

### Deduplication Engine (`dedup_tracker.py`)
To prevent redundant LLM extractions and duplicate data ingestion across repeated pipeline runs:

1. **Fingerprint Generation**:
   - Every entity payload generates a deterministic SHA-256 fingerprint:
     $$\text{Fingerprint} = \text{SHA-256}\left(\text{Normalized Title} \,||\, \text{Canonical URL} \,||\, \text{Published Date}\right)$$
2. **Storage Architecture (SQLite ➔ Redis Migration Path)**:
   - **Demo/Single-Node (SQLite)**: Uses SQLite WAL (Write-Ahead Logging) mode with an indexed `dedup_records` table storing `(fingerprint, source_name, processed_at)`.
   - **Production/Distributed (Redis Bloom Filter)**: Replaced with Redis Bloom Filters (`BF.ADD` / `BF.EXISTS`). Bloom filters provide $O(1)$ space-efficient probabilistic lookup across distributed ingestion nodes with a guaranteed false positive rate $<0.1\%$.

### 24-Hour Freshness Guarantee & Audit Heuristics
News articles and job postings enforce a strict 24-hour sliding freshness window:

```
                          [ Raw Item Metadata ]
                                    │
                                    ▼
                         [ Date Parser Module ]
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
[ Absolute ISO Parse ]    [ Relative Date Parse ]       [ HTML Metadata Heuristic ]
("2026-09-01T14:00Z")     ("3 hours ago" -> UTC)        (<time>, datePublished)
    │                               │                               │
    └───────────────────────────────┼───────────────────────────────┘
                                    ▼
                         [ Normalize to UTC ISO ]
                                    │
                                    ▼
                      [ Freshness Filter (< 24h?) ]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
             [ ACCEPTED ]                        [ REJECTED ]
       (Pass to LLM & Export)               (Audit Logged & Dropped)
```

- **Heuristic Fallback Hierarchy**:
  1. Absolute ISO 8601 parsing via `python-dateutil`.
  2. Relative time string conversion (`"2 hours ago"`, `"yesterday"` $\rightarrow$ UTC timestamp).
  3. Semantic HTML inspection (`<time datetime="...">`, `itemprop="datePublished"`).
  4. OpenGraph meta tag inspection (`article:published_time`).
  5. Fallback comparison against previous crawl execution timestamp stored in `data/processed/last_run.json`.
- **Audit Traceability**: Every heuristic fallback decision is logged to `data/processed/freshness_audit.log` for auditability.

---

## 4. Storage Strategy & Intelligence Graph Topology

### Relational Storage Justification (SQLite / PostgreSQL)
For structured entity records (Startups, Products, Research Papers, Jobs, News), a relational database engine (SQLite for local demo, PostgreSQL for production) is the optimal choice:

1. **Schema Compliance & Type Safety**:
   - Pydantic v2 schemas map 1:1 to relational table definitions with strict column constraints, foreign key references, and timestamp indexing.
2. **ACID Transaction Integrity**:
   - Atomic writes ensure idempotent pipeline executions: pipeline runs either commit a complete batch of validated entity records or roll back cleanly.
3. **JSONB Semi-Structured Flexibility**:
   - Fields such as `topics`, `authors`, and nested `SourceMetadata` are persisted using PostgreSQL `JSONB` columns, enabling GIN-indexed JSON query performance without sacrificing table normalization.

### Graph Database Topology (Neo4j / Amazon Neptune - Production Vision)
While relational storage excels at schema validation and tabular aggregation, modeling an **AI Ecosystem Intelligence Graph** in production requires modeling multi-hop relationship topologies.

```
       (Startup: OpenAI)
           │      │
   [:PRODUCES]  [:HIRES]
           │      │
           ▼      ▼
    (Product:  (Job: Research
     ChatGPT)   Scientist)
           │
     [:UTILIZES]
           │
           ▼
    (Paper: Attention Is All You Need)
```

#### Why a Graph Database for Production Scale?
1. **Multi-Hop Traversal Performance**:
   - Querying *"Find all startups that published a research paper cited by a competitor's product team"* requires 4+ relational JOINs, resulting in exponential $O(N^k)$ execution latency.
   - A Graph DB (Neo4j using Cypher) utilizes **index-free adjacency**, executing multi-hop traversals in $O(1)$ time per node hop regardless of total database size.
2. **Dynamic Relationship Properties**:
   - Relationships carry metadata (e.g., `(Startup)-[:RAISED_ROUND {amount: "$6.6B", date: "2024-10"}]->(FundingRound)`).

#### Storage Layer Architecture Trade-offs

| System Component | Relational DB (PostgreSQL) | Graph DB (Neo4j) | Key Trade-off Decision |
| :--- | :--- | :--- | :--- |
| **Primary Role** | Structured Entity Storage & Deduplication | Multi-hop Intelligence Graph Querying | Postgres ensures schema validation; Neo4j enables relationship discovery. |
| **Query Strengths** | Column filtering, aggregation, CRUD | Traversals (`Startup` ➔ `Product` ➔ `Paper` ➔ `Job`) | Use Postgres for ingestion pipeline targets; sync graph edges to Neo4j downstream. |
| **Write Performance** | High-throughput batch inserts | Moderate-throughput graph updates | Batch write relational entities first, stream graph delta edge updates via CDC. |

---

## 5. Entity Resolution 2.0 & Persistent Human Review Pipeline

### Overview
Entity Resolution 2.0 transforms entity deduplication into a production-grade resolution engine separating **deterministic candidate generation (blocking)** from **explainable multi-signal scoring** and **human-in-the-loop governance**.

```
                   [ Input Candidate Entity ]
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 1. Deterministic Candidate Generation        │
         │    (Token Inverted Index + 3-gram Blocking)  │
         │    O(N) -> O(K) reduction (K <= 20)          │
         └──────────────────────┬───────────────────────┘
                                │ Plausible Candidates
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 2. Multi-Signal Deterministic Scoring        │
         │    - Exact Alias Match        (1.00)         │
         │    - Normalized Suffix Match  (0.98)         │
         │    - Acronym Expansion Match  (0.95)         │
         │    - Normalized Token Ratio   [0.0, 1.0]     │
         │    - RapidFuzz Token Sort/Set [0.0, 1.0]     │
         └──────────────────────┬───────────────────────┘
                                │ Best Scored Candidate (Confidence C)
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 3. Explicit Decision Routing Thresholds      │
         └──────┬───────────────┼───────────────┬───────┘
                │               │               │
        C >= 0.85      0.65 <= C < 0.85       C < 0.65
                │               │               │
                ▼               ▼               ▼
         [ AUTO_MATCH ]  [ NEEDS_REVIEW ]  [ NO_MATCH ]
         Auto-merge into Persistent Review  Create New
         Canonical Entity Queue (SQLite)    Distinct Entity
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 4. Human Review & Audit Lifecycle            │
         │    - GET  /reviews?status=pending            │
         │    - POST /reviews/{id}/approve              │
         │    - POST /reviews/{id}/reject               │
         │    - Immutable SQL Audit Trail               │
         │    - Concurrency Lock & 409 Conflict Safety  │
         └──────────────────────────────────────────────┘
```

### Deterministic Confidence Scoring & Signal Model
Confidence scoring evaluates explainable signals without heuristic guesswork:
1. **Exact Alias Match** ($C = 1.00$): The raw string exactly matches a canonical name or known alias.
2. **Normalized String Match** ($C = 0.98$): Equality after stripping legal corporate suffixes (`Inc`, `Corp`, `LLC`, `PBC`, `GmbH`) and punctuation.
3. **Acronym Expansion Match** ($C = 0.95$): Single-token abbreviation expands to the first letters of canonical multi-word tokens (e.g. `AWS` $\rightarrow$ `Amazon Web Services`).
4. **Fuzzy RapidFuzz Token Sort / Set Ratio**: Token-level Levenshtein similarity accounting for word reordering.

### Threshold Rationale
- **`AUTO_MATCH` ($\ge 0.85$)**: High-confidence match threshold determined by zero-false-positive empirical validation. Guarantees safe automated merging.
- **`NEEDS_REVIEW` ($0.65 \le C < 0.85$)**: Ambiguous matches (e.g., sub-brands like `Scale Studio` vs `Scale AI`, `Synthesia Media` vs `Synthesia`). Persisted to `ReviewQueue` for human decision.
- **`NO_MATCH` ($< 0.65$)**: Unrelated entities or weak similarities. Automatically treated as distinct entities.

### Human Review Queue & Audit Architecture
- **Storage Layer**: ACID-compliant SQLite (`data/processed/resolution_reviews.db`) with an auto-synced JSONL mirror (`resolution_review_queue.jsonl`).
- **Atomic State Transitions**: Thread-safe updates condition on `WHERE id = ? AND (status = 'PENDING' OR decision = 'REVIEW')`, rejecting concurrent double-adjudication with HTTP 409 Conflict.
- **Immutable Audit Log**: Every lifecycle state change (`ENQUEUE`, `APPROVE`, `REJECT`) records an append-only entry in `resolution_review_audit` with reviewer identity, timestamp, previous state, new state, notes, and matching signals snapshot.

### Operational Metrics & Prometheus Exposition
Exposes Prometheus gauges, counters, and histograms via `src/observability/metrics.py`:
- `graphone_entity_resolution_total`: Cumulative resolutions processed.
- `graphone_entity_resolution_auto_matched`: Resolutions auto-merged.
- `graphone_entity_resolution_needs_review`: Resolutions enqueued for review.
- `graphone_entity_resolution_no_match`: Resolutions marked distinct.
- `graphone_entity_resolution_review_approved`: Human approved reviews.
- `graphone_entity_resolution_review_rejected`: Human rejected reviews.
- `graphone_entity_resolution_confidence_average`: Running average resolution confidence.
- `graphone_entity_resolution_queue_size`: Active pending review queue count.
- `graphone_entity_resolution_latency_seconds_average`: Running average resolution duration.

---

## 6. Graph Data Model & High-Performance Query Engine (Phase 4)

### Overview
Phase 4 introduces a production-oriented Graph Query Engine operating on top of the Knowledge Graph built from ingested entity streams. It features in-memory dual-adjacency indexing ($O(1)$ neighbor retrieval), cycle-safe Breadth-First Search (BFS) multi-hop traversals, shortest-path discovery, subgraph extraction, and strict resource bounds.

```
                  [ Knowledge Graph Data Model ]
         (JSONL: graph_nodes.jsonl & graph_edges.jsonl)
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ In-Memory Dual Adjacency Indexing Engine     │
         │ - Nodes Index:   ID -> GraphNode             │
         │ - Name Index:    normalized_name -> ID       │
         │ - Out Adjacency: source_id -> [GraphEdge]    │
         │ - In Adjacency:  target_id -> [GraphEdge]    │
         └──────────────────────┬───────────────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────┐
     ▼                          ▼                          ▼
[ Node / Neighbor Queries ] [ Multi-Hop BFS Traversal ] [ Shortest Path Discovery ]
- Exact ID & Name Lookup     - Depth-bounded (max <= 10) - Cycle-safe bidirectional BFS
- Directional Filtering      - Visited tracking          - Step-by-step Cypher tokens
- Edge & Node Label Filters  - Result limits (<= 500)    - Hop count & edge sequences
- Custom Property Matching   - Truncation detection      - Graceful unreachable return
```

### Graph Data Model Schema

#### Node Schema & Stable Identifiers
Nodes use stable prefixed identifiers: `node-{entity_type}-{id_or_slug}`.

| Node Type | Stable ID Format | Properties |
| :--- | :--- | :--- |
| **`Startup`** | `node-startup-{id}` | `name`, `location`, `funding`, `stage` |
| **`Product`** | `node-product-{id}` | `name`, `maker`, `pricing` |
| **`ResearchPaper`** | `node-paper-{id}` | `name` (title), `authors`, `journal`, `year` |
| **`Job`** | `node-job-{id}` | `name` (title), `company`, `salary` |
| **`News`** | `node-news-{id}` | `name` (headline), `published_at`, `sentiment` |

#### Edge Types & Relational Topology

| Edge Type | Source Node | Target Node | Semantics & Default Confidence |
| :--- | :--- | :--- | :--- |
| **`PRODUCES`** | `Startup` | `Product` | Company develops/offers product ($0.95 - 0.99$) |
| **`POSTED_BY`** | `Job` | `Startup` | Open employment position offered by company ($0.90$) |
| **`MENTIONS`** | `News` | `Startup` / `Product` | Industry press reporting on entity ($0.85$) |
| **`UTILIZES`** / **`CITES`** | `ResearchPaper` | `Product` / `Startup` | Technical methodology applied in product/research ($0.90$) |
| **`COMPETES_WITH`** | `Product` | `Product` | Competitive market alternative ($0.85$) |

### Query Engine Operations & Endpoints

| Endpoint | Method | Request Payload / Params | Capabilities & Guarantees |
| :--- | :---: | :--- | :--- |
| `/graph/nodes/{node_id}` | `GET` | `node_id: str` | $O(1)$ node retrieval by ID or normalized name. Returns 404 if missing. |
| `/graph/nodes/{node_id}/neighbors` | `GET` | `relation`, `node_label`, `direction`, `limit` | 1-hop exploration with edge type, node label, and direction filtering. |
| `/graph/query/traverse` | `POST` | `TraverseQueryRequest` (`root_id`, `max_depth`, `direction`, `limit`) | Bounded multi-hop traversal with cycle protection and limit truncation. |
| `/graph/query/path` | `POST` | `PathQueryRequest` (`from_node`, `to_node`, `max_depth`, `relation`) | Shortest path discovery returning ordered node/edge chains and steps. |
| `/graph/query/subgraph` | `POST` | `SubgraphQueryRequest` (`node_ids`, `hops`, `relation`, `limit`) | Induced neighborhood subgraph extraction around multiple seed nodes. |
| `/graph/query/relationships` | `POST` | `RelationshipQueryRequest` (`relation`, `source_label`, `target_label`) | Global triple pattern search with confidence thresholding. |
| `/graph/stats` | `GET` | None | Graph topological stats and query execution latency averages. |

### Query Safety & Resource Bounds
- **Cycle Termination Safety**: All traversals maintain `visited_nodes` sets. Cyclic relationships (e.g. $A \rightarrow B \rightarrow C \rightarrow A$) terminate safely without infinite loops or redundant node visits.
- **Configurable Clamping**:
  - `max_depth`: Clamped between $1$ and $10$ (`MAX_ALLOWED_DEPTH`). Defaults: $2$ (traverse), $5$ (path).
  - `limit`: Clamped between $1$ and $500$ (`MAX_RESULT_LIMIT`).
  - `is_truncated`: Set to `True` when result sets reach the requested limit.
- **Safe Fallbacks**: Missing entities return HTTP 404 or structured `QueryResult` with zero counts; disconnected components in path discovery safely return `found: False`.

### Measured Empirical Performance Benchmark

Benchmarks executed across deterministic synthetic graphs using `evaluation/graph_benchmark.py`:

| Scale | Topology (Nodes / Edges) | Node Lookup | 1-Hop Neighbors | 2-Hop Traverse | 3-Hop Traverse | Shortest Path | Subgraph Extract |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small** | 100 / 250 | $<0.001$ ms | 0.010 ms | 0.027 ms | 0.073 ms | 0.180 ms | 0.018 ms |
| **Medium** | 1,000 / 3,500 | $<0.001$ ms | 0.009 ms | 0.029 ms | 0.144 ms | 3.249 ms | 0.024 ms |
| **Large** | 5,000 / 20,000 | $<0.001$ ms | 0.011 ms | 0.043 ms | 0.183 ms | 4.840 ms | 0.029 ms |

### Observability & Prometheus Exposition
Integrated with `src/observability/metrics.py`:
- `graphone_graph_queries_total{query_type="..."}`: Cumulative queries by operation type.
- `graphone_graph_query_errors_total`: Cumulative query execution errors.
- `graphone_graph_query_avg_latency_seconds`: Running average query duration.
- `graphone_graph_nodes_visited_total`: Cumulative graph nodes explored during traversals.
- `graphone_graph_nodes_returned_total`: Cumulative graph nodes returned in response payloads.
- `graphone_graph_results_truncated_total`: Cumulative query results capped by limits.
