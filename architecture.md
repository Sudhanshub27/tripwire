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

---

## 6. Phase 5: Graph + Vector Hybrid Retrieval Engine

### Overview & Motivation
Standard retrieval in RAG and entity search systems typically relies on either isolated dense vector similarity or isolated structured graph traversal. Both approaches have distinct failure modes:
1. **Vector-Only Failure Mode**: Fails to surface connected entities that do not share semantic lexical or embedding overlap with the query string (e.g. searching for a startup does not surface its newly released product unless explicitly mentioned in the text).
2. **Graph-Only Failure Mode**: Requires an exact known root node to begin traversal and cannot perform natural-language semantic discovery across unstructured text.

Phase 5 introduces a unified **Graph + Vector Hybrid Retrieval System** that bridges dense embedding similarity search (LanceDB / TF vectors) with bounded Knowledge Graph multi-hop exploration (BFS query engine).

```
[ User Natural Language Query ]
               │
               ▼
   [ Vector Retrieval Stage ]
   (LanceDB Dense Similarity Search)
               │
               ├─────────────────────────────────────────┐
               ▼ (Top-K Seeds)                           │
   [ Entity Identity Bridge ]                            │ (Vector Candidates)
   (Map Vector IDs/Titles ➔ Graph Nodes)                 │
               │                                         │
               ▼                                         │
   [ Graph Expansion Stage ]                             │
   (Bounded Multi-Hop BFS Traversal)                     │
   (Direction, Edge Type, Depth Limits)                  │
               │                                         │
               ├─────────────────────────────────────────┘
               ▼
   [ Candidate Merging & Deduplication ]
   (Stable-ID Mapping: node-{type}-{slug})
   (Sources Classification: VECTOR, GRAPH, BOTH)
               │
               ▼
   [ Deterministic Ranking & Scoring ]
   - Weighted Linear Fusion: S = w_v * S_v + w_g * S_g
   - Reciprocal Rank Fusion: RRF(c) = w_v/(k+R_v) + w_g/(k+R_g)
   - Tie-breaking: (-score, -vec_score, -graph_score, id)
               │
               ▼
   [ Structured Results & Provenance Evidence ]
   (Candidates + Scores + Path Provenance + Explanation)
```

### Retrieval Stages

#### Stage 1: Vector Retrieval
- Computes dense normalized embedding vectors for the natural language query.
- Retrieves the top $N$ candidates (`vector_limit`, default: 10, max: 100) from the vector store (`VectorStoreManager`).
- Associates cosine similarity score $S_{\text{vector}} \in [0.0, 1.0]$ and records direct vector hits.

#### Stage 2: Graph Expansion from Vector Seeds
- Vector candidates are mapped to canonical `GraphNode` instances in the Knowledge Graph via exact ID, normalized slug, or canonical name lookup.
- Seeds with no graph representation or zero incident edges remain safely as `VECTOR`-only candidates without error.
- From valid graph seeds, level-by-level BFS multi-hop traversal executes up to `graph_depth` (1–5 hops, default: 1 or 2).
- Edge traversal respects `direction` (`OUTGOING`, `INCOMING`, `BOTH`) and optional `edge_types` filters (e.g. `['PRODUCES']`).
- To prevent infinite loops and redundant exploration, a `visited_nodes` set enforces safe cycle termination.
- Results are bounded by `max_graph_candidates` (default: 20, max: 100). If the expansion limit is reached, `is_truncated` is flagged.

#### Stage 3: Candidate Merging & Stable-ID Deduplication
- Candidates from both retrieval paths are merged into a unified set using canonical stable identifiers (`node-{entity_type}-{slug}`).
- Candidates are tagged with their discovery source:
  - `VECTOR`: Surfaced solely by semantic similarity search.
  - `GRAPH`: Discovered solely via relational graph expansion from a seed.
  - `BOTH`: Surfaced by semantic search AND reached through graph relationships.
- Relationship provenance is preserved: each graph candidate carries a structured path description (e.g. `Anthropic --[PRODUCES (0.95)]--> Claude 3.5 Sonnet`).

### Ranking Strategy & Score Semantics

#### 1. Graph Relevance & Distance Decay
Graph relevance for an entity reached at distance $d \ge 1$ from seed $s$ is computed via hyperbolic distance decay and cumulative edge confidence:
$$\text{decay}(d) = \frac{1.0}{1.0 + 0.5 \times d}$$
$$S_{\text{graph}} = S_{\text{seed\_vector}} \times \text{decay}(d) \times \prod_{e \in \text{path}} \text{confidence}(e)$$
Entities directly connected ($d=1$) to high-scoring vector seeds with high edge confidence receive high graph relevance; entities multiple hops away decay proportionally.

#### 2. Weighted Linear Fusion (Default)
Normalized weights $w_v, w_g$ ($w_v + w_g = 1.0$, default $0.6 / 0.4$):
$$S_{\text{final}} = w_v \cdot S_{\text{vector}} + w_g \cdot S_{\text{graph}}$$
- For `BOTH`: receives combined score benefiting from both vector and graph relevance.
- For `VECTOR`: $S_{\text{final}} = w_v \cdot S_{\text{vector}}$.
- For `GRAPH`: $S_{\text{final}} = w_g \cdot S_{\text{graph}}$.

#### 3. Reciprocal Rank Fusion (RRF)
Supported via `ranking_strategy = "rrf"` with smoothing constant $k$ (default: 60):
$$\text{RRF}(c) = \frac{w_v}{k + R_v(c)} + \frac{w_g}{k + R_g(c)}$$
where $R_v(c), R_g(c)$ are 1-based ranks within vector and graph candidate lists (evaluating to $0$ if not surfaced in that pathway).

#### 4. Deterministic Tie-Breaking
Ranking order is 100% deterministic for identical inputs. Ties are resolved by the tuple:
$$\text{sort\_key} = \left(-\text{round}(S_{\text{final}}, 6), -\text{round}(S_{\text{vector}}, 6), -\text{round}(S_{\text{graph}}, 6), \text{candidate\_id}\right)$$

### API Specifications

#### `POST /retrieval/hybrid` and `/api/retrieval/hybrid`
Executes hybrid vector + graph retrieval with full parameter validation and telemetry.

**Request Schema (`HybridRetrievalRequest`):**
```json
{
  "query": "Anthropic AI frontier research and Claude reasoning models",
  "candidate_type": "all",
  "vector_limit": 10,
  "graph_depth": 2,
  "max_graph_candidates": 20,
  "edge_types": ["PRODUCES", "POSTED_BY"],
  "direction": "both",
  "vector_weight": 0.6,
  "graph_weight": 0.4,
  "ranking_strategy": "weighted",
  "rrf_k": 60,
  "min_score": 0.0,
  "limit": 10
}
```

**Response Schema (`HybridRetrievalResponse`):**
```json
{
  "query": "Anthropic AI frontier research and Claude reasoning models",
  "candidate_type": "all",
  "total_candidates": 2,
  "vector_candidates_count": 2,
  "graph_candidates_count": 2,
  "merged_candidates_count": 2,
  "returned_count": 2,
  "execution_ms": 0.170,
  "is_truncated": false,
  "ranking_strategy": "weighted",
  "results": [
    {
      "id": "node-startup-anthropic",
      "candidate_type": "startup",
      "title": "Anthropic",
      "vector_score": 0.8800,
      "graph_score": null,
      "graph_distance": null,
      "path_evidence": [],
      "sources": ["VECTOR"],
      "source": "VECTOR",
      "final_score": 0.5280,
      "ranking_explanation": "Vector-only match: Vector similarity 0.8800 (weighted contribution: 0.5280, weight: 0.60)...",
      "payload": { "name": "Anthropic", "stage": "Series E", "location": "San Francisco, CA" }
    },
    {
      "id": "node-product-claude",
      "candidate_type": "product",
      "title": "Claude 3.5 Sonnet",
      "vector_score": 0.8200,
      "graph_score": 0.5573,
      "graph_distance": 1,
      "path_evidence": ["Anthropic --[PRODUCES (0.95)]--> Claude 3.5 Sonnet"],
      "sources": ["VECTOR", "GRAPH"],
      "source": "BOTH",
      "final_score": 0.7149,
      "ranking_explanation": "Weighted fusion (w_v=0.60, w_g=0.40): Vector similarity 0.8200 (0.4920) + Graph score 0.5573 (0.2229) at distance 1 -> Final score 0.7149.",
      "payload": { "name": "Claude 3.5 Sonnet", "maker": "Anthropic", "pricing": "Paid" }
    }
  ]
}
```

### Empirical Evaluation & Measured Results

Evaluated using `evaluation/retrieval_benchmark.py` across the standardized ground-truth benchmark dataset (`evaluation/datasets/retrieval_eval_dataset.json`):

| Metric | Vector-Only Baseline | Graph-Only Baseline | Hybrid Retrieval (Vector + Graph) | Delta vs Vector |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1** | 75.0% | 87.5% | **75.0%** | $+0.0\%$ |
| **Recall@3** | 87.5% | 100.0% | **93.8%** | $+6.3\%$ |
| **Recall@5** | 93.8% | 100.0% | **100.0%** | **$+6.2\%$** |
| **Recall@10** | 93.8% | 100.0% | **100.0%** | **$+6.2\%$** |
| **Precision@5** | 22.5% | 25.0% | **25.0%** | $+2.5\%$ |
| **MRR** | 0.9167 | 1.0000 | **0.9167** | $+0.0000$ |
| **NDCG@5** | 0.8922 | 1.0000 | **0.9172** | **$+0.0250$** |
| **MAP@5** | 0.8167 | 1.0000 | **0.8583** | **$+0.0416$** |
| **Mean Latency** | 0.120 ms | 0.030 ms | **0.170 ms** | $+0.050$ ms |
| **p95 Latency** | 0.230 ms | 0.070 ms | **0.300 ms** | $+0.070$ ms |

**Key Findings:**
1. **Recall Improvement**: Hybrid retrieval achieves **100.0% Recall@5** (compared to 93.8% for vector-only), successfully uncovering connected entities (such as products produced by a startup or papers linked to a method) where pure vector search missed the secondary entity.
2. **NDCG & MAP Gains**: Ranking quality improved across the board: NDCG@5 increased from **0.8922 to 0.9172** and MAP@5 increased from **0.8167 to 0.8583**.
3. **Sub-Millisecond Overhead**: Graph expansion and candidate merging add only **0.050 ms** average latency overhead over pure vector search.

### Observability Metrics
Integrated with `src/observability/metrics.py`:
- `graphone_hybrid_queries_total`: Total hybrid queries executed.
- `graphone_hybrid_query_errors_total`: Total queries returning errors.
- `graphone_hybrid_query_avg_latency_seconds`: Average hybrid execution latency.
- `graphone_hybrid_vector_candidates_total`: Cumulative vector candidates evaluated.
- `graphone_hybrid_graph_candidates_total`: Cumulative graph candidates discovered.
- `graphone_hybrid_merged_candidates_total`: Cumulative merged candidates before limits.
- `graphone_hybrid_final_results_total`: Total ranked candidates returned in payloads.
- `graphone_hybrid_graph_truncated_total`: Total queries where graph expansion reached `max_graph_candidates`.

### Known Limitations & Future Work
1. **In-Memory Graph Adjacency**: Current graph indexing operates in-process via optimized adjacency dictionaries. For datasets exceeding $10^7$ edges, migration to an external graph store (Neo4j / Memgraph) or LanceDB Graph extension is recommended.
2. **Fixed Distance Decay**: The decay formula $\frac{1}{1 + 0.5 \cdot d}$ is static. Learning relationship-specific decay coefficients per edge type via ranking optimization is an area for future tuning.

---

## 7. Phase 6: Citation-Aware Grounded RAG Engine

### Overview & Motivation
Standard Retrieval-Augmented Generation (RAG) models frequently generate answers with hallucinated facts, nonexistent citations, or citations attached to irrelevant retrieved chunks. Furthermore, many systems conflate citation reference formatting with semantic entailment.

Phase 6 introduces a production-ready **Citation-Aware Grounded RAG Engine** with verifiable source provenance, evidence-aware abstention, and multi-stage citation validation.

```
[ Natural Language Question ]
              │
              ▼
  [ Stage 1: Hybrid Retrieval ]
  (Dense Vector Search + Knowledge Graph Multi-Hop BFS)
              │
              ▼
  [ Stage 2: Provenance Assembly ]
  (1-Indexed Chunks: doc_id, chunk_id, title, url, page, text)
              │
              ▼
  [ Stage 3: Evidence Policy & Sufficiency Check ]
  - Token coverage & retrieval score thresholds
  - If evidence missing/unrelated ➔ ABSTAIN ("I do not have sufficient verifiable evidence...")
              │
              ▼ (If Sufficient)
  [ Stage 4: Grounded Answer Generation ]
  (Synthesizes facts from sources with inline [1], [2] citations)
              │
              ▼
  [ Stage 5: Multi-Stage Citation Validation Layer ]
  ├─ Reference Validity: 1 <= [N] <= len(sources); no fabricated IDs
  ├─ Semantic Entailment: Verifies cited chunk actually supports the claim
  ├─ Conflict Detection: Identifies contradictions across sources
  └─ Fault Tolerance: Graceful handling of validator timeouts/exceptions
              │
              ▼
  [ Structured Response & Telemetry Export ]
  (answer, citations, sources, abstained, validation_status, execution metrics)
```

### Citation Format & Source Mapping
Answers format citations using bracketed index markers:
- Single citation: `[1]`
- Grouped joint citations: `[1][2]` or `[1, 2]` (supporting claims synthesized jointly from multiple chunks)
- Sequential claim citations: `Sentence A [1]. Sentence B [2].`

Every citation references a 1-based source index corresponding directly to a retrieved chunk in the `sources` list.

### Provenance Data Model
Every retrieved chunk carries full provenance metadata via `SourceChunkProvenance`:
- `index`: 1-based reference integer matching `[index]`.
- `source_id`: Canonical entity ID (`node-{type}-{slug}`).
- `document_id`: Stable document identifier (`doc-{type}-{slug}`).
- `chunk_id`: Stable chunk identifier (`{doc_id}:chunk-{N}`).
- `title`: Display title or canonical entity name.
- `source_url`: Verified origin URL or preprint link (`None` if unavailable; never fabricated).
- `page_number`: Document page number (`None` if unavailable).
- `section`: Document section header (`None` if unavailable).
- `text`: Verifiable content text excerpt.
- `retrieval_method`: Retrieval pathway (`VECTOR`, `GRAPH`, `BOTH`).
- `retrieval_score`: Deterministic hybrid retrieval score.
- `metadata`: Raw verified entity attributes.

### Multi-Stage Citation Validation
The validation layer (`CitationValidator`) evaluates generated answers and citations across several distinct stages:

#### 1. Reference Validity
Checks that every citation marker `[N]` in the text resolves to an actual retrieved chunk ($1 \le N \le \text{len}(\text{sources})$).
Rejects:
- Nonexistent citation markers (e.g. `[99]`).
- Fabricated source IDs or document IDs claiming identity not present in retrieved context.
Status on failure: `INVALID_REFERENCE`.

#### 2. Semantic Entailment & Claim Support
Distinguishes reference validity from semantic support. A citation ID may resolve to a valid chunk, but that chunk may fail to support the claim (e.g. citing an article about Claude when making a claim about solar panels).
- Checks that non-title factual terms and numbers (currency, dates, metrics) in the claim are attested in the cited source text.
- Supports joint multi-source verification: when citations `[1, 2]` are attached to a synthesized claim, support is verified against the joint union of the cited sources.
Status on failure: `UNSUPPORTED_CLAIM`.

#### 3. Conflicting Evidence Detection
Detects when multiple retrieved sources make conflicting assertions about the same entity (e.g., conflicting founding dates or funding amounts), adding diagnostic warnings to the response without crashing.

#### 4. Safe Error Handling
If an external validator (e.g., LLM provider) encounters a network timeout, quota limit, or parsing exception, the system safely falls back to deterministic heuristic validation and logs `VALIDATION_ERROR` telemetry rather than terminating the pipeline.

### Evidence-Aware Abstention Policy
The system refuses to generate speculative or ungrounded answers when evidence is inadequate.
- Evaluates query content token coverage against the retrieved sources.
- When coverage is low or query terms are absent:
  - Sets `abstained = True`.
  - Returns standard grounded abstention statement: `"I do not have sufficient verifiable evidence in the retrieved sources to answer this question. The available sources are insufficient to support a grounded factual answer."`.
  - Sets `validation_status = ValidationStatus.INSUFFICIENT_EVIDENCE`.
  - Sets `citations = []`.

### API Schemas

#### `POST /api/rag/query` & `POST /rag/query`
Request:
```json
{
  "query": "Where is Anthropic located and what foundation models does it develop?",
  "candidate_type": "all",
  "vector_limit": 10,
  "graph_depth": 2,
  "vector_weight": 0.6,
  "graph_weight": 0.4,
  "ranking_strategy": "weighted",
  "limit": 5,
  "semantic_validation": true
}
```
Response (`RAGQueryResponse`):
```json
{
  "query": "Where is Anthropic located and what foundation models does it develop?",
  "answer": "Anthropic is based in San Francisco, CA [1] and develops Claude 3.5 Sonnet [2].",
  "citations": [
    {
      "citation_id": 1,
      "source_id": "node-startup-anthropic",
      "chunk_id": "doc-startup-anthropic:chunk-0",
      "document_id": "doc-startup-anthropic",
      "claim_text": "Anthropic is based in San Francisco, CA [1]",
      "reference_valid": true,
      "claim_supported": true
    },
    {
      "citation_id": 2,
      "source_id": "node-product-claude",
      "chunk_id": "doc-product-claude:chunk-0",
      "document_id": "doc-product-claude",
      "claim_text": "and develops Claude 3.5 Sonnet [2].",
      "reference_valid": true,
      "claim_supported": true
    }
  ],
  "sources": [
    {
      "index": 1,
      "source_id": "node-startup-anthropic",
      "document_id": "doc-startup-anthropic",
      "chunk_id": "doc-startup-anthropic:chunk-0",
      "title": "Anthropic, Inc.",
      "source_url": "https://techcrunch.com/anthropic",
      "page_number": 1,
      "section": "Company Profile",
      "text": "Anthropic is an AI safety research company based in San Francisco, CA...",
      "retrieval_method": "VECTOR",
      "retrieval_score": 0.92
    }
  ],
  "abstained": false,
  "validation_status": "VALID",
  "execution_ms": 1.45,
  "generation_ms": 0.22,
  "validation_ms": 0.15
}
```

#### `POST /api/rag/validate` & `POST /rag/validate`
Validates external or pre-generated answers and citations against provided sources.

### Empirical Evaluation & Measured Results
Evaluated using `evaluation/citation_evaluator.py` across 12 canonical ground-truth test scenarios (`evaluation/datasets/citation_eval_dataset.json`):

| Metric | Measured Value | Denominator / Reference |
| :--- | :---: | :--- |
| **Citation Reference Validity** | **78.6%** | Valid citations / total citation markers (includes adversarial test cases) |
| **Claim-Level Citation Coverage** | **87.5%** | Cited factual claims / total factual claims |
| **Citation Faithfulness** | **72.7%** | Semantically entailed citations / valid citations |
| **Unsupported-Claim Rate** | **12.5%** | Unsupported claims / total factual claims |
| **Abstention Correctness** | **83.3%** | Correct abstentions / total test queries |
| **Validation Latency (Mean)** | **0.06 ms** | p50: 0.03 ms, p95: 0.22 ms |

### Observability Metrics
Integrated with `src/observability/metrics.py`:
- `graphone_rag_queries_total`: Total RAG queries executed.
- `graphone_rag_citation_validations_total`: Total citation markers validated.
- `graphone_rag_invalid_citations_total`: Total unresolvable citations detected.
- `graphone_rag_unsupported_claims_total`: Total unsupported factual claims flagged.
- `graphone_rag_abstentions_total`: Total queries where system abstained.
- `graphone_rag_validation_failures_total`: Total validator errors or timeouts.
- `graphone_rag_query_avg_latency_seconds`: Average RAG query latency.
- `graphone_rag_validation_avg_latency_seconds`: Average citation validation latency.

### Known Limitations
1. **Heuristic Token Overlap**: While the deterministic heuristic validator operates at sub-millisecond speeds (< 0.1 ms), nuanced linguistic inference (such as double negations or complex counterfactuals) benefits from an LLM-based validator when latency permits.
2. **Document Segmentation**: For lengthy technical papers, chunks currently represent executive summaries and abstracts; fine-grained PDF multi-page bounding box citations will be incorporated in future indexing expansions.
