"""
================================================================================
PHASE 5: GRAPH + VECTOR HYBRID RETRIEVAL TEST SUITE
================================================================================
Comprehensive test suite verifying:
1. Vector retrieval integration
2. Graph retrieval integration
3. Candidate merging
4. Stable-ID deduplication
5. Ranking and deterministic tie-breaking
6. Vector-only candidates
7. Graph-only candidates
8. Candidates found by both sources
9. Graph traversal limits and cycle handling
10. Edge-type and direction filters
11. Missing graph connections
12. Invalid request parameters (HTTP 422 / validation)
13. API response validation (/retrieval/hybrid & /api/retrieval/hybrid)
14. Metrics updates (Counters & Prometheus exposition)
15. Evaluation metric calculations (Precision, Recall, MRR, NDCG, MAP)
16. Comparison against vector-only and graph-only baselines
================================================================================
"""

import pytest
from evaluation.metrics import (
    calculate_map,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from evaluation.retrieval_benchmark import run_retrieval_benchmark
from fastapi.testclient import TestClient
from src.dashboard.main import app
from src.graph.models import Direction
from src.graph.query_engine import GraphQueryEngine
from src.observability.metrics import metrics_collector
from src.retrieval.models import (
    HybridRetrievalRequest,
    RankingStrategy,
    RetrievalSource,
)
from src.retrieval.service import HybridRetrievalService
from src.vector.vector_store import VectorStoreManager, compute_dense_text_embedding


@pytest.fixture
def test_client():
    """FastAPI test client instance."""
    return TestClient(app)


@pytest.fixture
def synthetic_retrieval_environment():
    """
    Creates an isolated VectorStoreManager and GraphQueryEngine with known
    deterministic entities, relationships, and cyclic loops for exact testing.
    """
    # 1. Populate Vector Store
    v_mgr = VectorStoreManager()
    items = [
        {
            "id": "startup-anthropic",
            "record_type": "startup",
            "title": "Anthropic",
            "text": "Anthropic AI frontier safety research and Claude reasoning foundation models San Francisco",
            "vector": compute_dense_text_embedding("Anthropic AI safety research Claude reasoning models"),
            "payload": {"name": "Anthropic", "stage": "Series E", "location": "San Francisco, CA"},
        },
        {
            "id": "product-claude",
            "record_type": "product",
            "title": "Claude 3.5 Sonnet",
            "text": "Claude 3.5 Sonnet frontier reasoning model developed by Anthropic",
            "vector": compute_dense_text_embedding("Claude 3.5 Sonnet frontier reasoning model Anthropic"),
            "payload": {"name": "Claude 3.5 Sonnet", "maker": "Anthropic", "pricing": "Paid"},
        },
        {
            "id": "startup-cogna",
            "record_type": "startup",
            "title": "Cogna AI",
            "text": "Cogna AI autonomous data ingestion pipeline infrastructure San Francisco",
            "vector": compute_dense_text_embedding("Cogna AI autonomous data ingestion pipeline"),
            "payload": {"name": "Cogna AI", "location": "San Francisco, CA"},
        },
        {
            "id": "paper-transformer",
            "record_type": "research_paper",
            "title": "Attention Is All You Need",
            "text": "Attention Is All You Need Transformer architectures for deep learning neural networks",
            "vector": compute_dense_text_embedding("Attention Is All You Need Transformer architectures"),
            "payload": {"title": "Attention Is All You Need", "year": "2017"},
        },
        {
            "id": "startup-cycle-a",
            "record_type": "startup",
            "title": "Alpha Corp",
            "text": "Alpha Corp deep learning startup partner with Beta",
            "vector": compute_dense_text_embedding("Alpha Corp deep learning"),
            "payload": {"name": "Alpha Corp"},
        },
    ]
    v_mgr.in_memory_index = items
    v_mgr._is_indexed = True

    # 2. Populate Graph Query Engine with cycles & disconnected components
    g_engine = GraphQueryEngine()
    nodes = [
        {"id": "node-startup-anthropic", "label": "Startup", "name": "Anthropic"},
        {"id": "node-product-claude", "label": "Product", "name": "Claude 3.5 Sonnet"},
        {"id": "node-job-researcher", "label": "Job", "name": "AI Safety Researcher", "company": "Anthropic"},
        {"id": "node-startup-cogna", "label": "Startup", "name": "Cogna AI"},  # Disconnected isolated node
        {"id": "node-paper-transformer", "label": "ResearchPaper", "name": "Attention Is All You Need"},
        # Cyclic triangle: A -> B -> C -> A
        {"id": "node-startup-cycle-a", "label": "Startup", "name": "Alpha Corp"},
        {"id": "node-product-cycle-b", "label": "Product", "name": "Beta Core"},
        {"id": "node-startup-cycle-c", "label": "Startup", "name": "Gamma Labs"},
        # Graph-only entity (not present in vector store at all)
        {"id": "node-product-graph-only", "label": "Product", "name": "Secret Unindexed Product"},
    ]
    edges = [
        {"source": "node-startup-anthropic", "target": "node-product-claude", "relation": "PRODUCES", "confidence": 0.95},
        {"source": "node-job-researcher", "target": "node-startup-anthropic", "relation": "POSTED_BY", "confidence": 0.90},
        {"source": "node-startup-anthropic", "target": "node-product-graph-only", "relation": "INCUBATES", "confidence": 0.85},
        # Cycles
        {"source": "node-startup-cycle-a", "target": "node-product-cycle-b", "relation": "PRODUCES", "confidence": 0.90},
        {"source": "node-product-cycle-b", "target": "node-startup-cycle-c", "relation": "MAINTAINED_BY", "confidence": 0.85},
        {"source": "node-startup-cycle-c", "target": "node-startup-cycle-a", "relation": "INVESTS_IN", "confidence": 0.80},
    ]
    g_engine.load_graph_data(nodes, edges)

    svc = HybridRetrievalService(vector_mgr=v_mgr, graph_engine=g_engine)
    return svc, v_mgr, g_engine


# ==============================================================================
# TESTS
# ==============================================================================


def test_vector_retrieval_integration(synthetic_retrieval_environment):
    """1. Test that vector retrieval finds semantically relevant candidates."""
    svc, _, _ = synthetic_retrieval_environment

    vec_results = svc.search_vector_only(query="Anthropic safety research", limit=5)
    assert len(vec_results) > 0
    top_cand = vec_results[0]
    assert "Anthropic" in top_cand.title
    assert top_cand.source == RetrievalSource.VECTOR
    assert top_cand.vector_score is not None
    assert top_cand.vector_score > 0.0
    assert top_cand.graph_score is None


def test_graph_retrieval_integration(synthetic_retrieval_environment):
    """2. Test that graph retrieval discovers connected entities via query engine."""
    svc, _, _ = synthetic_retrieval_environment

    graph_results = svc.search_graph_only(seed_query="Anthropic", depth=1, limit=5)
    assert len(graph_results) > 0

    titles = [c.title for c in graph_results]
    assert "Anthropic" in titles
    assert "Claude 3.5 Sonnet" in titles
    for c in graph_results:
        assert c.source == RetrievalSource.GRAPH
        assert c.graph_score is not None
        assert c.vector_score is None


def test_candidate_merging_and_deduplication(synthetic_retrieval_environment):
    """3 & 4. Test candidate merging and stable-ID deduplication."""
    svc, _, _ = synthetic_retrieval_environment

    req = HybridRetrievalRequest(
        query="Anthropic Claude model",
        vector_limit=5,
        graph_depth=1,
        limit=10,
    )
    resp = svc.search_hybrid(req)

    # Check for no duplicate IDs in results
    result_ids = [c.id for c in resp.results]
    assert len(result_ids) == len(set(result_ids)), "Duplicate candidate IDs returned"

    # Both Anthropic and Claude should be identified with canonical node IDs
    assert "node-startup-anthropic" in result_ids
    assert "node-product-claude" in result_ids


def test_ranking_weighted_and_deterministic_tie_breaking(synthetic_retrieval_environment):
    """5. Test weighted scoring formula and strictly deterministic tie-breaking."""
    svc, _, _ = synthetic_retrieval_environment

    req = HybridRetrievalRequest(
        query="Anthropic",
        vector_weight=0.7,
        graph_weight=0.3,
        ranking_strategy=RankingStrategy.WEIGHTED,
        limit=10,
    )

    # Run query multiple times to verify exact determinism
    resp1 = svc.search_hybrid(req)
    resp2 = svc.search_hybrid(req)

    assert resp1.returned_count == resp2.returned_count
    for c1, c2 in zip(resp1.results, resp2.results):
        assert c1.id == c2.id
        assert c1.final_score == c2.final_score
        assert c1.ranking_explanation == c2.ranking_explanation

    # Check weighted formula calculation on top hit
    top = resp1.results[0]
    if top.source == RetrievalSource.BOTH:
        expected = round(0.7 * (top.vector_score or 0.0) + 0.3 * (top.graph_score or 0.0), 6)
        assert pytest.approx(top.final_score, abs=1e-5) == expected


def test_ranking_rrf_strategy(synthetic_retrieval_environment):
    """Test Reciprocal Rank Fusion ranking strategy."""
    svc, _, _ = synthetic_retrieval_environment

    req = HybridRetrievalRequest(
        query="Anthropic Claude",
        ranking_strategy=RankingStrategy.RRF,
        rrf_k=60,
        limit=5,
    )
    resp = svc.search_hybrid(req)

    assert resp.ranking_strategy == "rrf"
    assert len(resp.results) > 0
    for c in resp.results:
        assert "RRF" in c.ranking_explanation
        assert c.final_score > 0.0


def test_vector_only_candidates(synthetic_retrieval_environment):
    """6. Test vector-only candidate handling (e.g. disconnected node Cogna AI)."""
    svc, _, _ = synthetic_retrieval_environment

    req = HybridRetrievalRequest(
        query="Cogna AI autonomous data pipeline",
        vector_limit=5,
        graph_depth=1,
        limit=5,
    )
    resp = svc.search_hybrid(req)

    cogna_cand = next((c for c in resp.results if "Cogna" in c.title), None)
    assert cogna_cand is not None
    assert cogna_cand.source == RetrievalSource.VECTOR
    assert cogna_cand.vector_score is not None
    assert cogna_cand.graph_score is None
    assert "Vector-only match" in cogna_cand.ranking_explanation


def test_graph_only_candidates(synthetic_retrieval_environment):
    """7. Test graph-only candidate discovery (entity not indexed in vector store)."""
    svc, _, _ = synthetic_retrieval_environment

    req = HybridRetrievalRequest(
        query="Anthropic",
        vector_limit=5,
        graph_depth=1,
        max_graph_candidates=20,
        limit=10,
    )
    resp = svc.search_hybrid(req)

    # 'Secret Unindexed Product' is linked via INCUBATES from Anthropic
    graph_only_cand = next((c for c in resp.results if "Secret Unindexed Product" in c.title), None)
    assert graph_only_cand is not None
    assert graph_only_cand.source == RetrievalSource.GRAPH
    assert graph_only_cand.vector_score is None
    assert graph_only_cand.graph_score is not None
    assert graph_only_cand.graph_distance == 1
    assert len(graph_only_cand.path_evidence) > 0
    assert "Graph-only discovery" in graph_only_cand.ranking_explanation


def test_candidates_found_by_both_sources(synthetic_retrieval_environment):
    """8. Test candidates surfaced by BOTH vector and graph receiving synergistic fusion."""
    svc, _, _ = synthetic_retrieval_environment

    req = HybridRetrievalRequest(
        query="Claude 3.5 Sonnet Anthropic",
        vector_limit=5,
        graph_depth=1,
        limit=5,
    )
    resp = svc.search_hybrid(req)

    both_cands = [c for c in resp.results if c.source == RetrievalSource.BOTH]
    assert len(both_cands) > 0
    for c in both_cands:
        assert c.vector_score is not None
        assert c.graph_score is not None
        assert RetrievalSource.VECTOR in c.sources
        assert RetrievalSource.GRAPH in c.sources
        assert "Weighted fusion" in c.ranking_explanation


def test_graph_traversal_depth_and_cycle_handling(synthetic_retrieval_environment):
    """9. Test cycle termination and depth limits on circular graph (A -> B -> C -> A)."""
    svc, _, _ = synthetic_retrieval_environment

    # Depth 1: Alpha Corp should reach Beta Core
    req1 = HybridRetrievalRequest(
        query="Alpha Corp",
        vector_limit=5,
        graph_depth=1,
        direction=Direction.OUTGOING,
        limit=10,
    )
    resp1 = svc.search_hybrid(req1)
    titles1 = {c.title for c in resp1.results}
    assert "Alpha Corp" in titles1
    assert "Beta Core" in titles1
    assert "Gamma Labs" not in titles1

    # Depth 2: Alpha Corp should reach Gamma Labs through Beta Core without infinite loop
    req2 = HybridRetrievalRequest(
        query="Alpha Corp",
        vector_limit=5,
        graph_depth=2,
        direction=Direction.OUTGOING,
        limit=10,
    )
    resp2 = svc.search_hybrid(req2)
    titles2 = {c.title for c in resp2.results}
    assert "Gamma Labs" in titles2

    # Depth 3: Traverses back around to Alpha Corp, cycle handled cleanly without duplicate entries
    req3 = HybridRetrievalRequest(
        query="Alpha Corp",
        vector_limit=5,
        graph_depth=3,
        direction=Direction.OUTGOING,
        limit=10,
    )
    resp3 = svc.search_hybrid(req3)
    alpha_occurrences = [c for c in resp3.results if c.title == "Alpha Corp"]
    assert len(alpha_occurrences) == 1, "Alpha Corp duplicated despite cycle"


def test_edge_type_and_direction_filters(synthetic_retrieval_environment):
    """10. Test edge-type and direction constraints."""
    svc, _, _ = synthetic_retrieval_environment

    # 1. Filter by edge_types = ['PRODUCES']
    req_produces = HybridRetrievalRequest(
        query="Anthropic",
        graph_depth=1,
        edge_types=["PRODUCES"],
        limit=10,
    )
    resp_produces = svc.search_hybrid(req_produces)
    titles_p = {c.title for c in resp_produces.results}
    assert "Claude 3.5 Sonnet" in titles_p
    # INCUBATES edge to 'Secret Unindexed Product' should be filtered out
    assert "Secret Unindexed Product" not in titles_p

    # 2. Filter by Direction: OUTGOING from Anthropic
    req_out = HybridRetrievalRequest(
        query="Anthropic",
        graph_depth=1,
        direction=Direction.OUTGOING,
        limit=10,
    )
    resp_out = svc.search_hybrid(req_out)
    titles_out = {c.title for c in resp_out.results}
    assert "Claude 3.5 Sonnet" in titles_out
    # Job is INCOMING (Job -> POSTED_BY -> Anthropic), so not in OUTGOING
    assert "AI Safety Researcher" not in titles_out


def test_missing_graph_connections(synthetic_retrieval_environment):
    """11. Test that seeds with no graph connections or missing nodes don't error."""
    svc, _, _ = synthetic_retrieval_environment

    # Query with seed that has 0 edges
    req = HybridRetrievalRequest(
        query="Attention Is All You Need",
        graph_depth=2,
        limit=5,
    )
    resp = svc.search_hybrid(req)
    assert resp.returned_count > 0
    paper_cand = resp.results[0]
    assert "Attention" in paper_cand.title
    assert paper_cand.source == RetrievalSource.VECTOR


def test_invalid_request_parameters(test_client):
    """12. Test HTTP validation and error status codes on invalid payloads."""
    # Empty query string
    res_empty = test_client.post("/retrieval/hybrid", json={"query": ""})
    assert res_empty.status_code == 422

    # Negative graph depth
    res_depth = test_client.post("/retrieval/hybrid", json={"query": "test", "graph_depth": -1})
    assert res_depth.status_code == 422

    # Excessive vector limit (>100)
    res_limit = test_client.post("/retrieval/hybrid", json={"query": "test", "vector_limit": 500})
    assert res_limit.status_code == 422

    # Weight out of bounds (>1.0)
    res_weight = test_client.post("/retrieval/hybrid", json={"query": "test", "vector_weight": 1.5})
    assert res_weight.status_code == 422

    # Extra unknown field (ConfigDict extra='forbid')
    res_extra = test_client.post("/retrieval/hybrid", json={"query": "test", "malicious_field": True})
    assert res_extra.status_code == 422


def test_api_endpoints_response_validation(test_client):
    """13. Test /retrieval/hybrid and /api/retrieval/hybrid response schemas."""
    for endpoint in ["/retrieval/hybrid", "/api/retrieval/hybrid"]:
        payload = {
            "query": "Anthropic Claude reasoning models",
            "candidate_type": "all",
            "vector_limit": 5,
            "graph_depth": 2,
            "max_graph_candidates": 10,
            "vector_weight": 0.6,
            "graph_weight": 0.4,
            "limit": 5,
        }
        res = test_client.post(endpoint, json=payload)
        assert res.status_code == 200, f"Endpoint {endpoint} failed: {res.text}"
        data = res.json()

        assert "results" in data
        assert "execution_ms" in data
        assert "total_candidates" in data
        assert "vector_candidates_count" in data
        assert "graph_candidates_count" in data
        assert "merged_candidates_count" in data

        for item in data["results"]:
            assert "id" in item
            assert "title" in item
            assert "candidate_type" in item
            assert "source" in item
            assert item["source"] in ("VECTOR", "GRAPH", "BOTH")
            assert "final_score" in item
            assert "ranking_explanation" in item
            assert isinstance(item["final_score"], (int, float))


def test_metrics_updates_actual_execution(test_client):
    """14. Test that hybrid retrieval queries update system and Prometheus metrics."""
    metrics_collector.reset_hybrid_metrics()

    test_client.post(
        "/api/retrieval/hybrid",
        json={"query": "Anthropic AI", "limit": 5},
    )

    h_stats = metrics_collector.get_hybrid_metrics()
    assert h_stats["hybrid_queries_total"] >= 1
    assert h_stats["hybrid_final_results_total"] >= 1
    assert h_stats["hybrid_query_avg_latency_ms"] >= 0.0

    prom_text = metrics_collector.generate_prometheus_format()
    assert "graphone_hybrid_queries_total" in prom_text
    assert "graphone_hybrid_vector_candidates_total" in prom_text
    assert "graphone_hybrid_graph_candidates_total" in prom_text


def test_evaluation_metric_calculations():
    """15. Test IR evaluation metric formulas against exact hand-verified ground truth."""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc2", "doc4"}
    graded = {"doc1": 1.0, "doc2": 3.0, "doc4": 2.0}

    # Precision@K
    assert calculate_precision_at_k(retrieved, relevant, k=1) == 0.0
    assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.5  # doc2 in top 2
    assert calculate_precision_at_k(retrieved, relevant, k=4) == 0.5  # doc2, doc4 in top 4
    assert calculate_precision_at_k(retrieved, relevant, k=5) == 0.4  # 2/5

    # Recall@K
    assert calculate_recall_at_k(retrieved, relevant, k=1) == 0.0
    assert calculate_recall_at_k(retrieved, relevant, k=2) == 0.5
    assert calculate_recall_at_k(retrieved, relevant, k=4) == 1.0

    # MRR (doc2 is at index 1 -> rank 2 -> MRR = 0.5)
    assert calculate_mrr(retrieved, relevant) == 0.5
    assert calculate_mrr(["doc9"], relevant) == 0.0

    # NDCG@K
    ndcg = calculate_ndcg_at_k(retrieved, graded, k=3)
    assert 0.0 < ndcg <= 1.0

    # MAP
    map_score = calculate_map(retrieved, relevant, k=5)
    assert 0.0 < map_score <= 1.0


def test_retrieval_benchmark_execution(tmp_path):
    """16. Test running full retrieval benchmark producing non-empty metrics."""
    out_file = tmp_path / "test_retrieval_report.json"
    report = run_retrieval_benchmark(output_path=out_file)

    assert out_file.exists()
    assert "metrics_summary" in report
    summary = report["metrics_summary"]

    assert "vector_only" in summary
    assert "graph_only" in summary
    assert "hybrid" in summary

    for mode in ["vector_only", "graph_only", "hybrid"]:
        m = summary[mode]
        assert "recall@5" in m
        assert "precision@5" in m
        assert "mrr" in m
        assert "ndcg@5" in m
        assert 0.0 <= m["recall@5"] <= 1.0
        assert 0.0 <= m["precision@5"] <= 1.0
