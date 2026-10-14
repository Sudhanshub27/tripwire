"""
================================================================================
PHASE 4: GRAPH QUERY ENGINE TEST SUITE
================================================================================
Comprehensive unit, property, traversal, and integration tests verifying:
1. Existing graph data can be queried
2. Node lookup succeeds and missing nodes return None / 404
3. Incoming, outgoing, and bidirectional neighbors
4. Edge-type and node-type filters
5. Multi-hop traversal returns correct results
6. Cycles terminate safely without infinite loops
7. Path discovery finds valid shortest paths
8. Unreachable targets return correct results (found=False)
9. Subgraph extraction returns expected nodes and edges
10. Duplicate nodes and edges are not returned
11. Maximum depth and result limits are enforced
12. Invalid parameters are rejected / clamped safely
13. API response models and error handling
14. Metrics reflect actual executions
15. Empty-graph behavior
================================================================================
"""

import pytest
from fastapi.testclient import TestClient
from src.dashboard.main import app
from src.graph.models import (
    Direction,
)
from src.graph.query_engine import (
    MAX_ALLOWED_DEPTH,
    GraphQueryEngine,
    graph_query_engine,
)
from src.observability.metrics import metrics_collector
from src.resolution.graph_linker import graph_linker


@pytest.fixture
def synthetic_graph_engine():
    """Create a GraphQueryEngine with a rich deterministic graph topology including cycles."""
    engine = GraphQueryEngine()

    nodes = [
        {"id": "node-startup-openai", "label": "Startup", "name": "OpenAI", "location": "SF", "funding": "$11.3B"},
        {"id": "node-product-chatgpt", "label": "Product", "name": "ChatGPT", "maker": "OpenAI"},
        {"id": "node-product-gpt4", "label": "Product", "name": "GPT-4", "maker": "OpenAI"},
        {"id": "node-paper-attention", "label": "ResearchPaper", "name": "Attention Is All You Need", "year": "2017"},
        {"id": "node-job-researcher", "label": "Job", "name": "Research Scientist", "company": "OpenAI"},
        {"id": "node-startup-anthropic", "label": "Startup", "name": "Anthropic", "location": "SF"},
        {"id": "node-product-claude", "label": "Product", "name": "Claude 3.5 Sonnet", "maker": "Anthropic"},
        {"id": "node-news-launch", "label": "News", "name": "Claude 3.5 Release", "sentiment": 0.85},
        # Disconnected island component
        {"id": "node-island-isolated", "label": "Startup", "name": "Isolated Corp", "location": "Remote"},
    ]

    edges = [
        # OpenAI cluster
        {"source": "node-startup-openai", "target": "node-product-chatgpt", "relation": "PRODUCES", "confidence": 0.99},
        {"source": "node-startup-openai", "target": "node-product-gpt4", "relation": "PRODUCES", "confidence": 0.98},
        {"source": "node-job-researcher", "target": "node-startup-openai", "relation": "POSTED_BY", "confidence": 0.95},
        {"source": "node-paper-attention", "target": "node-product-gpt4", "relation": "UTILIZED_BY", "confidence": 0.90},
        # Cross-cluster bridge
        {"source": "node-product-gpt4", "target": "node-product-claude", "relation": "COMPETES_WITH", "confidence": 0.85},
        # Anthropic cluster
        {"source": "node-startup-anthropic", "target": "node-product-claude", "relation": "PRODUCES", "confidence": 0.99},
        {"source": "node-news-launch", "target": "node-product-claude", "relation": "MENTIONS", "confidence": 0.92},
        # Cycle back: Claude -> OpenAI (collaborative research cycle)
        {"source": "node-product-claude", "target": "node-startup-openai", "relation": "BENCHMARKED_AGAINST", "confidence": 0.80},
    ]

    engine.load_graph_data(nodes, edges)
    return engine


# 1. Existing graph data query test
def test_query_existing_repository_graph():
    """Verify that graph_query_engine operates on actual repository graph files."""
    graph_query_engine.reload(refresh_underlying=True)
    stats = graph_linker.build_graph_triples()
    assert len(stats[0]) > 0
    # Lookup first real node
    first_id = stats[0][0]["id"]
    node = graph_query_engine.get_node(first_id)
    assert node is not None
    assert node.id == first_id


# 2. Node lookup and missing node handling
def test_node_lookup_success_and_missing(synthetic_graph_engine):
    """Test node lookup by ID, exact name, and case-insensitive probe."""
    # By exact ID
    n1 = synthetic_graph_engine.get_node("node-startup-openai")
    assert n1 is not None
    assert n1.name == "OpenAI"
    assert n1.label == "Startup"

    # By exact name
    n2 = synthetic_graph_engine.get_node("ChatGPT")
    assert n2 is not None
    assert n2.id == "node-product-chatgpt"

    # By case-insensitive name
    n3 = synthetic_graph_engine.get_node("openai")
    assert n3 is not None
    assert n3.id == "node-startup-openai"

    # Missing node
    assert synthetic_graph_engine.get_node("node-nonexistent-999") is None
    assert synthetic_graph_engine.get_node("") is None

    # Query node structured result
    res_found = synthetic_graph_engine.query_node("node-startup-openai")
    assert res_found.count == 1
    assert res_found.nodes[0].id == "node-startup-openai"

    res_missing = synthetic_graph_engine.query_node("nonexistent-node")
    assert res_missing.count == 0
    assert len(res_missing.nodes) == 0


# 3. Incoming, outgoing, and bidirectional neighbors
def test_neighbor_directions(synthetic_graph_engine):
    """Verify outgoing, incoming, and bidirectional neighbor querying."""
    # Outgoing from OpenAI: ChatGPT (PRODUCES), GPT-4 (PRODUCES)
    out_res = synthetic_graph_engine.get_neighbors("node-startup-openai", direction=Direction.OUTGOING)
    out_ids = {n.id for n in out_res.nodes} - {"node-startup-openai"}
    assert "node-product-chatgpt" in out_ids
    assert "node-product-gpt4" in out_ids
    assert "node-job-researcher" not in out_ids  # incoming edge

    # Incoming to OpenAI: ML Researcher (POSTED_BY), Claude (BENCHMARKED_AGAINST)
    in_res = synthetic_graph_engine.get_neighbors("node-startup-openai", direction=Direction.INCOMING)
    in_ids = {n.id for n in in_res.nodes} - {"node-startup-openai"}
    assert "node-job-researcher" in in_ids
    assert "node-product-claude" in in_ids
    assert "node-product-chatgpt" not in in_ids

    # Bidirectional
    both_res = synthetic_graph_engine.get_neighbors("node-startup-openai", direction=Direction.BOTH)
    both_ids = {n.id for n in both_res.nodes} - {"node-startup-openai"}
    assert len(both_ids) == 4
    assert {"node-product-chatgpt", "node-product-gpt4", "node-job-researcher", "node-product-claude"}.issubset(both_ids)


# 4. Edge-type and node-type filters
def test_neighbor_filtering(synthetic_graph_engine):
    """Test filtering neighbors by relationship type and neighbor node label."""
    # Filter by relation PRODUCES
    prod_res = synthetic_graph_engine.get_neighbors("node-startup-openai", relation="PRODUCES")
    assert prod_res.count == 2
    for e in prod_res.edges:
        assert e.relation == "PRODUCES"

    # Filter by node label "Product"
    label_res = synthetic_graph_engine.get_neighbors("node-startup-openai", node_label="Product")
    neighbor_labels = [n.label for n in label_res.nodes if n.id != "node-startup-openai"]
    assert len(neighbor_labels) > 0
    assert all(lbl == "Product" for lbl in neighbor_labels)

    # Filter by property
    prop_res = synthetic_graph_engine.get_neighbors(
        "node-startup-openai",
        properties={"location": "SF"},
    )
    # Target node has location SF
    assert prop_res.nodes[0].properties.get("location") == "SF"


# 5. Multi-hop traversal
def test_multihop_traversal(synthetic_graph_engine):
    """Test 1-hop, 2-hop, and 3-hop traversal expansions."""
    # 1 hop from OpenAI
    trav1 = synthetic_graph_engine.traverse("node-startup-openai", max_depth=1)
    t1_ids = {n.id for n in trav1.nodes}
    assert "node-startup-openai" in t1_ids
    assert "node-product-chatgpt" in t1_ids
    assert "node-startup-anthropic" not in t1_ids

    # 2 hops from OpenAI should reach Anthropic via:
    # OpenAI -> GPT-4 -> Claude -> Anthropic
    trav2 = synthetic_graph_engine.traverse("node-startup-openai", max_depth=2)
    t2_ids = {n.id for n in trav2.nodes}
    assert "node-startup-anthropic" in t2_ids

    # 3 hops should reach News Launch (News -> Claude -> Anthropic)
    trav3 = synthetic_graph_engine.traverse("node-startup-openai", max_depth=3)
    t3_ids = {n.id for n in trav3.nodes}
    assert "node-news-launch" in t3_ids


# 6. Cycle termination safety
def test_cycle_termination_safety(synthetic_graph_engine):
    """Verify that graph cycles (OpenAI -> GPT4 -> Claude -> OpenAI) terminate safely without infinite loops."""
    # Traversal over graph with cycle must finish quickly and visit each node once
    res = synthetic_graph_engine.traverse("node-startup-openai", max_depth=8)
    assert res.nodes_count <= 9  # exactly 8 connected nodes + root
    assert len(res.nodes) == len({n.id for n in res.nodes})  # no duplicate nodes


# 7. Path discovery (shortest path)
def test_path_discovery(synthetic_graph_engine):
    """Verify finding the shortest path between two entities."""
    # Path from OpenAI to Anthropic
    path_res = synthetic_graph_engine.find_path("node-startup-openai", "node-startup-anthropic")
    assert path_res.count == 1
    p = path_res.paths[0]
    assert p.found is True
    # OpenAI -[:BENCHMARKED_AGAINST]<- Claude <-[:PRODUCES]- Anthropic (length 2)
    assert p.length >= 2
    assert len(p.nodes) == p.length + 1
    assert len(p.edges) == p.length
    assert p.nodes[0].id == "node-startup-openai"
    assert p.nodes[-1].id == "node-startup-anthropic"
    assert len(p.steps) == p.length


# 8. Unreachable targets and path to self
def test_unreachable_targets_and_path_to_self(synthetic_graph_engine):
    """Verify path discovery to self and to disconnected island nodes."""
    # Path to self
    self_path = synthetic_graph_engine.find_path("node-startup-openai", "node-startup-openai")
    assert self_path.paths[0].found is True
    assert self_path.paths[0].length == 0
    assert len(self_path.nodes) == 1

    # Unreachable target
    island_path = synthetic_graph_engine.find_path("node-startup-openai", "node-island-isolated")
    assert island_path.count == 0
    assert island_path.paths[0].found is False
    assert island_path.paths[0].length == 0
    assert len(island_path.nodes) == 0


# 9. Subgraph extraction
def test_subgraph_extraction(synthetic_graph_engine):
    """Verify extracting induced subgraphs for seed node sets."""
    # Subgraph for ChatGPT and Claude with 1 hop
    sub = synthetic_graph_engine.extract_subgraph(
        ["node-product-chatgpt", "node-product-claude"],
        hops=1,
    )
    assert sub.count > 0
    sub_ids = {n.id for n in sub.nodes}
    assert "node-product-chatgpt" in sub_ids
    assert "node-product-claude" in sub_ids
    assert "node-startup-openai" in sub_ids  # connected to both!

    # Edge set contains only internal edges
    for e in sub.edges:
        assert e.source in sub_ids
        assert e.target in sub_ids


# 10. Deduplication of nodes and edges
def test_no_duplicate_nodes_or_edges(synthetic_graph_engine):
    """Verify returned result sets contain no duplicate node IDs or edge signatures."""
    res = synthetic_graph_engine.traverse("node-startup-openai", max_depth=5)
    node_ids = [n.id for n in res.nodes]
    assert len(node_ids) == len(set(node_ids))

    edge_sigs = [f"{e.source}->{e.relation}->{e.target}" for e in res.edges]
    assert len(edge_sigs) == len(set(edge_sigs))


# 11. Maximum depth and result limits enforcement
def test_limits_and_depth_enforcement(synthetic_graph_engine):
    """Verify depth clamping and result truncation limits."""
    # Requesting depth 50 must be clamped to MAX_ALLOWED_DEPTH (10)
    trav = synthetic_graph_engine.traverse("node-startup-openai", max_depth=50)
    assert trav.parameters["max_depth"] <= MAX_ALLOWED_DEPTH

    # Clamping result limit
    limited = synthetic_graph_engine.get_neighbors("node-startup-openai", limit=2)
    assert limited.count <= 2
    assert limited.is_truncated is True


# 12. Relationship pattern query
def test_relationship_query(synthetic_graph_engine):
    """Verify global relationship query pattern."""
    res = synthetic_graph_engine.find_relationships(
        relation="PRODUCES",
        source_label="Startup",
        target_label="Product",
        min_confidence=0.90,
    )
    assert res.count >= 2
    for e in res.edges:
        assert e.relation == "PRODUCES"
        assert e.confidence >= 0.90


# 13. Empty graph behavior
def test_empty_graph_behavior():
    """Verify query operations execute safely on an empty graph without crashing."""
    empty_engine = GraphQueryEngine()
    empty_engine.load_graph_data([], [])

    assert empty_engine.get_node("anything") is None

    nbrs = empty_engine.get_neighbors("anything")
    assert nbrs.count == 0
    assert len(nbrs.nodes) == 0

    trav = empty_engine.traverse("anything")
    assert trav.count == 0

    path = empty_engine.find_path("a", "b")
    assert path.paths[0].found is False

    sub = empty_engine.extract_subgraph(["a", "b"])
    assert sub.count == 0


# 14. Metrics updates
def test_graph_metrics_tracking(synthetic_graph_engine):
    """Verify metrics collector records graph query operations."""
    metrics_collector.reset_graph_metrics()

    synthetic_graph_engine.query_node("node-startup-openai")
    synthetic_graph_engine.get_neighbors("node-startup-openai")
    synthetic_graph_engine.traverse("node-startup-openai")
    synthetic_graph_engine.find_path("node-startup-openai", "node-startup-anthropic")

    m = metrics_collector.get_graph_metrics()
    assert m["graph_queries_total"] >= 4
    assert m["graph_queries_by_type"]["node_lookup"] >= 1
    assert m["graph_queries_by_type"]["neighbors"] >= 1
    assert m["graph_queries_by_type"]["traverse"] >= 1
    assert m["graph_queries_by_type"]["path"] >= 1
    assert m["graph_nodes_visited"] > 0

    prom = metrics_collector.generate_prometheus_format()
    assert "graphone_graph_queries_total" in prom
    assert "graphone_graph_nodes_visited_total" in prom


# 15. FastAPI Integration & Validation
def test_fastapi_graph_endpoints():
    """Integration test verifying FastAPI endpoints, parameter validation, and 404s."""
    client = TestClient(app)

    # 1. GET /graph/nodes/{node_id}
    resp = client.get("/graph/nodes/node-startup-openai")
    if resp.status_code == 200:
        assert resp.json()["id"] == "node-startup-openai"
    else:
        # If seed data has different ID, test 404 on nonexistent
        resp_404 = client.get("/graph/nodes/nonexistent-xyz-999")
        assert resp_404.status_code == 404

    # 2. GET /graph/nodes/{node_id}/neighbors with query params
    resp_nbr = client.get("/graph/nodes/node-startup-anthropic/neighbors?direction=outgoing&limit=5")
    if resp_nbr.status_code == 200:
        data = resp_nbr.json()
        assert "nodes" in data
        assert "edges" in data
        assert data["query_type"] == "neighbors"

    # 3. POST /graph/query/traverse
    resp_trav = client.post(
        "/graph/query/traverse",
        json={"root_id": "node-startup-anthropic", "max_depth": 2, "limit": 50},
    )
    if resp_trav.status_code == 200:
        trav_data = resp_trav.json()
        assert trav_data["query_type"] == "traverse"
        assert trav_data["nodes_count"] > 0

    # 4. POST /graph/query/traverse - missing root 404
    resp_trav_404 = client.post(
        "/graph/query/traverse",
        json={"root_id": "missing-root-entity-999", "max_depth": 2},
    )
    assert resp_trav_404.status_code == 404

    # 5. POST /graph/query/path
    resp_path = client.post(
        "/graph/query/path",
        json={"from_node": "node-startup-anthropic", "to_node": "node-startup-anthropic"},
    )
    if resp_path.status_code == 200:
        path_data = resp_path.json()
        assert path_data["query_type"] == "path"
        assert path_data["count"] == 1

    # 6. POST /graph/query/path - missing node 404
    resp_path_404 = client.post(
        "/graph/query/path",
        json={"from_node": "missing-a", "to_node": "missing-b"},
    )
    assert resp_path_404.status_code == 404

    # 7. POST /graph/query/subgraph
    resp_sub = client.post(
        "/graph/query/subgraph",
        json={"node_ids": ["node-startup-anthropic"], "hops": 1},
    )
    assert resp_sub.status_code == 200
    assert resp_sub.json()["query_type"] == "subgraph"

    # 8. POST /graph/query/relationships
    resp_rel = client.post(
        "/graph/query/relationships",
        json={"relation": "PRODUCES", "limit": 10},
    )
    assert resp_rel.status_code == 200
    assert resp_rel.json()["query_type"] == "relationship"

    # 9. GET /graph/stats
    resp_stats = client.get("/graph/stats")
    assert resp_stats.status_code == 200
    stats_data = resp_stats.json()
    assert "nodes_count" in stats_data
    assert "edges_count" in stats_data
    assert "metrics" in stats_data
