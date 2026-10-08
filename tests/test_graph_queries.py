"""
================================================================================
TEST SUITE FOR KNOWLEDGE GRAPH QUERIES & MULTI-HOP TRAVERSALS
================================================================================
Covers:
- Node lookup by ID and case-insensitive name
- Direct neighbors retrieval with relation and direction filtering
- Missing node error handling (404 / None)
- One-hop traversal
- Multi-hop traversal (2+ hops)
- Maximum hop enforcement (safely capped traversal depth)
- Path discovery (shortest path between entities)
- FastAPI endpoints (/api/graph/entity, neighbors, traverse, path)
================================================================================
"""

import pytest
from fastapi.testclient import TestClient
from src.dashboard.main import app
from src.resolution.graph_linker import (
    MAX_TRAVERSAL_HOPS,
    KnowledgeGraphLinker,
)


@pytest.fixture
def synthetic_graph():
    """Create a KnowledgeGraphLinker with a controlled synthetic graph topology."""
    linker = KnowledgeGraphLinker()

    nodes = [
        {"id": "node-startup-acme", "label": "Startup", "name": "Acme AI", "location": "SF"},
        {"id": "node-product-bot", "label": "Product", "name": "Acme Bot", "maker": "Acme AI"},
        {"id": "node-product-vision", "label": "Product", "name": "Acme Vision", "maker": "Acme AI"},
        {"id": "node-job-engineer", "label": "Job", "name": "ML Engineer", "company": "Acme AI"},
        {"id": "node-startup-beta", "label": "Startup", "name": "Beta Corp", "location": "NYC"},
        {"id": "node-product-core", "label": "Product", "name": "Beta Core", "maker": "Beta Corp"},
        {"id": "node-island-isolated", "label": "Startup", "name": "Island Corp"},
    ]

    edges = [
        {"source": "node-startup-acme", "target": "node-product-bot", "relation": "PRODUCES", "confidence": 0.95},
        {"source": "node-startup-acme", "target": "node-product-vision", "relation": "PRODUCES", "confidence": 0.95},
        {"source": "node-job-engineer", "target": "node-startup-acme", "relation": "POSTED_BY", "confidence": 0.90},
        {"source": "node-startup-beta", "target": "node-product-core", "relation": "PRODUCES", "confidence": 0.95},
        # Bridge edge connecting Acme and Beta components
        {"source": "node-product-core", "target": "node-product-bot", "relation": "INTEGRATES_WITH", "confidence": 0.85},
    ]

    linker._cached_nodes = nodes
    linker._cached_edges = edges
    return linker


def test_node_lookup(synthetic_graph):
    """Test looking up a node by exact ID and case-insensitive name."""
    # Lookup by exact ID
    node1 = synthetic_graph.get_node("node-startup-acme")
    assert node1 is not None
    assert node1["name"] == "Acme AI"
    assert node1["label"] == "Startup"

    # Lookup by name
    node2 = synthetic_graph.get_node("Acme Bot")
    assert node2 is not None
    assert node2["id"] == "node-product-bot"

    # Lookup case-insensitive
    node3 = synthetic_graph.get_node("acme ai")
    assert node3 is not None
    assert node3["id"] == "node-startup-acme"


def test_missing_node(synthetic_graph):
    """Test looking up a non-existent entity returns None."""
    assert synthetic_graph.get_node("node-nonexistent-99999") is None
    assert synthetic_graph.get_node("") is None
    assert synthetic_graph.get_neighbors("node-nonexistent-99999") is None
    assert synthetic_graph.traverse("node-nonexistent-99999") is None


def test_neighbors(synthetic_graph):
    """Test direct neighbors retrieval and relation filtering."""
    # Neighbors of Acme AI (has 2 PRODUCES outgoing, 1 POSTED_BY incoming)
    res = synthetic_graph.get_neighbors("node-startup-acme")
    assert res is not None
    assert res["node"]["id"] == "node-startup-acme"
    assert len(res["neighbors"]) == 3
    assert len(res["edges"]) == 3
    assert len(res["nodes"]) == 4  # Root + 3 neighbors

    # Filter by relation PRODUCES
    produces_res = synthetic_graph.get_neighbors("node-startup-acme", relation="PRODUCES")
    assert produces_res is not None
    assert len(produces_res["neighbors"]) == 2
    for n in produces_res["neighbors"]:
        assert n["relation"] == "PRODUCES"

    # Filter by relation POSTED_BY
    jobs_res = synthetic_graph.get_neighbors("node-startup-acme", relation="POSTED_BY")
    assert jobs_res is not None
    assert len(jobs_res["neighbors"]) == 1
    assert jobs_res["neighbors"][0]["relation"] == "POSTED_BY"


def test_one_hop_traversal(synthetic_graph):
    """Test 1-hop traversal retrieves root and immediate neighbors only."""
    res = synthetic_graph.traverse("node-startup-acme", hops=1)
    assert res is not None
    assert res["root"]["id"] == "node-startup-acme"
    assert res["hops"] == 1

    node_ids = {n["id"] for n in res["nodes"]}
    assert "node-startup-acme" in node_ids
    assert "node-product-bot" in node_ids
    assert "node-product-vision" in node_ids
    assert "node-job-engineer" in node_ids
    # 2nd-degree neighbor Beta Core should NOT be in 1-hop
    assert "node-product-core" not in node_ids


def test_multi_hop_traversal(synthetic_graph):
    """Test 2-hop traversal expands beyond direct neighbors to 2nd-degree nodes."""
    # From Acme AI:
    # 1 hop -> Acme Bot, Acme Vision, ML Engineer
    # 2 hops -> Beta Core (via Acme Bot -[INTEGRATES_WITH]-> Beta Core)
    res = synthetic_graph.traverse("node-startup-acme", hops=2)
    assert res is not None
    assert res["hops"] == 2

    node_ids = {n["id"] for n in res["nodes"]}
    assert "node-startup-acme" in node_ids
    assert "node-product-bot" in node_ids
    assert "node-product-core" in node_ids  # Reached via 2 hops!

    # 3 hops should reach Beta Corp (Beta Core -[PRODUCES]-> Beta Corp)
    res_3hop = synthetic_graph.traverse("node-startup-acme", hops=3)
    node_ids_3hop = {n["id"] for n in res_3hop["nodes"]}
    assert "node-startup-beta" in node_ids_3hop


def test_max_hop_enforcement(synthetic_graph):
    """Test that requesting excessive hops is safely bounded to MAX_TRAVERSAL_HOPS."""
    # Requesting 20 hops must be capped at MAX_TRAVERSAL_HOPS (5)
    res = synthetic_graph.traverse("node-startup-acme", hops=20)
    assert res is not None
    assert res["hops"] <= MAX_TRAVERSAL_HOPS
    assert res["hops"] == 5


def test_path_discovery(synthetic_graph):
    """Test discovering the shortest path between two connected entities."""
    # Path from Acme AI to Beta Corp
    res = synthetic_graph.find_path("node-startup-acme", "node-startup-beta")
    assert res is not None
    assert res["found"] is True
    assert res["length"] == 3  # Acme AI -> Acme Bot -> Beta Core -> Beta Corp
    assert len(res["nodes"]) == 4
    assert len(res["edges"]) == 3
    assert len(res["path"]) > 0

    # Path to self
    self_res = synthetic_graph.find_path("node-startup-acme", "node-startup-acme")
    assert self_res is not None
    assert self_res["found"] is True
    assert self_res["length"] == 0

    # Path to disconnected island node
    island_res = synthetic_graph.find_path("node-startup-acme", "node-island-isolated")
    assert island_res is not None
    assert island_res["found"] is False
    assert island_res["length"] == 0


def test_backend_graph_api_endpoints():
    """Integration tests for FastAPI graph query endpoints."""
    client = TestClient(app)

    # 1. GET /api/graph
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data

    # 2. GET /api/graph/entity/{id}
    if data["nodes"]:
        first_node = data["nodes"][0]
        entity_id = first_node["id"]
        resp_entity = client.get(f"/api/graph/entity/{entity_id}")
        assert resp_entity.status_code == 200
        assert resp_entity.json()["id"] == entity_id

        # 3. GET /api/graph/neighbors/{id}
        resp_nbr = client.get(f"/api/graph/neighbors/{entity_id}")
        assert resp_nbr.status_code == 200
        assert "neighbors" in resp_nbr.json()
        assert "nodes" in resp_nbr.json()
        assert "edges" in resp_nbr.json()

        # 4. GET /api/graph/traverse/{id}?hops=2
        resp_trav = client.get(f"/api/graph/traverse/{entity_id}?hops=2")
        assert resp_trav.status_code == 200
        trav_data = resp_trav.json()
        assert "nodes" in trav_data
        assert "edges" in trav_data
        assert trav_data["hops"] == 2

    # 5. Missing node 404
    resp_404 = client.get("/api/graph/entity/nonexistent-node-xyz")
    assert resp_404.status_code == 404

    # 6. GET /api/graph/path?from=...&to=...
    if len(data["nodes"]) >= 2:
        n1 = data["nodes"][0]["id"]
        n2 = data["nodes"][1]["id"]
        resp_path = client.get(f"/api/graph/path?from={n1}&to={n2}")
        assert resp_path.status_code == 200
        assert "found" in resp_path.json()
        assert "nodes" in resp_path.json()
        assert "edges" in resp_path.json()
