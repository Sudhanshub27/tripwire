"""
================================================================================
GRAPHONE PIPELINE: GRAPH QUERY ENGINE PERFORMANCE BENCHMARK
================================================================================
Evaluates graph traversal, path discovery, and neighborhood exploration across
deterministic synthetic graphs of multiple scales (Small: 100 nodes, Medium: 1K nodes,
Large: 5K nodes) measuring execution latency, throughput, and memory bounds.
================================================================================
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from rich.console import Console
from rich.table import Table
from src.graph.models import Direction
from src.graph.query_engine import GraphQueryEngine

logger = structlog.get_logger(__name__)
console = Console()

OUTPUT_REPORT_PATH = Path("evaluation/reports/graph_benchmark.json")


def generate_deterministic_synthetic_graph(
    node_count: int,
    edge_count: int,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate a deterministic synthetic graph with typed nodes and directional relationships.
    Creates structured clusters, multi-hop chains, and cross-cluster bridges.
    """
    rng = random.Random(seed)
    node_types = ["Startup", "Product", "Job", "ResearchPaper", "News"]
    relations = ["PRODUCES", "POSTED_BY", "CITES", "UTILIZES", "MENTIONS", "PARTNERS_WITH"]

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for i in range(node_count):
        label = node_types[i % len(node_types)]
        nid = f"syn-{label.lower()}-{i:05d}"
        name = f"Synthetic {label} {i}"
        nodes.append({
            "id": nid,
            "label": label,
            "name": name,
            "cluster": i % 10,
            "version": f"1.{i % 5}",
        })

    # Ensure strong connectivity by generating a spanning backbone chain
    for i in range(node_count - 1):
        edges.append({
            "source": nodes[i]["id"],
            "target": nodes[i + 1]["id"],
            "relation": relations[i % len(relations)],
            "confidence": round(0.85 + (i % 15) * 0.01, 2),
        })

    # Add remaining random cluster and bridge edges
    edge_set = {(e["source"], e["target"]) for e in edges}

    attempts = 0
    while len(edges) < edge_count and attempts < edge_count * 5:
        attempts += 1
        u = rng.choice(nodes)["id"]
        v = rng.choice(nodes)["id"]
        if u != v and (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append({
                "source": u,
                "target": v,
                "relation": rng.choice(relations),
                "confidence": round(0.80 + rng.random() * 0.20, 2),
            })

    return nodes, edges


def run_benchmark_on_scale(
    name: str,
    node_count: int,
    edge_count: int,
    iterations: int = 10,
) -> Dict[str, Any]:
    """Execute benchmark operations across a graph scale and record timings."""
    nodes, edges = generate_deterministic_synthetic_graph(node_count, edge_count)
    engine = GraphQueryEngine()
    engine.load_graph_data(nodes, edges)

    # Test nodes
    root_id = nodes[0]["id"]
    mid_id = nodes[node_count // 2]["id"]
    last_id = nodes[node_count - 1]["id"]

    benchmarks: Dict[str, Dict[str, Any]] = {}

    # 1. Node Lookup Benchmark
    latencies = []
    for _ in range(iterations * 5):
        t0 = time.perf_counter()
        node = engine.get_node(mid_id)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["node_lookup"] = {
        "avg_ms": round(sum(latencies) / len(latencies), 4),
        "min_ms": round(min(latencies), 4),
        "max_ms": round(max(latencies), 4),
        "nodes_returned": 1 if node else 0,
    }

    # 2. Neighbors Query Benchmark (1-hop)
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        nbr_res = engine.get_neighbors(root_id, direction=Direction.BOTH, limit=100)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["neighbors_1hop"] = {
        "avg_ms": round(sum(latencies) / len(latencies), 4),
        "min_ms": round(min(latencies), 4),
        "max_ms": round(max(latencies), 4),
        "nodes_returned": nbr_res.nodes_count,
        "edges_returned": nbr_res.edges_count,
    }

    # 3. 2-Hop Traversal Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        trav2 = engine.traverse(root_id, max_depth=2, limit=200)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["traversal_2hop"] = {
        "avg_ms": round(sum(latencies) / len(latencies), 4),
        "min_ms": round(min(latencies), 4),
        "max_ms": round(max(latencies), 4),
        "nodes_visited": trav2.visited_nodes_count,
        "nodes_returned": trav2.nodes_count,
    }

    # 4. 3-Hop Traversal Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        trav3 = engine.traverse(root_id, max_depth=3, limit=300)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["traversal_3hop"] = {
        "avg_ms": round(sum(latencies) / len(latencies), 4),
        "min_ms": round(min(latencies), 4),
        "max_ms": round(max(latencies), 4),
        "nodes_visited": trav3.visited_nodes_count,
        "nodes_returned": trav3.nodes_count,
    }

    # 5. Shortest Path Discovery Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        path_res = engine.find_path(root_id, mid_id, max_depth=8)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["path_discovery"] = {
        "avg_ms": round(sum(latencies) / len(latencies), 4),
        "min_ms": round(min(latencies), 4),
        "max_ms": round(max(latencies), 4),
        "path_found": bool(path_res.paths and path_res.paths[0].found),
        "path_length": path_res.paths[0].length if path_res.paths else 0,
    }

    # 6. Subgraph Extraction Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sub_res = engine.extract_subgraph([root_id, mid_id, last_id], hops=1, limit=250)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    benchmarks["subgraph_extraction"] = {
        "avg_ms": round(sum(latencies) / len(latencies), 4),
        "min_ms": round(min(latencies), 4),
        "max_ms": round(max(latencies), 4),
        "nodes_returned": sub_res.nodes_count,
        "edges_returned": sub_res.edges_count,
    }

    return {
        "scale_name": name,
        "node_count": node_count,
        "edge_count": edge_count,
        "benchmarks": benchmarks,
    }


def run_full_graph_benchmark() -> Dict[str, Any]:
    """Execute benchmarks across Small, Medium, and Large synthetic graphs."""
    scales = [
        ("Small Graph", 100, 250),
        ("Medium Graph", 1000, 3500),
        ("Large Graph", 5000, 20000),
    ]

    results = []
    for name, n_count, e_count in scales:
        console.print(f"[bold blue]Evaluating {name}[/bold blue] ({n_count} nodes, {e_count} edges)...")
        res = run_benchmark_on_scale(name, n_count, e_count)
        results.append(res)

    # Render results summary table
    table = Table(title="GraphOne Phase 4: Graph Query Engine Performance Benchmark")
    table.add_column("Scale", style="cyan", no_wrap=True)
    table.add_column("Topology (Nodes / Edges)", style="white")
    table.add_column("Node Lookup (avg ms)", justify="right")
    table.add_column("1-Hop Neighbors (avg ms)", justify="right")
    table.add_column("2-Hop Traverse (avg ms)", justify="right")
    table.add_column("3-Hop Traverse (avg ms)", justify="right")
    table.add_column("Shortest Path (avg ms)", justify="right")
    table.add_column("Subgraph Extract (avg ms)", justify="right")

    for r in results:
        b = r["benchmarks"]
        table.add_row(
            r["scale_name"],
            f"{r['node_count']} / {r['edge_count']}",
            f"{b['node_lookup']['avg_ms']:.3f} ms",
            f"{b['neighbors_1hop']['avg_ms']:.3f} ms",
            f"{b['traversal_2hop']['avg_ms']:.3f} ms",
            f"{b['traversal_3hop']['avg_ms']:.3f} ms",
            f"{b['path_discovery']['avg_ms']:.3f} ms",
            f"{b['subgraph_extraction']['avg_ms']:.3f} ms",
        )

    console.print(table)

    report_payload = {
        "benchmark": "Graph Query Engine Traversal & Path Discovery",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scales": results,
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    console.print(f"[green]✓ Saved benchmark results to {OUTPUT_REPORT_PATH}[/green]")

    return report_payload


if __name__ == "__main__":
    run_full_graph_benchmark()
