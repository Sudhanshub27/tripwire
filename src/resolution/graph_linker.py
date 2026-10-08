"""
================================================================================
GRAPHONE PIPELINE: KNOWLEDGE GRAPH LINKAGE & RELATION EXTRACTION ENGINE
================================================================================

Transforms extracted entity JSONL records into a relational Knowledge Graph with
typed nodes and edges (Graph Triples). Supports graph querying (node lookup,
direct neighbors, relationship filtering, path discovery, and bounded multi-hop
traversals with configurable maximum hop guards). Generates GraphML and Cypher
export scripts for Neo4j / NetworkX graph visualization.
================================================================================
"""

import json
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)

# Configurable safe traversal depth limits
MAX_TRAVERSAL_HOPS: int = 5
DEFAULT_MAX_HOPS: int = 2


class KnowledgeGraphLinker:
    """Knowledge Graph Construction, Querying, Traversal, and Export Engine."""

    def __init__(self):
        self.nodes_file = settings.DATA_PROCESSED_DIR / "graph_nodes.jsonl"
        self.edges_file = settings.DATA_PROCESSED_DIR / "graph_edges.jsonl"
        self._cached_nodes: Optional[List[Dict[str, Any]]] = None
        self._cached_edges: Optional[List[Dict[str, Any]]] = None

    def build_graph_triples(self, refresh: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build graph nodes and relational edges across all ingested entities."""
        if not refresh and self._cached_nodes is not None and self._cached_edges is not None:
            return self._cached_nodes, self._cached_edges

        from src.dashboard.processed_reader import get_all_processed_records
        from src.resolution.entity_resolver import normalize_entity_name

        all_records = get_all_processed_records()
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_ids: set[str] = set()

        # 1. Process Startup Nodes
        for startup in all_records.get("startups", []):
            s_name = startup.get("canonical_name") or startup.get("name") or "Unknown Startup"
            s_id = f"node-startup-{startup.get('id', s_name.lower().replace(' ', '-'))}"
            if s_id not in node_ids:
                node_ids.add(s_id)
                nodes.append({
                    "id": s_id,
                    "label": "Startup",
                    "name": s_name,
                    "location": startup.get("location"),
                    "funding": startup.get("total_funding"),
                    "stage": startup.get("stage"),
                })

        # 2. Process Product Nodes & PRODUCES edges
        for product in all_records.get("products", []):
            p_name = product.get("canonical_name") or product.get("name") or product.get("title") or "Unknown Product"
            p_id = f"node-product-{product.get('id', p_name.lower().replace(' ', '-'))}"
            if p_id not in node_ids:
                node_ids.add(p_id)
                nodes.append({
                    "id": p_id,
                    "label": "Product",
                    "name": p_name,
                    "maker": product.get("maker_company"),
                    "pricing": product.get("pricing_model"),
                })

            # Connect Product to Maker Startup
            maker = product.get("maker_company")
            if maker:
                matching_startup_id = None
                norm_maker = normalize_entity_name(maker)
                for node in nodes:
                    if node["label"] == "Startup":
                        norm_node_name = normalize_entity_name(node["name"])
                        if (
                            (norm_maker and norm_node_name and (norm_maker in norm_node_name or norm_node_name in norm_maker))
                            or maker.lower() in node["name"].lower()
                            or node["name"].lower() in maker.lower()
                        ):
                            matching_startup_id = node["id"]
                            break
                if matching_startup_id:
                    edges.append({
                        "source": matching_startup_id,
                        "target": p_id,
                        "relation": "PRODUCES",
                        "confidence": 0.95,
                    })

        # 3. Process Research Paper Nodes
        for paper in all_records.get("research_papers", []):
            title = paper.get("title") or "Untitled Research Paper"
            paper_id = f"node-paper-{paper.get('id', title.lower()[:20].replace(' ', '-'))}"
            if paper_id not in node_ids:
                node_ids.add(paper_id)
                nodes.append({
                    "id": paper_id,
                    "label": "ResearchPaper",
                    "name": title,
                    "authors": paper.get("authors", []),
                    "journal": paper.get("journal_conference"),
                })

        # 4. Process Job Nodes & POSTED_BY edges
        for job in all_records.get("jobs", []):
            j_title = job.get("title") or "Unknown Position"
            comp = job.get("company")
            j_id = f"node-job-{job.get('id', j_title.lower()[:20].replace(' ', '-'))}"
            if j_id not in node_ids:
                node_ids.add(j_id)
                nodes.append({
                    "id": j_id,
                    "label": "Job",
                    "name": j_title,
                    "company": comp,
                    "salary": job.get("salary_range"),
                })

            if comp:
                for node in nodes:
                    if node["label"] == "Startup" and comp.lower() in node["name"].lower():
                        edges.append({
                            "source": j_id,
                            "target": node["id"],
                            "relation": "POSTED_BY",
                            "confidence": 0.90,
                        })
                        break

        # Persist graph nodes and edges to JSONL
        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.nodes_file, "w", encoding="utf-8") as f:
                for n in nodes:
                    f.write(json.dumps(n) + "\n")
            with open(self.edges_file, "w", encoding="utf-8") as f:
                for e in edges:
                    f.write(json.dumps(e) + "\n")
        except Exception as e:
            logger.warning("Failed writing graph nodes/edges JSONL files", error=str(e))

        self._cached_nodes = nodes
        self._cached_edges = edges
        logger.info("Built Knowledge Graph Triples", nodes_count=len(nodes), edges_count=len(edges))
        return nodes, edges

    def _ensure_graph(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Ensure graph is initialized and returned."""
        return self.build_graph_triples()

    def get_node(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Lookup a node by its unique node id or case-insensitive name.
        Returns node dict or None if not found.
        """
        nodes, _ = self._ensure_graph()
        if not entity_id or not entity_id.strip():
            return None

        clean_query = entity_id.strip().lower()

        # 1. Exact ID match
        for n in nodes:
            if n["id"].lower() == clean_query:
                return n

        # 2. Exact Name match
        for n in nodes:
            if n.get("name", "").lower() == clean_query:
                return n

        # 3. Partial ID / Name match
        for n in nodes:
            if clean_query in n["id"].lower() or clean_query in n.get("name", "").lower():
                return n

        return None

    def get_neighbors(
        self,
        entity_id: str,
        relation: Optional[str] = None,
        direction: str = "both",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve direct (1-hop) neighbors of an entity.
        Supports relationship filtering ('PRODUCES', 'POSTED_BY') and direction ('both', 'outgoing', 'incoming').
        Returns structured dict with nodes, edges, and neighbors list, or None if entity is missing.
        """
        nodes, edges = self._ensure_graph()
        target_node = self.get_node(entity_id)
        if not target_node:
            return None

        target_id = target_node["id"]
        rel_filter = relation.upper().strip() if relation else None

        connected_edges: List[Dict[str, Any]] = []
        neighbor_ids: set[str] = set()
        neighbors_detail: List[Dict[str, Any]] = []

        for e in edges:
            src = e["source"]
            tgt = e["target"]
            rel = e.get("relation", "")

            if rel_filter and rel.upper() != rel_filter:
                continue

            if direction in ("outgoing", "both") and src == target_id:
                connected_edges.append(e)
                neighbor_ids.add(tgt)
                neighbors_detail.append({
                    "neighbor_id": tgt,
                    "relation": rel,
                    "direction": "outgoing",
                    "confidence": e.get("confidence", 1.0),
                })
            elif direction in ("incoming", "both") and tgt == target_id:
                connected_edges.append(e)
                neighbor_ids.add(src)
                neighbors_detail.append({
                    "neighbor_id": src,
                    "relation": rel,
                    "direction": "incoming",
                    "confidence": e.get("confidence", 1.0),
                })

        nodes_by_id = {n["id"]: n for n in nodes}
        neighbor_nodes = [nodes_by_id[nid] for nid in neighbor_ids if nid in nodes_by_id]

        for item in neighbors_detail:
            nid = item["neighbor_id"]
            if nid in nodes_by_id:
                item["node"] = nodes_by_id[nid]

        result_nodes = [target_node] + neighbor_nodes

        return {
            "node": target_node,
            "nodes": result_nodes,
            "edges": connected_edges,
            "neighbors": neighbors_detail,
            "path": [],
        }

    def find_path(
        self,
        from_entity: str,
        to_entity: str,
        max_depth: int = MAX_TRAVERSAL_HOPS,
        relation: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Discover shortest path between two entities using Breadth-First Search (BFS).
        Enforces safe maximum traversal depth.
        Returns structured dict with nodes, edges, path tokens, or None if either node is missing.
        """
        nodes, edges = self._ensure_graph()
        start_node = self.get_node(from_entity)
        end_node = self.get_node(to_entity)

        if not start_node or not end_node:
            return None

        start_id = start_node["id"]
        end_id = end_node["id"]

        nodes_by_id = {n["id"]: n for n in nodes}
        safe_max_depth = min(max(1, max_depth), MAX_TRAVERSAL_HOPS)
        rel_filter = relation.upper().strip() if relation else None

        if start_id == end_id:
            return {
                "from": start_node,
                "to": end_node,
                "found": True,
                "length": 0,
                "nodes": [start_node],
                "edges": [],
                "path": [start_id],
            }

        # Build bidirectional adjacency: node_id -> [(neighbor_id, edge_dict, direction)]
        adj: Dict[str, List[Tuple[str, Dict[str, Any], str]]] = {}
        for e in edges:
            rel = e.get("relation", "")
            if rel_filter and rel.upper() != rel_filter:
                continue
            src, tgt = e["source"], e["target"]
            adj.setdefault(src, []).append((tgt, e, "outgoing"))
            adj.setdefault(tgt, []).append((src, e, "incoming"))

        queue: deque[Tuple[str, List[str], List[Dict[str, Any]], List[str]]] = deque([
            (start_id, [start_id], [], [start_id])
        ])
        visited = {start_id}
        found_path: Optional[Tuple[List[str], List[Dict[str, Any]], List[str]]] = None

        while queue:
            curr_id, path_nodes, path_edges, path_tokens = queue.popleft()

            if len(path_edges) >= safe_max_depth:
                continue

            for neighbor_id, edge, direction in adj.get(curr_id, []):
                rel = edge.get("relation", "CONNECTED_TO")
                rel_token = f"-[:{rel}]->" if direction == "outgoing" else f"<-[:{rel}]-"

                if neighbor_id == end_id:
                    found_path = (
                        path_nodes + [end_id],
                        path_edges + [edge],
                        path_tokens + [rel_token, end_id],
                    )
                    break

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((
                        neighbor_id,
                        path_nodes + [neighbor_id],
                        path_edges + [edge],
                        path_tokens + [rel_token, neighbor_id],
                    ))

            if found_path:
                break

        if found_path:
            p_nodes, p_edges, p_tokens = found_path
            return {
                "from": start_node,
                "to": end_node,
                "found": True,
                "length": len(p_edges),
                "nodes": [nodes_by_id[nid] for nid in p_nodes if nid in nodes_by_id],
                "edges": p_edges,
                "path": p_tokens,
            }

        return {
            "from": start_node,
            "to": end_node,
            "found": False,
            "length": 0,
            "nodes": [],
            "edges": [],
            "path": [],
        }

    def traverse(
        self,
        entity_id: str,
        hops: int = DEFAULT_MAX_HOPS,
        max_hops: int = MAX_TRAVERSAL_HOPS,
        relation: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Bounded multi-hop traversal starting from entity_id up to safe configurable maximum hops.
        Prevents infinite loops and runaway graph expansion.
        Returns structured { nodes, edges, path } or None if entity missing.
        """
        nodes, edges = self._ensure_graph()
        root_node = self.get_node(entity_id)
        if not root_node:
            return None

        root_id = root_node["id"]
        safe_hops = min(max(1, hops), max_hops, MAX_TRAVERSAL_HOPS)
        rel_filter = relation.upper().strip() if relation else None

        nodes_by_id = {n["id"]: n for n in nodes}

        # Build adjacency mapping: node_id -> [(neighbor_id, edge_dict)]
        adj: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for e in edges:
            rel = e.get("relation", "")
            if rel_filter and rel.upper() != rel_filter:
                continue
            src, tgt = e["source"], e["target"]
            adj.setdefault(src, []).append((tgt, e))
            adj.setdefault(tgt, []).append((src, e))

        # BFS level-order traversal
        visited_nodes: set[str] = {root_id}
        visited_edges: List[Dict[str, Any]] = []
        edge_signatures: set[str] = set()

        current_frontier = {root_id}

        for _ in range(1, safe_hops + 1):
            next_frontier = set()
            for curr_node_id in current_frontier:
                for neighbor_id, edge in adj.get(curr_node_id, []):
                    # Record unique edge
                    sig = f"{edge['source']}->{edge.get('relation', '')}->{edge['target']}"
                    if sig not in edge_signatures:
                        edge_signatures.add(sig)
                        visited_edges.append(edge)

                    if neighbor_id not in visited_nodes:
                        visited_nodes.add(neighbor_id)
                        next_frontier.add(neighbor_id)

            current_frontier = next_frontier
            if not current_frontier:
                break

        subgraph_nodes = [nodes_by_id[nid] for nid in visited_nodes if nid in nodes_by_id]

        return {
            "root": root_node,
            "hops": safe_hops,
            "max_hops_enforced": MAX_TRAVERSAL_HOPS,
            "nodes": subgraph_nodes,
            "edges": visited_edges,
            "path": [],
        }

    def generate_cypher_import_script(self) -> str:
        """Generate Cypher queries for Neo4j database import."""
        nodes, edges = self.build_graph_triples()
        cypher_lines = ["// --- Graphone Pipeline Neo4j Cypher Import Script ---"]

        for n in nodes:
            name_clean = n["name"].replace('"', '\\"')
            cypher_lines.append(f'MERGE (n:{n["label"]} {{id: "{n["id"]}", name: "{name_clean}"}});')

        for e in edges:
            cypher_lines.append(
                f'MATCH (a {{id: "{e["source"]}"}}), (b {{id: "{e["target"]}"}}) '
                f'MERGE (a)-[r:{e["relation"]} {{confidence: {e["confidence"]}}}]->(b);'
            )

        return "\n".join(cypher_lines) + "\n"


graph_linker = KnowledgeGraphLinker()
