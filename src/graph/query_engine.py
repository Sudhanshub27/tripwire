"""
================================================================================
GRAPHONE PIPELINE: PRODUCTION GRAPH QUERY ENGINE
================================================================================
High-performance graph query engine supporting node lookups, filtered
neighborhood exploration, bounded multi-hop traversals, path discovery,
subgraph extraction, and relationship pattern queries with in-memory indexed
adjacency, cycle protection, and observability metrics.
================================================================================
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from src.graph.models import (
    Direction,
    GraphEdge,
    GraphNode,
    PathResult,
    PathStep,
    QueryResult,
    QueryType,
)
from src.observability.metrics import metrics_collector

logger = structlog.get_logger(__name__)

# System-wide safety constraints
MAX_ALLOWED_DEPTH: int = 10
DEFAULT_TRAVERSAL_DEPTH: int = 2
DEFAULT_PATH_DEPTH: int = 5
MAX_RESULT_LIMIT: int = 500
DEFAULT_RESULT_LIMIT: int = 100


class GraphQueryEngine:
    """
    Production-grade query engine operating over the Knowledge Graph.
    Maintains indexed adjacency lists for O(1) neighbor lookups and
    executes cycle-safe BFS graph traversals with resource bounds.
    """

    def __init__(self, linker: Optional[Any] = None):
        self._linker = linker
        self._nodes_by_id: Dict[str, GraphNode] = {}
        self._nodes_by_name: Dict[str, str] = {}  # lowercase normalized name -> node ID
        self._adj_out: Dict[str, List[GraphEdge]] = {}  # source_id -> list of outgoing edges
        self._adj_in: Dict[str, List[GraphEdge]] = {}   # target_id -> list of incoming edges
        self._all_edges: List[GraphEdge] = []
        self._is_indexed: bool = False

    def _ensure_linker(self):
        if self._linker is None:
            from src.resolution.graph_linker import graph_linker
            self._linker = graph_linker

    def reload(self, refresh_underlying: bool = False) -> None:
        """Reload and re-index graph nodes and edges from storage or linker."""
        self._ensure_linker()
        raw_nodes, raw_edges = self._linker.build_graph_triples(refresh=refresh_underlying)
        self.load_graph_data(raw_nodes, raw_edges)

    def load_graph_data(
        self,
        raw_nodes: List[Dict[str, Any]],
        raw_edges: List[Dict[str, Any]],
    ) -> None:
        """
        Build optimized in-memory adjacency indices from raw node and edge definitions.
        Supports both raw dictionaries and Pydantic models.
        """
        nodes_by_id: Dict[str, GraphNode] = {}
        nodes_by_name: Dict[str, str] = {}
        adj_out: Dict[str, List[GraphEdge]] = {}
        adj_in: Dict[str, List[GraphEdge]] = {}
        all_edges: List[GraphEdge] = []

        # 1. Index Nodes
        for n in raw_nodes:
            if isinstance(n, GraphNode):
                node_obj = n
            else:
                nid = str(n.get("id", "")).strip()
                label = str(n.get("label", "Entity")).strip()
                name = str(n.get("name", nid)).strip()
                # Store all remaining attributes as properties
                reserved = {"id", "label", "name", "properties"}
                props = {k: v for k, v in n.items() if k not in reserved}
                if "properties" in n and isinstance(n["properties"], dict):
                    props.update(n["properties"])
                node_obj = GraphNode(id=nid, label=label, name=name, properties=props)

            if node_obj.id:
                nodes_by_id[node_obj.id] = node_obj
                nodes_by_name[node_obj.name.lower().strip()] = node_obj.id
                adj_out.setdefault(node_obj.id, [])
                adj_in.setdefault(node_obj.id, [])

        # 2. Index Edges & Adjacency
        for e in raw_edges:
            if isinstance(e, GraphEdge):
                edge_obj = e
            else:
                src = str(e.get("source", "")).strip()
                tgt = str(e.get("target", "")).strip()
                rel = str(e.get("relation", "CONNECTED_TO")).strip().upper()
                conf = float(e.get("confidence", 1.0))
                reserved = {"source", "target", "relation", "confidence", "properties"}
                props = {k: v for k, v in e.items() if k not in reserved}
                if "properties" in e and isinstance(e["properties"], dict):
                    props.update(e["properties"])
                edge_obj = GraphEdge(
                    source=src,
                    target=tgt,
                    relation=rel,
                    confidence=conf,
                    properties=props,
                )

            if edge_obj.source in nodes_by_id or edge_obj.target in nodes_by_id:
                all_edges.append(edge_obj)
                adj_out.setdefault(edge_obj.source, []).append(edge_obj)
                adj_in.setdefault(edge_obj.target, []).append(edge_obj)

        self._nodes_by_id = nodes_by_id
        self._nodes_by_name = nodes_by_name
        self._adj_out = adj_out
        self._adj_in = adj_in
        self._all_edges = all_edges
        self._is_indexed = True

        logger.info(
            "Graph query index built",
            nodes_count=len(self._nodes_by_id),
            edges_count=len(self._all_edges),
        )

    def _ensure_indexed(self) -> None:
        """Lazy load and index graph data on first query execution."""
        if not self._is_indexed:
            self.reload()

    # ==========================================================================
    # Core Query Methods
    # ==========================================================================

    def get_node(self, node_id_or_name: str) -> Optional[GraphNode]:
        """
        Retrieve a single node by its stable ID, exact name, or case-insensitive query.
        Returns None if node does not exist.
        """
        self._ensure_indexed()
        if not node_id_or_name or not node_id_or_name.strip():
            return None

        clean_query = node_id_or_name.strip()
        clean_lower = clean_query.lower()

        # 1. Exact ID match (O(1))
        if clean_query in self._nodes_by_id:
            return self._nodes_by_id[clean_query]

        # 2. Exact Name match (O(1))
        if clean_lower in self._nodes_by_name:
            target_id = self._nodes_by_name[clean_lower]
            return self._nodes_by_id.get(target_id)

        # 3. Case-insensitive ID probe
        for nid, node in self._nodes_by_id.items():
            if nid.lower() == clean_lower:
                return node

        # 4. Substring partial fallback match
        for nid, node in self._nodes_by_id.items():
            if clean_lower in nid.lower() or clean_lower in node.name.lower():
                return node

        return None

    def query_node(self, node_id_or_name: str) -> QueryResult:
        """Execute structured node lookup query with performance metrics."""
        start = time.perf_counter()
        self._ensure_indexed()
        node = self.get_node(node_id_or_name)
        duration_ms = (time.perf_counter() - start) * 1000.0

        nodes = [node] if node else []
        res = QueryResult(
            query_type=QueryType.NODE_LOOKUP,
            parameters={"node_id_or_name": node_id_or_name},
            nodes=nodes,
            edges=[],
            count=len(nodes),
            nodes_count=len(nodes),
            edges_count=0,
            visited_nodes_count=1 if node else 0,
            execution_ms=round(duration_ms, 3),
            is_truncated=False,
            message="Node found" if node else f"Node '{node_id_or_name}' not found",
        )
        metrics_collector.record_graph_query(
            query_type="node_lookup",
            latency_seconds=duration_ms / 1000.0,
            nodes_visited=res.visited_nodes_count,
            nodes_returned=res.nodes_count,
            is_error=node is None,
        )
        return res

    def get_neighbors(
        self,
        node_id_or_name: str,
        relation: Optional[str] = None,
        node_label: Optional[str] = None,
        direction: Direction = Direction.BOTH,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> QueryResult:
        """
        Retrieve direct (1-hop) neighbors of a node with filtering by relationship type,
        neighbor node label, traversal direction, and supported node/edge properties.
        """
        start = time.perf_counter()
        self._ensure_indexed()

        target_node = self.get_node(node_id_or_name)
        if not target_node:
            duration_ms = (time.perf_counter() - start) * 1000.0
            metrics_collector.record_graph_query(
                query_type="neighbors",
                latency_seconds=duration_ms / 1000.0,
                is_error=True,
            )
            return QueryResult(
                query_type=QueryType.NEIGHBORS,
                parameters={
                    "node_id": node_id_or_name,
                    "relation": relation,
                    "direction": direction.value,
                    "limit": limit,
                },
                nodes=[],
                edges=[],
                count=0,
                nodes_count=0,
                edges_count=0,
                visited_nodes_count=0,
                execution_ms=round(duration_ms, 3),
                is_truncated=False,
                message=f"Root entity '{node_id_or_name}' not found in knowledge graph",
            )

        tid = target_node.id
        rel_filter = relation.upper().strip() if relation else None
        lbl_filter = node_label.strip() if node_label else None
        clamped_limit = min(max(1, limit), MAX_RESULT_LIMIT)

        candidate_edges: List[Tuple[GraphEdge, str]] = []  # (edge, neighbor_id)

        # Collect outgoing edges
        if direction in (Direction.OUTGOING, Direction.BOTH):
            for edge in self._adj_out.get(tid, []):
                if rel_filter and edge.relation != rel_filter:
                    continue
                candidate_edges.append((edge, edge.target))

        # Collect incoming edges
        if direction in (Direction.INCOMING, Direction.BOTH):
            for edge in self._adj_in.get(tid, []):
                if rel_filter and edge.relation != rel_filter:
                    continue
                candidate_edges.append((edge, edge.source))

        # Filter by neighbor node properties and labels
        matched_neighbors: Dict[str, GraphNode] = {}
        matched_edges: List[GraphEdge] = []
        is_truncated = False

        for edge, nbr_id in candidate_edges:
            nbr_node = self._nodes_by_id.get(nbr_id)
            if not nbr_node:
                continue

            if lbl_filter and nbr_node.label.lower() != lbl_filter.lower():
                continue

            if properties and not self._matches_properties(nbr_node, properties):
                continue

            if nbr_id not in matched_neighbors:
                if len(matched_neighbors) >= clamped_limit:
                    is_truncated = True
                    break
                matched_neighbors[nbr_id] = nbr_node

            matched_edges.append(edge)

        duration_ms = (time.perf_counter() - start) * 1000.0
        result_nodes = [target_node] + list(matched_neighbors.values())

        res = QueryResult(
            query_type=QueryType.NEIGHBORS,
            parameters={
                "node_id": tid,
                "relation": relation,
                "node_label": node_label,
                "direction": direction.value,
                "limit": clamped_limit,
            },
            nodes=result_nodes,
            edges=matched_edges,
            count=len(matched_neighbors),
            nodes_count=len(result_nodes),
            edges_count=len(matched_edges),
            visited_nodes_count=len(matched_neighbors) + 1,
            execution_ms=round(duration_ms, 3),
            is_truncated=is_truncated,
            message="Neighbors retrieved successfully",
        )

        metrics_collector.record_graph_query(
            query_type="neighbors",
            latency_seconds=duration_ms / 1000.0,
            nodes_visited=res.visited_nodes_count,
            nodes_returned=res.nodes_count,
            is_truncated=is_truncated,
        )
        return res

    def traverse(
        self,
        root_id_or_name: str,
        max_depth: int = DEFAULT_TRAVERSAL_DEPTH,
        relation: Optional[str] = None,
        node_label: Optional[str] = None,
        direction: Direction = Direction.BOTH,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> QueryResult:
        """
        Execute bounded multi-hop graph traversal starting from a root node up to max_depth.
        Guarantees safe cycle termination by tracking visited sets.
        Enforces maximum depth and total result limits.
        """
        start = time.perf_counter()
        self._ensure_indexed()

        root_node = self.get_node(root_id_or_name)
        if not root_node:
            duration_ms = (time.perf_counter() - start) * 1000.0
            metrics_collector.record_graph_query(
                query_type="traverse",
                latency_seconds=duration_ms / 1000.0,
                is_error=True,
            )
            return QueryResult(
                query_type=QueryType.TRAVERSE,
                parameters={
                    "root_id": root_id_or_name,
                    "max_depth": max_depth,
                    "relation": relation,
                    "direction": direction.value,
                    "limit": limit,
                },
                nodes=[],
                edges=[],
                count=0,
                nodes_count=0,
                edges_count=0,
                visited_nodes_count=0,
                execution_ms=round(duration_ms, 3),
                is_truncated=False,
                message=f"Root entity '{root_id_or_name}' not found",
            )

        root_id = root_node.id
        safe_depth = min(max(1, max_depth), MAX_ALLOWED_DEPTH)
        clamped_limit = min(max(1, limit), MAX_RESULT_LIMIT)
        rel_filter = relation.upper().strip() if relation else None
        lbl_filter = node_label.strip() if node_label else None

        visited_nodes: Set[str] = {root_id}
        traversed_edges: List[GraphEdge] = []
        edge_signatures: Set[str] = set()
        is_truncated = False

        # Frontier BFS level-order traversal
        current_frontier: Set[str] = {root_id}

        for _depth in range(1, safe_depth + 1):
            next_frontier: Set[str] = set()
            for curr_id in current_frontier:
                adjacent_edges: List[Tuple[GraphEdge, str]] = []

                if direction in (Direction.OUTGOING, Direction.BOTH):
                    for e in self._adj_out.get(curr_id, []):
                        adjacent_edges.append((e, e.target))
                if direction in (Direction.INCOMING, Direction.BOTH):
                    for e in self._adj_in.get(curr_id, []):
                        adjacent_edges.append((e, e.source))

                for edge, neighbor_id in adjacent_edges:
                    if rel_filter and edge.relation != rel_filter:
                        continue

                    neighbor_node = self._nodes_by_id.get(neighbor_id)
                    if not neighbor_node:
                        continue

                    if lbl_filter and neighbor_node.label.lower() != lbl_filter.lower():
                        continue

                    if properties and not self._matches_properties(neighbor_node, properties):
                        continue

                    sig = f"{edge.source}->{edge.relation}->{edge.target}"
                    if sig not in edge_signatures:
                        edge_signatures.add(sig)
                        traversed_edges.append(edge)

                    if neighbor_id not in visited_nodes:
                        if len(visited_nodes) >= clamped_limit:
                            is_truncated = True
                            break
                        visited_nodes.add(neighbor_id)
                        next_frontier.add(neighbor_id)

                if is_truncated:
                    break

            if is_truncated or not next_frontier:
                break
            current_frontier = next_frontier

        duration_ms = (time.perf_counter() - start) * 1000.0
        nodes = [self._nodes_by_id[nid] for nid in visited_nodes if nid in self._nodes_by_id]

        res = QueryResult(
            query_type=QueryType.TRAVERSE,
            parameters={
                "root_id": root_id,
                "max_depth": safe_depth,
                "relation": relation,
                "node_label": node_label,
                "direction": direction.value,
                "limit": clamped_limit,
            },
            nodes=nodes,
            edges=traversed_edges,
            count=len(nodes),
            nodes_count=len(nodes),
            edges_count=len(traversed_edges),
            visited_nodes_count=len(visited_nodes),
            execution_ms=round(duration_ms, 3),
            is_truncated=is_truncated,
            message="Traversal completed",
        )

        metrics_collector.record_graph_query(
            query_type="traverse",
            latency_seconds=duration_ms / 1000.0,
            nodes_visited=res.visited_nodes_count,
            nodes_returned=res.nodes_count,
            is_truncated=is_truncated,
        )
        return res

    def find_path(
        self,
        from_node_query: str,
        to_node_query: str,
        max_depth: int = DEFAULT_PATH_DEPTH,
        relation: Optional[str] = None,
        direction: Direction = Direction.BOTH,
    ) -> QueryResult:
        """
        Discover the shortest path connecting from_node to to_node using Breadth-First Search.
        Prevents infinite loops from cycles and enforces maximum depth guards.
        """
        start = time.perf_counter()
        self._ensure_indexed()

        start_node = self.get_node(from_node_query)
        end_node = self.get_node(to_node_query)

        if not start_node or not end_node:
            duration_ms = (time.perf_counter() - start) * 1000.0
            missing = []
            if not start_node:
                missing.append(f"from_node '{from_node_query}'")
            if not end_node:
                missing.append(f"to_node '{to_node_query}'")

            metrics_collector.record_graph_query(
                query_type="path",
                latency_seconds=duration_ms / 1000.0,
                is_error=True,
            )
            return QueryResult(
                query_type=QueryType.PATH,
                parameters={
                    "from_node": from_node_query,
                    "to_node": to_node_query,
                    "max_depth": max_depth,
                },
                nodes=[],
                edges=[],
                paths=[PathResult(found=False, length=0, path=[], nodes=[], edges=[])],
                count=0,
                nodes_count=0,
                edges_count=0,
                visited_nodes_count=0,
                execution_ms=round(duration_ms, 3),
                is_truncated=False,
                message=f"Path discovery failed: {', '.join(missing)} not found",
            )

        start_id = start_node.id
        end_id = end_node.id
        safe_depth = min(max(1, max_depth), MAX_ALLOWED_DEPTH)
        rel_filter = relation.upper().strip() if relation else None

        # Trivial path to self
        if start_id == end_id:
            duration_ms = (time.perf_counter() - start) * 1000.0
            path_res = PathResult(
                found=True,
                length=0,
                path=[start_id],
                nodes=[start_node],
                edges=[],
                steps=[],
            )
            return QueryResult(
                query_type=QueryType.PATH,
                parameters={
                    "from_node": start_id,
                    "to_node": end_id,
                    "max_depth": safe_depth,
                },
                nodes=[start_node],
                edges=[],
                paths=[path_res],
                count=1,
                nodes_count=1,
                edges_count=0,
                visited_nodes_count=1,
                execution_ms=round(duration_ms, 3),
                is_truncated=False,
                message="Path to self found (length 0)",
            )

        # BFS state: (current_node_id, [node_ids], [edges], [tokens], [steps])
        queue: deque[Tuple[str, List[str], List[GraphEdge], List[str], List[PathStep]]] = deque([
            (start_id, [start_id], [], [start_id], [])
        ])
        visited_nodes: Set[str] = {start_id}
        discovered_path: Optional[PathResult] = None

        while queue:
            curr_id, path_nids, path_edges, path_tokens, path_steps = queue.popleft()

            if len(path_edges) >= safe_depth:
                continue

            # Gather adjacent edges according to direction
            neighbors: List[Tuple[GraphEdge, str, str]] = []  # (edge, neighbor_id, direction_str)
            if direction in (Direction.OUTGOING, Direction.BOTH):
                for e in self._adj_out.get(curr_id, []):
                    neighbors.append((e, e.target, "outgoing"))
            if direction in (Direction.INCOMING, Direction.BOTH):
                for e in self._adj_in.get(curr_id, []):
                    neighbors.append((e, e.source, "incoming"))

            for edge, neighbor_id, dir_str in neighbors:
                if rel_filter and edge.relation != rel_filter:
                    continue

                if neighbor_id not in self._nodes_by_id:
                    continue

                rel_token = f"-[:{edge.relation}]->" if dir_str == "outgoing" else f"<-[:{edge.relation}]-"
                step = PathStep(
                    from_node=curr_id,
                    relation=edge.relation,
                    direction=dir_str,
                    to_node=neighbor_id,
                    confidence=edge.confidence,
                )

                if neighbor_id == end_id:
                    full_nids = path_nids + [end_id]
                    full_edges = path_edges + [edge]
                    full_tokens = path_tokens + [rel_token, end_id]
                    full_steps = path_steps + [step]
                    nodes_list = [self._nodes_by_id[nid] for nid in full_nids if nid in self._nodes_by_id]

                    discovered_path = PathResult(
                        found=True,
                        length=len(full_edges),
                        path=full_tokens,
                        nodes=nodes_list,
                        edges=full_edges,
                        steps=full_steps,
                    )
                    break

                if neighbor_id not in visited_nodes:
                    visited_nodes.add(neighbor_id)
                    queue.append((
                        neighbor_id,
                        path_nids + [neighbor_id],
                        path_edges + [edge],
                        path_tokens + [rel_token, neighbor_id],
                        path_steps + [step],
                    ))

            if discovered_path:
                break

        duration_ms = (time.perf_counter() - start) * 1000.0

        if discovered_path:
            res = QueryResult(
                query_type=QueryType.PATH,
                parameters={
                    "from_node": start_id,
                    "to_node": end_id,
                    "max_depth": safe_depth,
                    "relation": relation,
                },
                nodes=discovered_path.nodes,
                edges=discovered_path.edges,
                paths=[discovered_path],
                count=1,
                nodes_count=len(discovered_path.nodes),
                edges_count=len(discovered_path.edges),
                visited_nodes_count=len(visited_nodes),
                execution_ms=round(duration_ms, 3),
                is_truncated=False,
                message=f"Shortest path discovered (hops: {discovered_path.length})",
            )
        else:
            res = QueryResult(
                query_type=QueryType.PATH,
                parameters={
                    "from_node": start_id,
                    "to_node": end_id,
                    "max_depth": safe_depth,
                    "relation": relation,
                },
                nodes=[],
                edges=[],
                paths=[PathResult(found=False, length=0, path=[], nodes=[], edges=[])],
                count=0,
                nodes_count=0,
                edges_count=0,
                visited_nodes_count=len(visited_nodes),
                execution_ms=round(duration_ms, 3),
                is_truncated=False,
                message=f"No connecting path found within {safe_depth} hops",
            )

        metrics_collector.record_graph_query(
            query_type="path",
            latency_seconds=duration_ms / 1000.0,
            nodes_visited=res.visited_nodes_count,
            nodes_returned=res.nodes_count,
            is_error=False,
        )
        return res

    def extract_subgraph(
        self,
        seed_node_queries: List[str],
        hops: int = 1,
        relation: Optional[str] = None,
        node_label: Optional[str] = None,
        direction: Direction = Direction.BOTH,
        limit: int = 200,
    ) -> QueryResult:
        """
        Extract an induced subgraph centered around one or more seed nodes expanded by `hops`.
        Returns all participating nodes and connecting edges with deduplication.
        """
        start = time.perf_counter()
        self._ensure_indexed()

        clamped_hops = min(max(0, hops), 5)
        clamped_limit = min(max(1, limit), 1000)

        seed_nodes: Dict[str, GraphNode] = {}
        for q in seed_node_queries:
            node = self.get_node(q)
            if node:
                seed_nodes[node.id] = node

        if not seed_nodes:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return QueryResult(
                query_type=QueryType.SUBGRAPH,
                parameters={"seed_nodes": seed_node_queries, "hops": hops, "limit": limit},
                nodes=[],
                edges=[],
                count=0,
                nodes_count=0,
                edges_count=0,
                visited_nodes_count=0,
                execution_ms=round(duration_ms, 3),
                is_truncated=False,
                message="No seed nodes matched",
            )

        visited_nodes: Set[str] = set(seed_nodes.keys())
        current_frontier: Set[str] = set(seed_nodes.keys())
        rel_filter = relation.upper().strip() if relation else None
        lbl_filter = node_label.strip() if node_label else None
        is_truncated = False

        for _ in range(clamped_hops):
            next_frontier: Set[str] = set()
            for curr_id in current_frontier:
                adj_edges: List[Tuple[GraphEdge, str]] = []
                if direction in (Direction.OUTGOING, Direction.BOTH):
                    for e in self._adj_out.get(curr_id, []):
                        adj_edges.append((e, e.target))
                if direction in (Direction.INCOMING, Direction.BOTH):
                    for e in self._adj_in.get(curr_id, []):
                        adj_edges.append((e, e.source))

                for edge, nbr_id in adj_edges:
                    if rel_filter and edge.relation != rel_filter:
                        continue
                    nbr_node = self._nodes_by_id.get(nbr_id)
                    if not nbr_node:
                        continue
                    if lbl_filter and nbr_node.label.lower() != lbl_filter.lower():
                        continue

                    if nbr_id not in visited_nodes:
                        if len(visited_nodes) >= clamped_limit:
                            is_truncated = True
                            break
                        visited_nodes.add(nbr_id)
                        next_frontier.add(nbr_id)

                if is_truncated:
                    break

            if is_truncated or not next_frontier:
                break
            current_frontier = next_frontier

        # Collect all connecting edges within the induced subgraph
        subgraph_edges: List[GraphEdge] = []
        edge_signatures: Set[str] = set()

        for nid in visited_nodes:
            for edge in self._adj_out.get(nid, []):
                if edge.target in visited_nodes:
                    if rel_filter and edge.relation != rel_filter:
                        continue
                    sig = f"{edge.source}->{edge.relation}->{edge.target}"
                    if sig not in edge_signatures:
                        edge_signatures.add(sig)
                        subgraph_edges.append(edge)

        nodes = [self._nodes_by_id[nid] for nid in visited_nodes if nid in self._nodes_by_id]
        duration_ms = (time.perf_counter() - start) * 1000.0

        res = QueryResult(
            query_type=QueryType.SUBGRAPH,
            parameters={
                "seed_nodes": list(seed_nodes.keys()),
                "hops": clamped_hops,
                "relation": relation,
                "node_label": node_label,
                "limit": clamped_limit,
            },
            nodes=nodes,
            edges=subgraph_edges,
            count=len(nodes),
            nodes_count=len(nodes),
            edges_count=len(subgraph_edges),
            visited_nodes_count=len(visited_nodes),
            execution_ms=round(duration_ms, 3),
            is_truncated=is_truncated,
            message="Subgraph extracted successfully",
        )

        metrics_collector.record_graph_query(
            query_type="subgraph",
            latency_seconds=duration_ms / 1000.0,
            nodes_visited=res.visited_nodes_count,
            nodes_returned=res.nodes_count,
            is_truncated=is_truncated,
        )
        return res

    def find_relationships(
        self,
        relation: Optional[str] = None,
        source_label: Optional[str] = None,
        target_label: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> QueryResult:
        """
        Query global relationships matching pattern criteria:
        (Source:source_label)-[:relation]->(Target:target_label).
        """
        start = time.perf_counter()
        self._ensure_indexed()

        rel_filter = relation.upper().strip() if relation else None
        s_lbl = source_label.lower().strip() if source_label else None
        t_lbl = target_label.lower().strip() if target_label else None
        clamped_limit = min(max(1, limit), MAX_RESULT_LIMIT)

        matched_edges: List[GraphEdge] = []
        matched_node_ids: Set[str] = set()
        is_truncated = False

        for edge in self._all_edges:
            if rel_filter and edge.relation != rel_filter:
                continue

            if edge.confidence < min_confidence:
                continue

            src_node = self._nodes_by_id.get(edge.source)
            tgt_node = self._nodes_by_id.get(edge.target)

            if not src_node or not tgt_node:
                continue

            if s_lbl and src_node.label.lower() != s_lbl:
                continue

            if t_lbl and tgt_node.label.lower() != t_lbl:
                continue

            if len(matched_edges) >= clamped_limit:
                is_truncated = True
                break

            matched_edges.append(edge)
            matched_node_ids.add(src_node.id)
            matched_node_ids.add(tgt_node.id)

        nodes = [self._nodes_by_id[nid] for nid in matched_node_ids if nid in self._nodes_by_id]
        duration_ms = (time.perf_counter() - start) * 1000.0

        res = QueryResult(
            query_type=QueryType.RELATIONSHIP,
            parameters={
                "relation": relation,
                "source_label": source_label,
                "target_label": target_label,
                "min_confidence": min_confidence,
                "limit": clamped_limit,
            },
            nodes=nodes,
            edges=matched_edges,
            count=len(matched_edges),
            nodes_count=len(nodes),
            edges_count=len(matched_edges),
            visited_nodes_count=len(nodes),
            execution_ms=round(duration_ms, 3),
            is_truncated=is_truncated,
            message="Relationships queried successfully",
        )

        metrics_collector.record_graph_query(
            query_type="relationship",
            latency_seconds=duration_ms / 1000.0,
            nodes_visited=res.visited_nodes_count,
            nodes_returned=res.nodes_count,
            is_truncated=is_truncated,
        )
        return res

    def _matches_properties(self, node: GraphNode, properties: Dict[str, Any]) -> bool:
        """Check if a node satisfies dictionary property constraints."""
        for key, expected_val in properties.items():
            actual_val = node.get(key)
            if actual_val is None:
                return False
            if isinstance(expected_val, str) and isinstance(actual_val, str):
                if expected_val.lower() not in actual_val.lower():
                    return False
            elif actual_val != expected_val:
                return False
        return True


# Global singleton instance
graph_query_engine = GraphQueryEngine()
