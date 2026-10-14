"""
================================================================================
GRAPHONE PIPELINE: GRAPH + VECTOR HYBRID RETRIEVAL SERVICE
================================================================================
Combines dense vector similarity search with bounded Knowledge Graph multi-hop
traversals to deliver unified, relationship-aware, deduplicated entity discovery
with deterministic scoring, reciprocal rank fusion, and provenance evidence.
================================================================================
"""

import re
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from src.graph.models import Direction, GraphEdge, GraphNode
from src.graph.query_engine import GraphQueryEngine, graph_query_engine
from src.observability.metrics import metrics_collector
from src.retrieval.models import (
    HybridRetrievalRequest,
    HybridRetrievalResponse,
    RankingStrategy,
    RetrievalCandidate,
    RetrievalSource,
)
from src.vector.vector_store import VectorStoreManager, vector_store

logger = structlog.get_logger(__name__)


def normalize_entity_type(raw_type: str) -> str:
    """Normalize raw entity type label to standard lowercase format."""
    if not raw_type:
        return "entity"
    t = raw_type.lower().strip()
    if t in ("researchpaper", "research_paper", "paper"):
        return "research_paper"
    return t


def normalize_name_for_matching(name: str) -> str:
    """Normalize entity name for stable cross-engine identity matching."""
    if not name:
        return ""
    clean = re.sub(r"[^\w\s]", "", name.lower())
    return " ".join(clean.split())


class HybridRetrievalService:
    """
    Hybrid Retrieval Engine orchestrating:
    1. Vector similarity search across dense embeddings.
    2. Knowledge Graph expansion starting from vector-retrieved seeds.
    3. Stable-ID candidate deduplication & merging.
    4. Deterministic scoring (weighted linear fusion or reciprocal rank fusion).
    5. Path provenance evidence & structured ranking explanations.
    """

    def __init__(
        self,
        vector_mgr: Optional[VectorStoreManager] = None,
        graph_engine: Optional[GraphQueryEngine] = None,
    ):
        self.vector_store = vector_mgr or vector_store
        self.graph_engine = graph_engine or graph_query_engine

    def _ensure_initialized(self) -> None:
        """Ensure vector store and graph query engine indices are warm."""
        if not getattr(self.vector_store, "_is_indexed", False):
            self.vector_store.index_all_records()
        if not getattr(self.graph_engine, "_is_indexed", False):
            self.graph_engine.reload()

    def _find_graph_node_for_vector_candidate(
        self, candidate_id: str, title: str
    ) -> Optional[GraphNode]:
        """Resolve a vector candidate to its underlying GraphNode if one exists."""
        # 1. Exact node lookup by ID
        node = self.graph_engine.get_node(candidate_id)
        if node:
            return node

        # 2. Lookup with 'node-' prefix prepended
        if not candidate_id.startswith("node-"):
            node = self.graph_engine.get_node(f"node-{candidate_id}")
            if node:
                return node

        # 3. Lookup by exact title/name
        if title:
            node = self.graph_engine.get_node(title)
            if node:
                return node

        # 4. Lookup by normalized title
        norm_title = normalize_name_for_matching(title)
        if norm_title:
            for n in self.graph_engine._nodes_by_id.values():
                if normalize_name_for_matching(n.name) == norm_title:
                    return n

        return None

    def _get_stable_id(
        self,
        raw_id: str,
        title: str,
        cand_type: str,
        graph_node: Optional[GraphNode] = None,
    ) -> str:
        """Derive a canonical stable identifier across vector and graph representations."""
        if graph_node:
            return graph_node.id
        if raw_id.startswith("node-"):
            return raw_id
        slug = re.sub(r"[^\w\s-]", "", title.lower()).strip().replace(" ", "-")
        return f"node-{normalize_entity_type(cand_type)}-{slug}" if slug else f"node-{raw_id}"

    def search_hybrid(self, request: HybridRetrievalRequest) -> HybridRetrievalResponse:
        """Execute genuine graph + vector hybrid retrieval combining both pathways."""
        start_time = time.perf_counter()
        self._ensure_initialized()

        clean_query = request.query.strip()
        if not clean_query:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return HybridRetrievalResponse(
                query=request.query,
                candidate_type=request.candidate_type,
                total_candidates=0,
                vector_candidates_count=0,
                graph_candidates_count=0,
                merged_candidates_count=0,
                returned_count=0,
                results=[],
                execution_ms=round(duration_ms, 3),
                is_truncated=False,
                ranking_strategy=request.ranking_strategy.value,
                message="Empty query provided",
            )

        # ----------------------------------------------------------------------
        # STAGE 1: VECTOR RETRIEVAL
        # ----------------------------------------------------------------------
        target_record_type = request.candidate_type if request.candidate_type != "all" else None
        vector_matches = self.vector_store.search(
            query=clean_query,
            record_type=target_record_type,
            limit=request.vector_limit,
        )

        vector_candidates_dict: Dict[str, Dict[str, Any]] = {}
        vector_rank_map: Dict[str, int] = {}
        seed_nodes_for_expansion: List[Tuple[GraphNode, float, str]] = []

        for rank_idx, item in enumerate(vector_matches, start=1):
            raw_id = item["id"]
            title = item.get("title") or item.get("payload", {}).get("name") or raw_id
            ctype = normalize_entity_type(item.get("record_type") or "entity")
            sim_score = float(item.get("similarity_score", 0.0))

            matched_node = self._find_graph_node_for_vector_candidate(raw_id, title)
            stable_id = self._get_stable_id(raw_id, title, ctype, matched_node)

            vector_candidates_dict[stable_id] = {
                "id": stable_id,
                "title": title,
                "candidate_type": ctype,
                "vector_score": sim_score,
                "payload": item.get("payload", {}),
                "matched_node": matched_node,
            }
            vector_rank_map[stable_id] = rank_idx

            if matched_node:
                seed_nodes_for_expansion.append((matched_node, sim_score, title))

        # ----------------------------------------------------------------------
        # STAGE 2: GRAPH EXPANSION FROM VECTOR SEEDS
        # ----------------------------------------------------------------------
        graph_candidates_dict: Dict[str, Dict[str, Any]] = {}
        graph_rank_map: Dict[str, int] = {}
        is_graph_truncated = False
        allowed_edge_types = (
            {e.upper().strip() for e in request.edge_types}
            if request.edge_types
            else None
        )

        # BFS level-by-level queue for multi-hop expansion
        # Queue item: (current_node_id, distance, seed_node_id, seed_title, seed_vector_score, path_edges)
        visited_nodes: Set[str] = set()
        expansion_queue: deque = deque()

        for seed_node, seed_score, seed_title in seed_nodes_for_expansion:
            visited_nodes.add(seed_node.id)
            expansion_queue.append((seed_node.id, 0, seed_node.id, seed_title, seed_score, []))

        while expansion_queue:
            curr_id, dist, seed_id, seed_title, seed_score, path_edges = expansion_queue.popleft()

            if dist >= request.graph_depth:
                continue

            # Discover adjacent directed edges
            adjacent_edges: List[Tuple[GraphEdge, str]] = []
            if request.direction in (Direction.OUTGOING, Direction.BOTH):
                for e in self.graph_engine._adj_out.get(curr_id, []):
                    adjacent_edges.append((e, e.target))
            if request.direction in (Direction.INCOMING, Direction.BOTH):
                for e in self.graph_engine._adj_in.get(curr_id, []):
                    adjacent_edges.append((e, e.source))

            for edge, neighbor_id in adjacent_edges:
                if allowed_edge_types and edge.relation.upper() not in allowed_edge_types:
                    continue

                neighbor_node = self.graph_engine._nodes_by_id.get(neighbor_id)
                if not neighbor_node:
                    continue

                new_dist = dist + 1
                new_path_edges = path_edges + [edge]

                # Calculate cumulative path confidence and distance decay:
                # decay(d) = 1.0 / (1.0 + 0.5 * d)
                cum_conf = 1.0
                for pe in new_path_edges:
                    cum_conf *= float(getattr(pe, "confidence", 1.0))

                dist_decay = 1.0 / (1.0 + 0.5 * new_dist)
                calculated_graph_score = round(seed_score * dist_decay * cum_conf, 4)

                # Format human-readable relationship path description
                src_node = self.graph_engine._nodes_by_id.get(edge.source)
                tgt_node = self.graph_engine._nodes_by_id.get(edge.target)
                src_name = src_node.name if src_node else edge.source
                tgt_name = tgt_node.name if tgt_node else edge.target
                edge_desc = f"{src_name} --[{edge.relation} ({round(edge.confidence, 2)})]--> {tgt_name}"

                if neighbor_id not in graph_candidates_dict:
                    if len(graph_candidates_dict) >= request.max_graph_candidates:
                        is_graph_truncated = True
                        break

                    graph_candidates_dict[neighbor_id] = {
                        "id": neighbor_id,
                        "title": neighbor_node.name,
                        "candidate_type": normalize_entity_type(neighbor_node.label),
                        "graph_score": calculated_graph_score,
                        "graph_distance": new_dist,
                        "path_evidence": [edge_desc],
                        "node": neighbor_node,
                        "payload": neighbor_node.to_dict(),
                    }
                else:
                    existing = graph_candidates_dict[neighbor_id]
                    # Update with higher score if discovered via a stronger path
                    if calculated_graph_score > existing["graph_score"]:
                        existing["graph_score"] = calculated_graph_score
                        existing["graph_distance"] = min(existing["graph_distance"], new_dist)
                    if edge_desc not in existing["path_evidence"]:
                        existing["path_evidence"].append(edge_desc)

                if neighbor_id not in visited_nodes:
                    visited_nodes.add(neighbor_id)
                    if new_dist < request.graph_depth:
                        expansion_queue.append((
                            neighbor_id,
                            new_dist,
                            seed_id,
                            seed_title,
                            seed_score,
                            new_path_edges,
                        ))

            if is_graph_truncated:
                break

        # Assign graph ranks for RRF ranking
        sorted_graph_cand_ids = sorted(
            graph_candidates_dict.keys(),
            key=lambda cid: (-graph_candidates_dict[cid]["graph_score"], cid),
        )
        for g_rank, cid in enumerate(sorted_graph_cand_ids, start=1):
            graph_rank_map[cid] = g_rank

        # ----------------------------------------------------------------------
        # STAGE 3: CANDIDATE MERGING & DEDUPLICATION
        # ----------------------------------------------------------------------
        all_candidate_ids: Set[str] = set(vector_candidates_dict.keys()) | set(graph_candidates_dict.keys())
        merged_candidates: List[RetrievalCandidate] = []

        total_w = request.vector_weight + request.graph_weight
        w_v = (request.vector_weight / total_w) if total_w > 0 else 0.5
        w_g = (request.graph_weight / total_w) if total_w > 0 else 0.5

        for cid in all_candidate_ids:
            v_info = vector_candidates_dict.get(cid)
            g_info = graph_candidates_dict.get(cid)

            # Determine sources
            if v_info and g_info:
                sources = [RetrievalSource.VECTOR, RetrievalSource.GRAPH]
                primary_source = RetrievalSource.BOTH
            elif v_info:
                sources = [RetrievalSource.VECTOR]
                primary_source = RetrievalSource.VECTOR
            else:
                sources = [RetrievalSource.GRAPH]
                primary_source = RetrievalSource.GRAPH

            title = (v_info["title"] if v_info else g_info["title"])
            ctype = (v_info["candidate_type"] if v_info else g_info["candidate_type"])

            v_score: Optional[float] = v_info["vector_score"] if v_info else None
            g_score: Optional[float] = g_info["graph_score"] if g_info else None
            g_dist: Optional[int] = g_info["graph_distance"] if g_info else None
            path_ev: List[str] = g_info["path_evidence"] if g_info else []

            # Combined payload
            payload: Dict[str, Any] = {}
            if g_info and "payload" in g_info:
                payload.update(g_info["payload"])
            if v_info and "payload" in v_info:
                payload.update(v_info["payload"])

            # ------------------------------------------------------------------
            # STAGE 4: RANKING & DETERMINISTIC SCORING
            # ------------------------------------------------------------------
            if request.ranking_strategy == RankingStrategy.RRF:
                # Reciprocal Rank Fusion: score = w_v / (k + R_v) + w_g / (k + R_g)
                rrf_v = (w_v / (request.rrf_k + vector_rank_map[cid])) if cid in vector_rank_map else 0.0
                rrf_g = (w_g / (request.rrf_k + graph_rank_map[cid])) if cid in graph_rank_map else 0.0
                final_score = round(rrf_v + rrf_g, 6)
                explanation = (
                    f"RRF (k={request.rrf_k}): "
                    + (f"Vector rank #{vector_rank_map[cid]} (rrf: {round(rrf_v, 6)})" if cid in vector_rank_map else "No vector match")
                    + " + "
                    + (f"Graph rank #{graph_rank_map[cid]} (rrf: {round(rrf_g, 6)})" if cid in graph_rank_map else "No graph match")
                    + f" -> Final RRF score {final_score}."
                )
            else:
                # Weighted Linear Fusion: final = w_v * v_score + w_g * g_score
                v_component = (v_score or 0.0) * w_v
                g_component = (g_score or 0.0) * w_g
                final_score = round(v_component + g_component, 6)

                if primary_source == RetrievalSource.BOTH:
                    explanation = (
                        f"Weighted fusion (w_v={round(w_v, 2)}, w_g={round(w_g, 2)}): "
                        f"Vector similarity {round(v_score, 4)} ({round(v_component, 4)}) + "
                        f"Graph score {round(g_score, 4)} ({round(g_component, 4)}) at distance {g_dist} "
                        f"-> Final score {final_score}."
                    )
                elif primary_source == RetrievalSource.VECTOR:
                    explanation = (
                        f"Vector-only match: Vector similarity {round(v_score, 4)} "
                        f"(weighted contribution: {round(v_component, 4)}, weight: {round(w_v, 2)}). "
                        f"No connecting graph path found -> Final score {final_score}."
                    )
                else:
                    evidence_str = f" via '{path_ev[0]}'" if path_ev else ""
                    explanation = (
                        f"Graph-only discovery{evidence_str}: Graph score {round(g_score, 4)} "
                        f"(weighted contribution: {round(g_component, 4)}, weight: {round(w_g, 2)}) at distance {g_dist}. "
                        f"Not in top vector hits -> Final score {final_score}."
                    )

            candidate = RetrievalCandidate(
                id=cid,
                candidate_type=ctype,
                title=title,
                text=payload.get("description") or payload.get("abstract") or payload.get("tagline"),
                vector_score=v_score,
                graph_score=g_score,
                graph_distance=g_dist,
                path_evidence=path_ev,
                sources=sources,
                source=primary_source,
                final_score=final_score,
                ranking_explanation=explanation,
                payload=payload,
            )
            merged_candidates.append(candidate)

        # ----------------------------------------------------------------------
        # STAGE 5: FILTERING, DETERMINISTIC TIE-BREAKING, AND SIZING
        # ----------------------------------------------------------------------
        # Apply candidate_type filter if specified
        if request.candidate_type and request.candidate_type != "all":
            req_ctype = normalize_entity_type(request.candidate_type)
            merged_candidates = [
                c for c in merged_candidates
                if normalize_entity_type(c.candidate_type) == req_ctype
            ]

        # Apply min_score threshold
        if request.min_score > 0.0:
            merged_candidates = [
                c for c in merged_candidates
                if c.final_score >= request.min_score
            ]

        # Deterministic sort: (-final_score, -vector_score, -graph_score, id)
        merged_candidates.sort(
            key=lambda c: (
                -round(c.final_score, 6),
                -round(c.vector_score or 0.0, 6),
                -round(c.graph_score or 0.0, 6),
                c.id,
            )
        )

        total_merged_count = len(merged_candidates)
        final_results = merged_candidates[: request.limit]
        is_result_truncated = total_merged_count > request.limit

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Record system observability metrics
        metrics_collector.record_hybrid_query(
            latency_seconds=duration_ms / 1000.0,
            vector_candidates=len(vector_candidates_dict),
            graph_candidates=len(graph_candidates_dict),
            merged_candidates=total_merged_count,
            final_results=len(final_results),
            is_graph_truncated=is_graph_truncated,
            is_error=False,
        )

        return HybridRetrievalResponse(
            query=clean_query,
            candidate_type=request.candidate_type,
            total_candidates=len(final_results),
            vector_candidates_count=len(vector_candidates_dict),
            graph_candidates_count=len(graph_candidates_dict),
            merged_candidates_count=total_merged_count,
            returned_count=len(final_results),
            results=final_results,
            execution_ms=round(duration_ms, 3),
            is_truncated=is_result_truncated or is_graph_truncated,
            ranking_strategy=request.ranking_strategy.value,
            message="Hybrid retrieval completed successfully",
        )

    def search_vector_only(
        self,
        query: str,
        candidate_type: str = "all",
        limit: int = 10,
    ) -> List[RetrievalCandidate]:
        """Baseline Vector-Only retrieval for direct comparative evaluation."""
        self._ensure_initialized()
        clean_query = query.strip()
        if not clean_query:
            return []

        target_record_type = candidate_type if candidate_type != "all" else None
        matches = self.vector_store.search(
            query=clean_query,
            record_type=target_record_type,
            limit=limit,
        )

        results: List[RetrievalCandidate] = []
        for item in matches:
            raw_id = item["id"]
            title = item.get("title") or raw_id
            ctype = normalize_entity_type(item.get("record_type") or "entity")
            sim_score = float(item.get("similarity_score", 0.0))
            matched_node = self._find_graph_node_for_vector_candidate(raw_id, title)
            stable_id = self._get_stable_id(raw_id, title, ctype, matched_node)

            candidate = RetrievalCandidate(
                id=stable_id,
                candidate_type=ctype,
                title=title,
                text=item.get("payload", {}).get("description") or item.get("payload", {}).get("abstract"),
                vector_score=sim_score,
                graph_score=None,
                graph_distance=None,
                path_evidence=[],
                sources=[RetrievalSource.VECTOR],
                source=RetrievalSource.VECTOR,
                final_score=round(sim_score, 6),
                ranking_explanation=f"Vector-only baseline: Cosine similarity {round(sim_score, 4)}.",
                payload=item.get("payload", {}),
            )
            results.append(candidate)

        # Deterministic sort
        results.sort(key=lambda c: (-round(c.final_score, 6), c.id))
        return results[:limit]

    def search_graph_only(
        self,
        seed_query: str,
        depth: int = 1,
        limit: int = 10,
        edge_types: Optional[List[str]] = None,
        direction: Direction = Direction.BOTH,
    ) -> List[RetrievalCandidate]:
        """Baseline Graph-Only retrieval for direct comparative evaluation."""
        self._ensure_initialized()
        clean_seed = seed_query.strip()
        if not clean_seed:
            return []

        # Find seed node in graph
        seed_node = self.graph_engine.get_node(clean_seed)
        if not seed_node:
            return []

        # Multi-hop traversal using GraphQueryEngine
        trav_res = self.graph_engine.traverse(
            root_id_or_name=seed_node.id,
            max_depth=depth,
            relation=edge_types[0] if edge_types and len(edge_types) == 1 else None,
            direction=direction,
            limit=limit * 2,
        )

        results: List[RetrievalCandidate] = []
        for node in trav_res.nodes:
            is_root = node.id == seed_node.id
            dist = 0 if is_root else 1
            graph_score = 1.0 if is_root else round(1.0 / (1.0 + 0.5 * dist), 4)

            # Find matching edges connecting to this node
            edge_ev = [
                f"{e.source} --[{e.relation}]--> {e.target}"
                for e in trav_res.edges
                if e.target == node.id or e.source == node.id
            ]

            results.append(
                RetrievalCandidate(
                    id=node.id,
                    candidate_type=normalize_entity_type(node.label),
                    title=node.name,
                    text=node.properties.get("description") or node.properties.get("abstract"),
                    vector_score=None,
                    graph_score=graph_score,
                    graph_distance=dist,
                    path_evidence=edge_ev[:3],
                    sources=[RetrievalSource.GRAPH],
                    source=RetrievalSource.GRAPH,
                    final_score=round(graph_score, 6),
                    ranking_explanation=f"Graph-only baseline: Distance {dist} from root '{seed_node.name}' -> Score {graph_score}.",
                    payload=node.to_dict(),
                )
            )

        results.sort(key=lambda c: (-round(c.final_score, 6), c.id))
        return results[:limit]


# Global singleton instance
hybrid_retrieval_service = HybridRetrievalService()
