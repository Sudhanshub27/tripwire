import asyncio
import json
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from config.settings import configure_logging, settings
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.dashboard.processed_reader import (
    get_all_processed_records,
    get_live_benchmark_metrics,
    get_processed_entity_log,
    get_processed_stats,
    read_jsonl_records,
)
from src.graph import (
    Direction,
    PathQueryRequest,
    QueryResult,
    RelationshipQueryRequest,
    SubgraphQueryRequest,
    TraverseQueryRequest,
    graph_query_engine,
)
from src.observability.metrics import metrics_collector
from src.resolution.entity_resolver import EntityResolver
from src.resolution.graph_linker import graph_linker
from src.resolution.review_queue import review_queue
from src.retrieval import (
    CitationValidationRequest,
    CitationValidationResponse,
    HybridRetrievalRequest,
    HybridRetrievalResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    citation_validator,
    hybrid_retrieval_service,
    rag_service,
)
from src.vector.vector_store import vector_store

dashboard_resolver = EntityResolver()

configure_logging(settings.LOG_LEVEL)
logger = structlog.get_logger(__name__)

MOCK_DATA_DIR = settings.BASE_DIR / "src" / "dashboard" / "mock_data"
FRONTEND_DIST_DIR = settings.BASE_DIR / "src" / "dashboard" / "frontend" / "dist"
PROCESSED_DIR = settings.DATA_PROCESSED_DIR

# Global pipeline execution state
pipeline_state: Dict[str, Any] = {
    "status": "idle",  # "idle" | "running" | "completed" | "failed"
    "last_run_at": datetime.now(timezone.utc).isoformat(),
    "current_stage": None,
    "progress_pct": 0,
}


def load_mock_json(filename: str) -> Any:
    file_path = MOCK_DATA_DIR / filename
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sync_static_assets():
    """Copy logo and favicon assets to frontend public and dist directories."""
    pub_dir = settings.BASE_DIR / "src" / "dashboard" / "frontend" / "public"
    dist_dir = settings.BASE_DIR / "src" / "dashboard" / "frontend" / "dist"
    pub_dir.mkdir(parents=True, exist_ok=True)

    logo_src = settings.BASE_DIR / "logo.png"
    fav_src = settings.BASE_DIR / "favicon.png"

    if logo_src.exists():
        shutil.copy(logo_src, pub_dir / "logo.png")
        if dist_dir.exists():
            shutil.copy(logo_src, dist_dir / "logo.png")

    if fav_src.exists():
        shutil.copy(fav_src, pub_dir / "favicon.png")
        shutil.copy(fav_src, pub_dir / "favicon.ico")
        if dist_dir.exists():
            shutil.copy(fav_src, dist_dir / "favicon.png")
            shutil.copy(fav_src, dist_dir / "favicon.ico")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.setup_directories()
    try:
        sync_static_assets()
    except Exception as e:
        logger.warning("Failed executing asset sync", error=str(e))
    logger.info("Graphone Pipeline Dashboard backend started", mock_mode=settings.MOCK_MODE)
    yield


app = FastAPI(
    title="Tripwire Pipeline Dashboard API",
    description="FastAPI Backend and Management Interface for Async Ingestion System",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """Get runtime pipeline settings and mock_mode status."""
    return {"mockMode": settings.MOCK_MODE, "mock_mode": settings.MOCK_MODE}


@app.get("/logo.png")
async def get_logo_image():
    """Serve brand logo image asset."""
    logo_path = settings.BASE_DIR / "logo.png"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/png")
    frontend_logo = settings.BASE_DIR / "src" / "dashboard" / "frontend" / "public" / "logo.png"
    if frontend_logo.exists():
        return FileResponse(frontend_logo, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo file not found")


@app.get("/favicon.png")
@app.get("/favicon.ico")
async def get_favicon_image():
    """Serve site favicon image asset."""
    fav_path = settings.BASE_DIR / "favicon.png"
    if fav_path.exists():
        return FileResponse(fav_path, media_type="image/png")
    frontend_fav = settings.BASE_DIR / "src" / "dashboard" / "frontend" / "public" / "favicon.png"
    if frontend_fav.exists():
        return FileResponse(frontend_fav, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon file not found")


@app.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics() -> str:
    """Prometheus exposition metrics endpoint for operational system monitoring."""
    stats = get_processed_stats()
    if isinstance(stats, dict):
        if stats.get("entities"):
            for etype, edata in stats["entities"].items():
                metrics_collector.record_ingested_entity(etype, edata.get("count", 0))
        if stats.get("llm") and isinstance(stats["llm"], dict) and stats["llm"].get("tiers"):
            for tier in stats["llm"]["tiers"]:
                provider = tier.get("provider", "Heuristic")
                metrics_collector.record_llm_call(
                    provider,
                    latency_seconds=tier.get("avgLatencyMs", 0) / 1000.0,
                    token_count=tier.get("tokenCount", 0),
                )
    return metrics_collector.generate_prometheus_format()


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Get aggregated statistics across entity types, 7-run sparklines, and LLM tier breakdown."""
    if settings.MOCK_MODE:
        stats_data = load_mock_json("stats.json")
        llm_data = load_mock_json("llm_stats.json")
        if isinstance(stats_data, dict):
            stats_data["status"] = pipeline_state["status"]
            if pipeline_state["status"] == "running":
                stats_data["currentStage"] = pipeline_state["current_stage"]
                stats_data["progressPct"] = pipeline_state["progress_pct"]
            stats_data["llm"] = llm_data
            stats_data["mockMode"] = True
            stats_data["mock_mode"] = True
            stats_data["incremental"] = metrics_collector.get_incremental_stats()
            stats_data["rag"] = metrics_collector.get_rag_metrics()
            stats_data["hybrid"] = metrics_collector.get_hybrid_metrics()
            return stats_data

    # Real data processing pathway
    stats_data = get_processed_stats()
    stats_data["mockMode"] = settings.MOCK_MODE
    stats_data["mock_mode"] = settings.MOCK_MODE
    stats_data["incremental"] = metrics_collector.get_incremental_stats()
    stats_data["rag"] = metrics_collector.get_rag_metrics()
    stats_data["hybrid"] = metrics_collector.get_hybrid_metrics()

    stats_data["status"] = pipeline_state["status"]
    if pipeline_state["status"] == "running":
        stats_data["currentStage"] = pipeline_state["current_stage"]
        stats_data["progressPct"] = pipeline_state["progress_pct"]

    return stats_data



@app.get("/api/records/{record_type}")
async def get_records(
    record_type: str,
    search: Optional[str] = Query(default=None, description="Client fuzzy search query"),
) -> List[Dict[str, Any]]:
    """Fetch entity records for startup, product, research_paper, job, or news."""
    type_mapping = {
        "startup": ("startups.jsonl", "startups.json"),
        "product": ("products.jsonl", "products.json"),
        "research_paper": ("research_papers.jsonl", "research_papers.json"),
        "job": ("jobs.jsonl", "jobs.json"),
        "news": ("news.jsonl", "news.json"),
    }

    if record_type not in type_mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entity record_type '{record_type}'. Must be one of {list(type_mapping.keys())}",
        )

    jsonl_file, mock_file = type_mapping[record_type]

    if settings.MOCK_MODE:
        records = load_mock_json(mock_file)
        records = records if isinstance(records, list) else []
    else:
        records = read_jsonl_records(jsonl_file)

    if search:
        query = search.lower()
        records = [
            r
            for r in records
            if any(query in str(v).lower() for v in r.values() if isinstance(v, (str, list, dict)))
        ]

    return records


@app.get("/api/entity-log")
async def get_entity_log() -> Dict[str, Any]:
    """Fetch Entity Resolution logs and method breakdown."""
    if settings.MOCK_MODE:
        mock_log = load_mock_json("entity_resolution_log.json")
        return mock_log if isinstance(mock_log, dict) else {"summary": {}, "entries": []}

    return get_processed_entity_log()


class ReviewActionPayload(BaseModel):
    """Payload for human review approve/reject actions."""

    reviewer: Optional[str] = "human_reviewer"
    notes: Optional[str] = None


@app.get("/reviews")
@app.get("/api/reviews")
@app.get("/api/resolution/review")
async def get_resolution_reviews(
    status: Optional[str] = Query(default="pending", description="Review status filter: pending, approved, rejected, all"),
) -> List[Dict[str, Any]]:
    """Fetch persistent entity resolution review queue items with status filtering."""
    st = (status or "pending").lower().strip()
    if st in ("all", "*"):
        reviews = review_queue.list_all_reviews()
    elif st in ("pending", "review", "needs_review"):
        reviews = review_queue.get_pending_reviews()
    elif st in ("approved", "approve"):
        reviews = review_queue.get_reviews_by_status("APPROVED")
    elif st in ("rejected", "reject"):
        reviews = review_queue.get_reviews_by_status("REJECTED")
    else:
        reviews = review_queue.get_reviews_by_status(status.upper())

    if not reviews and settings.MOCK_MODE and st in ("pending", "review", "needs_review", "all"):
        return [
            {
                "id": "rev-00101",
                "candidate": "Open AI Inc",
                "canonical": "OpenAI",
                "matching_method": "fuzzy",
                "fuzzy_score": 0.86,
                "confidence": 0.82,
                "decision": "REVIEW",
                "status": "PENDING",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "TechCrunch RSS",
                "match_reason": "Ambiguous fuzzy similarity (86.0%) in review range [65.0%, 85.0%)",
            },
            {
                "id": "rev-00102",
                "candidate": "Anthropic AI Labs",
                "canonical": "Anthropic",
                "matching_method": "fuzzy",
                "fuzzy_score": 0.78,
                "confidence": 0.75,
                "decision": "REVIEW",
                "status": "PENDING",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "ProductHunt Feed",
                "match_reason": "Ambiguous fuzzy similarity (78.0%) in review range [65.0%, 85.0%)",
            },
        ]
    return reviews


@app.get("/reviews/stats")
@app.get("/api/reviews/stats")
@app.get("/api/resolution/review/stats")
async def get_resolution_review_stats() -> Dict[str, Any]:
    """Get aggregated review queue metrics (total, pending, approved, rejected)."""
    return review_queue.get_review_stats()


@app.get("/reviews/{review_id}")
@app.get("/api/reviews/{review_id}")
@app.get("/api/resolution/review/{review_id}")
async def get_resolution_review_by_id(review_id: str) -> Dict[str, Any]:
    """Fetch a single review item by ID, including its full audit trail."""
    review = review_queue.get_review_by_id(review_id)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review item '{review_id}' not found")
    return review


@app.get("/reviews/{review_id}/audit")
@app.get("/api/reviews/{review_id}/audit")
async def get_resolution_review_audit(review_id: str) -> List[Dict[str, Any]]:
    """Fetch immutable audit trail history for a specific review item."""
    review = review_queue.get_review_by_id(review_id)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review item '{review_id}' not found")
    return review_queue.get_review_audit_trail(review_id)


@app.post("/reviews/{review_id}/approve")
@app.post("/api/reviews/{review_id}/approve")
@app.post("/api/resolution/review/{review_id}/approve")
async def approve_resolution_review(
    review_id: str,
    payload: Optional[ReviewActionPayload] = None,
) -> Dict[str, Any]:
    """
    Approve ambiguous entity resolution match, merging candidate into canonical entity.
    Fails safely with 409 Conflict if review was already rejected or approved.
    """
    reviewer = payload.reviewer if payload else "human_reviewer"
    notes = payload.notes if payload else None

    try:
        updated = review_queue.approve_review(review_id, notes=notes, reviewer=reviewer)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if not updated:
        if settings.MOCK_MODE and review_id in ("rev-00101", "rev-00102"):
            return {
                "status": "approved",
                "review": {
                    "id": review_id,
                    "candidate": "Open AI Inc",
                    "canonical": "OpenAI",
                    "decision": "APPROVED",
                    "status": "APPROVED",
                },
            }
        raise HTTPException(status_code=404, detail=f"Review item '{review_id}' not found")

    dashboard_resolver.add_approved_alias(updated["candidate"], updated["canonical"])
    return {"status": "approved", "review": updated}


@app.post("/reviews/{review_id}/reject")
@app.post("/api/reviews/{review_id}/reject")
@app.post("/api/resolution/review/{review_id}/reject")
async def reject_resolution_review(
    review_id: str,
    payload: Optional[ReviewActionPayload] = None,
) -> Dict[str, Any]:
    """
    Reject ambiguous entity resolution match, keeping candidate as a distinct entity.
    Fails safely with 409 Conflict if review was already approved or rejected.
    """
    reviewer = payload.reviewer if payload else "human_reviewer"
    notes = payload.notes if payload else None

    try:
        updated = review_queue.reject_review(review_id, notes=notes, reviewer=reviewer)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if not updated:
        if settings.MOCK_MODE and review_id in ("rev-00101", "rev-00102"):
            return {
                "status": "rejected",
                "review": {
                    "id": review_id,
                    "candidate": "Open AI Inc",
                    "canonical": "OpenAI",
                    "decision": "REJECTED",
                    "status": "REJECTED",
                },
            }
        raise HTTPException(status_code=404, detail=f"Review item '{review_id}' not found")

    return {"status": "rejected", "review": updated}


def get_latest_actual_benchmark_report() -> Optional[Dict[str, Any]]:
    """Locate and return the contents of the latest actual benchmark JSON report in evaluation/reports/."""
    reports_dir = settings.BASE_DIR / "evaluation" / "reports"
    if not reports_dir.exists():
        return None

    json_files = [
        f for f in reports_dir.glob("*.json")
        if f.is_file() and f.stat().st_size > 0 and not f.name.startswith(".")
    ]
    if not json_files:
        return None

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed reading latest benchmark report file", file=str(latest_file), error=str(e))
        return None


@app.get("/api/benchmark/latest")
async def get_latest_benchmark_report_endpoint() -> Dict[str, Any]:
    """Fetch the latest actual Phase 1 benchmark JSON report."""
    report = get_latest_actual_benchmark_report()
    if not report:
        raise HTTPException(
            status_code=404,
            detail="No benchmark report available. Run the evaluation benchmark to generate a report.",
        )
    return report


@app.get("/api/benchmark")
async def get_benchmark_report() -> Dict[str, Any]:
    """Fetch reproducible scientific benchmark performance metrics."""
    report = get_latest_actual_benchmark_report()
    if report:
        return report

    # Dynamically aggregate metrics from live data/processed/*.jsonl log files
    return get_live_benchmark_metrics()


def get_latest_incremental_benchmark_report() -> Optional[Dict[str, Any]]:
    """Locate and return contents of incremental benchmark report in evaluation/reports/."""
    inc_file = settings.BASE_DIR / "evaluation" / "reports" / "incremental_benchmark.json"
    if inc_file.exists() and inc_file.stat().st_size > 0:
        try:
            with open(inc_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed reading incremental benchmark report file", file=str(inc_file), error=str(e))
    return None


@app.get("/api/benchmark/incremental")
async def get_incremental_benchmark_report_endpoint() -> Dict[str, Any]:
    """Fetch the latest Phase 2 Incremental Processing benchmark JSON report."""
    report = get_latest_incremental_benchmark_report()
    if report:
        return report

    stats = metrics_collector.get_incremental_stats()
    return {
        "metadata": {
            "mode": "live",
            "benchmark_name": "live_incremental_metrics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "discovered": stats.get("incremental_total", 0),
            "new": stats.get("incremental_new", 0),
            "changed": stats.get("incremental_changed", 0),
            "unchanged": stats.get("incremental_unchanged", 0),
            "skipped": stats.get("incremental_skipped", 0),
            "llm_calls_saved": stats.get("llm_calls_saved", 0),
            "embedding_calls_saved": stats.get("embedding_calls_saved", 0),
            "work_avoided_pct": stats.get("work_avoided_pct", 0.0),
        },
    }


@app.get("/api/graph")
async def get_knowledge_graph() -> Dict[str, Any]:
    """Fetch Knowledge Graph nodes and relational edges."""
    nodes, edges = graph_linker.build_graph_triples()
    return {"nodes": nodes, "edges": edges, "summary": {"nodeCount": len(nodes), "edgeCount": len(edges)}}


@app.get("/api/graph/export", response_class=PlainTextResponse)
async def export_cypher_graph() -> str:
    """Export Neo4j Cypher import script for Knowledge Graph visualization."""
    return graph_linker.generate_cypher_import_script()


@app.get("/api/graph/entity/{entity_id}")
async def get_graph_entity(entity_id: str) -> Dict[str, Any]:
    """Look up a single knowledge graph entity by ID or name."""
    node = graph_linker.get_node(entity_id)
    if not node:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_id}' not found in knowledge graph",
        )
    return node


@app.get("/api/graph/neighbors/{entity_id}")
async def get_graph_neighbors(
    entity_id: str,
    relation: Optional[str] = Query(default=None, description="Filter neighbors by relation type (e.g., PRODUCES, POSTED_BY)"),
    direction: str = Query(default="both", description="Direction of relations: both, outgoing, incoming"),
) -> Dict[str, Any]:
    """Retrieve direct (1-hop) neighbors of an entity in the knowledge graph."""
    res = graph_linker.get_neighbors(entity_id, relation=relation, direction=direction)
    if not res:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_id}' not found in knowledge graph",
        )
    return res


@app.get("/api/graph/path")
async def get_graph_path(
    from_entity: str = Query(..., alias="from", description="Source entity ID or name"),
    to_entity: str = Query(..., alias="to", description="Target entity ID or name"),
    max_depth: int = Query(default=5, ge=1, le=10, description="Maximum path search depth (hops)"),
    relation: Optional[str] = Query(default=None, description="Filter relations along path"),
) -> Dict[str, Any]:
    """Discover the shortest path between two entities in the knowledge graph."""
    res = graph_linker.find_path(from_entity, to_entity, max_depth=max_depth, relation=relation)
    if not res:
        raise HTTPException(
            status_code=404,
            detail="One or both entities not found in knowledge graph",
        )
    return res


@app.get("/api/graph/traverse/{entity_id}")
async def traverse_graph_subgraph(
    entity_id: str,
    hops: int = Query(default=2, ge=1, le=10, description="Traversal expansion depth (hops)"),
    relation: Optional[str] = Query(default=None, description="Filter relations traversed"),
) -> Dict[str, Any]:
    """Bounded multi-hop traversal expanding subgraph from an entity root with max-hop guard."""
    res = graph_linker.traverse(entity_id, hops=hops, relation=relation)
    if not res:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_id}' not found in knowledge graph",
        )
    return res


# ==============================================================================
# PHASE 4: GRAPH QUERY ENGINE REST API ENDPOINTS
# ==============================================================================


@app.get("/graph/nodes/{node_id}")
@app.get("/api/graph/nodes/{node_id}")
async def get_graph_node_endpoint(node_id: str) -> Dict[str, Any]:
    """Retrieve a single graph node by ID or name with 404 validation."""
    node = graph_query_engine.get_node(node_id)
    if not node:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{node_id}' not found in knowledge graph",
        )
    return node.to_dict()


@app.get("/graph/nodes/{node_id}/neighbors")
@app.get("/api/graph/nodes/{node_id}/neighbors")
async def get_graph_node_neighbors_endpoint(
    node_id: str,
    relation: Optional[str] = Query(default=None, description="Filter neighbors by relation type"),
    node_label: Optional[str] = Query(default=None, description="Filter neighbors by node label"),
    direction: Direction = Query(default=Direction.BOTH, description="Direction: outgoing, incoming, both"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum neighbors to return"),
) -> QueryResult:
    """Retrieve direct 1-hop neighbors of a node with filtering by relation, label, direction, and limit."""
    node = graph_query_engine.get_node(node_id)
    if not node:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{node_id}' not found in knowledge graph",
        )
    return graph_query_engine.get_neighbors(
        node_id_or_name=node_id,
        relation=relation,
        node_label=node_label,
        direction=direction,
        limit=limit,
    )


@app.post("/graph/query/traverse")
@app.post("/api/graph/query/traverse")
async def traverse_graph_endpoint(payload: TraverseQueryRequest) -> QueryResult:
    """Execute bounded multi-hop graph traversal starting from root_id."""
    node = graph_query_engine.get_node(payload.root_id)
    if not node:
        raise HTTPException(
            status_code=404,
            detail=f"Root node '{payload.root_id}' not found in knowledge graph",
        )
    return graph_query_engine.traverse(
        root_id_or_name=payload.root_id,
        max_depth=payload.max_depth,
        relation=payload.relation,
        node_label=payload.node_label,
        direction=payload.direction,
        properties=payload.properties,
        limit=payload.limit,
    )


@app.post("/graph/query/path")
@app.post("/api/graph/query/path")
async def find_graph_path_endpoint(payload: PathQueryRequest) -> QueryResult:
    """Discover shortest path between two nodes with cycle safety and depth bounding."""
    start_node = graph_query_engine.get_node(payload.from_node)
    end_node = graph_query_engine.get_node(payload.to_node)
    missing = []
    if not start_node:
        missing.append(f"from_node '{payload.from_node}'")
    if not end_node:
        missing.append(f"to_node '{payload.to_node}'")
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Path discovery node lookup failed: {', '.join(missing)} not found in knowledge graph",
        )
    return graph_query_engine.find_path(
        from_node_query=payload.from_node,
        to_node_query=payload.to_node,
        max_depth=payload.max_depth,
        relation=payload.relation,
        direction=payload.direction,
    )


@app.post("/graph/query/subgraph")
@app.post("/api/graph/query/subgraph")
async def extract_subgraph_endpoint(payload: SubgraphQueryRequest) -> QueryResult:
    """Extract induced neighborhood subgraph around seed nodes."""
    return graph_query_engine.extract_subgraph(
        seed_node_queries=payload.node_ids,
        hops=payload.hops,
        relation=payload.relation,
        node_label=payload.node_label,
        direction=payload.direction,
        limit=payload.limit,
    )


@app.post("/graph/query/relationships")
@app.post("/api/graph/query/relationships")
async def query_relationships_endpoint(payload: RelationshipQueryRequest) -> QueryResult:
    """Query global relationships matching triple pattern constraints."""
    return graph_query_engine.find_relationships(
        relation=payload.relation,
        source_label=payload.source_label,
        target_label=payload.target_label,
        min_confidence=payload.min_confidence,
        limit=payload.limit,
    )


@app.get("/graph/stats")
@app.get("/api/graph/stats")
async def get_graph_stats_endpoint() -> Dict[str, Any]:
    """Retrieve graph topological stats and query execution performance metrics."""
    nodes, edges = graph_linker.build_graph_triples()
    metrics = metrics_collector.get_graph_metrics()
    return {
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "metrics": metrics,
    }


@app.get("/api/search")
async def search_vector_store(
    q: str = Query(..., description="Natural language semantic search query"),
    type: Optional[str] = Query(default="all", description="Entity filter type: all, startup, product, research_paper, job, news"),
    limit: int = Query(default=10, ge=1, le=50),
) -> Dict[str, Any]:
    """Perform hybrid vector semantic search across all ingested entities."""
    matches = vector_store.search(query=q, record_type=type, limit=limit)
    return {"query": q, "filterType": type, "count": len(matches), "results": matches}


@app.post("/api/search/reindex")
async def reindex_vector_store() -> Dict[str, Any]:
    """Reindex all JSONL entity records into vector storage."""
    return vector_store.index_all_records()


@app.post("/retrieval/hybrid", response_model=HybridRetrievalResponse)
@app.post("/api/retrieval/hybrid", response_model=HybridRetrievalResponse)
async def query_hybrid_retrieval(req: HybridRetrievalRequest) -> HybridRetrievalResponse:
    """
    Genuine Graph + Vector Hybrid Retrieval Endpoint.
    Combines dense semantic vector retrieval with Knowledge Graph multi-hop expansion,
    deduplicates candidates by stable identifiers, and ranks via weighted fusion or RRF.
    """
    try:
        return hybrid_retrieval_service.search_hybrid(req)
    except Exception as e:
        logger.error("Hybrid retrieval execution failed", error=str(e), query=req.query)
        metrics_collector.hybrid_query_errors += 1
        raise HTTPException(status_code=500, detail=f"Hybrid retrieval failed: {str(e)}")


@app.post("/rag/query", response_model=RAGQueryResponse)
@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def query_rag_endpoint(req: RAGQueryRequest) -> RAGQueryResponse:
    """
    Citation-Aware Grounded RAG Endpoint.
    Retrieves evidence chunks via hybrid vector + graph traversal, produces factually
    grounded answers with verifiable [1], [2] citations, and runs strict citation validation.
    """
    try:
        return rag_service.answer_query(req)
    except Exception as e:
        logger.error("RAG query execution failed", error=str(e), query=req.query)
        metrics_collector.record_rag_query(
            latency_seconds=0.0,
            citations_count=0,
            invalid_citations_count=0,
            unsupported_claims_count=0,
            is_abstained=False,
            validation_status="VALIDATION_ERROR",
        )
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")


@app.post("/rag/validate", response_model=CitationValidationResponse)
@app.post("/api/rag/validate", response_model=CitationValidationResponse)
async def validate_citations_endpoint(req: CitationValidationRequest) -> CitationValidationResponse:
    """
    Standalone Citation and Claim Support Validation Endpoint.
    Validates citation reference validity, checks non-fabrication of sources,
    and runs semantic claim entailment checking.
    """
    t0 = time.perf_counter()
    try:
        val_res = citation_validator.validate(
            answer=req.answer,
            sources=req.sources,
            citations=req.citations,
            semantic_validation=req.semantic_validation,
            validator_type=req.validator_type,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        metrics_collector.record_citation_validation(
            is_valid=val_res.reference_valid,
            is_semantic_supported=val_res.semantic_valid,
            is_error=False,
            latency_seconds=duration_ms / 1000.0,
        )
        return CitationValidationResponse(
            validation_status=val_res.status,
            validation_result=val_res,
            citations=req.citations or [],
            execution_ms=round(duration_ms, 3),
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        logger.error("Citation validation failed with error", error=str(e))
        metrics_collector.record_citation_validation(
            is_valid=False,
            is_semantic_supported=False,
            is_error=True,
            latency_seconds=duration_ms / 1000.0,
        )
        raise HTTPException(status_code=500, detail=f"Citation validation failed: {str(e)}")




@app.get("/api/logs")
async def get_logs(
    source: str = Query(default="scrape.log", description="Log file source: scrape.log, llm_extraction.log, entity_resolution.log")
) -> List[Dict[str, Any]]:
    """Tail recent pipeline log entries color-coded by log level from live log files."""
    if source in ("llm_extraction.log", "llm_calls_log.jsonl"):
        llm_records = read_jsonl_records("llm_calls_log.jsonl")
        if llm_records:
            formatted_logs = []
            for r in llm_records[-50:]:
                succ = r.get("success", True)
                formatted_logs.append({
                    "timestamp": r.get("timestamp"),
                    "level": "INFO" if succ else "WARN",
                    "module": "src.llm.fallback_chain",
                    "message": f"[{r.get('provider')}] schema '{r.get('schema')}' {'succeeded' if succ else 'failed'} ({r.get('latency_ms')}ms, {r.get('token_count')} tokens)" + (f" - Error: {r.get('error')}" if r.get('error') else ""),
                })
            return formatted_logs

    if source in ("entity_resolution.log", "entity_mapping_log.jsonl"):
        er_log = get_processed_entity_log()
        entries = er_log.get("entries", [])
        if entries:
            formatted_logs = []
            for e in entries[-50:]:
                formatted_logs.append({
                    "timestamp": e.get("timestamp"),
                    "level": "INFO" if e.get("status") == "merged" else "DEBUG",
                    "module": "src.resolution.entity_resolver",
                    "message": f"[EntityResolver] Merged entity '{e.get('raw_name')}' -> '{e.get('canonical_name')}' (method: {e.get('method_used')}, confidence: {e.get('confidence_score')})",
                })
            return formatted_logs

    if source == "scrape.log":
        all_records = get_all_processed_records()
        scrape_events = []
        for r_type, records in all_records.items():
            if r_type == "entity_log":
                continue
            for r in records[-15:]:
                name_val = r.get("name") or r.get("title") or r.get("canonical_name") or "Extracted Payload"
                src_name = (r.get("source") or {}).get("name") if isinstance(r.get("source"), dict) else "AsyncScraper"
                scrape_events.append({
                    "timestamp": r.get("created_at") or r.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    "level": "INFO",
                    "module": "src.ingestion.async_scraper",
                    "message": f"[{src_name}] Ingested {r_type[:-1] if r_type.endswith('s') else r_type} item: '{name_val}'",
                })
        if scrape_events:
            scrape_events.sort(key=lambda x: str(x.get("timestamp", "")))
            return scrape_events[-50:]

    logs_data = load_mock_json("logs.json")
    if isinstance(logs_data, dict):
        return logs_data.get(source, [])
    return []


async def execute_pipeline_background():
    """Background task executing the actual pipeline orchestrator asynchronously."""
    global pipeline_state
    try:
        pipeline_state["status"] = "running"
        pipeline_state["progress_pct"] = 15
        pipeline_state["current_stage"] = "Async Scraping & Source Crawling"
        await asyncio.sleep(1.5)

        pipeline_state["progress_pct"] = 45
        pipeline_state["current_stage"] = "LLM Multi-Tier Structured Extraction"

        from src.main import PipelineOrchestrator
        orchestrator = PipelineOrchestrator(dry_run=True, limit=5)
        await orchestrator.run_all_pipelines()

        pipeline_state["progress_pct"] = 80
        pipeline_state["current_stage"] = "Entity Resolution & Schema Persistence"
        await asyncio.sleep(1.0)

        pipeline_state["progress_pct"] = 100
        pipeline_state["current_stage"] = "Completed"
        pipeline_state["status"] = "idle"
        pipeline_state["last_run_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        logger.error("Background pipeline execution failed", error=str(e))
        pipeline_state["status"] = "failed"
        pipeline_state["current_stage"] = "Failed"


@app.post("/api/run")
async def trigger_pipeline_run(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger pipeline execution background task from Dashboard UI button."""
    if pipeline_state["status"] == "running":
        return {"status": "already_running", "message": "Pipeline run is already in progress"}

    pipeline_state["status"] = "running"
    pipeline_state["progress_pct"] = 10
    pipeline_state["current_stage"] = "Initializing Pipeline Run"

    background_tasks.add_task(execute_pipeline_background)
    return {
        "status": "started",
        "message": "Pipeline run initiated successfully",
        "timestamp": pipeline_state["last_run_at"],
    }


# Mount built frontend static files if dist assets exist
assets_dir = FRONTEND_DIST_DIR / "assets"
if assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    file_path = FRONTEND_DIST_DIR / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend dist index.html not found")
