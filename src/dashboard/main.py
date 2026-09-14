import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from config.settings import configure_logging, settings
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.dashboard.processed_reader import (
    get_all_processed_records,
    get_live_benchmark_metrics,
    get_processed_entity_log,
    get_processed_stats,
    read_jsonl_records,
)
from src.observability.metrics import metrics_collector
from src.resolution.graph_linker import graph_linker
from src.vector.vector_store import vector_store

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
            return stats_data

    # Real data processing pathway
    stats_data = get_processed_stats()
    stats_data["mockMode"] = settings.MOCK_MODE
    stats_data["mock_mode"] = settings.MOCK_MODE

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


@app.get("/api/graph")
async def get_knowledge_graph() -> Dict[str, Any]:
    """Fetch Knowledge Graph nodes and relational edges."""
    nodes, edges = graph_linker.build_graph_triples()
    return {"nodes": nodes, "edges": edges, "summary": {"nodeCount": len(nodes), "edgeCount": len(edges)}}


@app.get("/api/graph/export", response_class=PlainTextResponse)
async def export_cypher_graph() -> str:
    """Export Neo4j Cypher import script for Knowledge Graph visualization."""
    return graph_linker.generate_cypher_import_script()


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
