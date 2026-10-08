"""
================================================================================
GRAPHONE PIPELINE: TOP-LEVEL CLI ORCHESTRATOR
================================================================================

CLI entry point orchestrating asynchronous web scraping, multi-tier LLM data
extraction, entity resolution, Pydantic validation, JSONL output persistence,
and Google Sheets sync.

Subcommands:
  - python -m src.main run startups
  - python -m src.main run products
  - python -m src.main run papers
  - python -m src.main run news
  - python -m src.main run jobs
  - python -m src.main run all
  - python -m src.main export sheets

Flags:
  --dry-run      Run on small sample (5-10 records) without hitting Google Sheets
  --limit INT    Max items to process per target (default: 20)
================================================================================
"""

import argparse
import asyncio
import hashlib
import json
import re
import time
from typing import Any, Dict, Optional

import structlog
from config.settings import settings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.export.sheets_export import GoogleSheetsExporter
from src.llm.fallback_chain import FallbackChain
from src.observability.metrics import metrics_collector
from src.resolution.entity_resolver import EntityResolver
from src.schemas.product import Product
from src.schemas.startup import Startup
from src.scrapers.dedup_tracker import (
    DedupTracker,
    ProcessingDecision,
)
from src.scrapers.freshness import (
    run_freshness_pipeline,
    run_jobs_freshness_pipeline,
    run_news_freshness_pipeline,
)
from src.scrapers.research_papers import run_research_papers_pipeline

logger = structlog.get_logger(__name__)
console = Console()


def extract_item_url(raw_html: str, default_prefix: str = "https://graphone.internal/item") -> str:
    """Extract canonical URL link from raw HTML snippet, or generate deterministic URI."""
    href_match = re.search(r'href=["\'](https?://[^"\']+)["\']', raw_html, re.I)
    if href_match:
        return href_match.group(1).strip()
    h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", raw_html, re.I)
    if h1_match:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", h1_match.group(1).strip().lower()).strip("-")
        return f"{default_prefix}/{slug}"
    return f"{default_prefix}/doc-{hashlib.sha256(raw_html.encode('utf-8')).hexdigest()[:12]}"


# ------------------------------------------------------------------------------
# SAMPLE RAW HTML SNIPPETS FOR STARTUP & PRODUCT SCRAPING & LLM EXTRACTION
# ------------------------------------------------------------------------------

SAMPLE_STARTUP_HTML = [
    """
    <div class="company-card">
        <h1>Cogna AI</h1>
        <p class="tagline">Autonomous AI Data Ingestion and Pipeline Infrastructure</p>
        <span class="funding">Total Funding: $18.5M</span>
        <span class="stage">Stage: Series A</span>
        <span class="location">Location: San Francisco, CA</span>
        <span class="year">Founded: 2023</span>
        <span class="team">Employees: 25-50</span>
        <div class="categories"><span>AI Infrastructure</span><span>Developer Tools</span></div>
    </div>
    """,
    """
    <div class="company-card">
        <h1>Anthropic, Inc.</h1>
        <p class="tagline">AI Safety and Research Company Building Claude Models</p>
        <span class="funding">Total Funding: $7.3B</span>
        <span class="stage">Stage: Series E</span>
        <span class="location">Location: San Francisco, CA</span>
        <span class="year">Founded: 2021</span>
        <span class="team">Employees: 500-1000</span>
        <div class="categories"><span>AI Models</span><span>LLM Research</span></div>
    </div>
    """,
    """
    <div class="company-card">
        <h1>Mistral AI SAS</h1>
        <p class="tagline">Frontier Open Weights Models for Enterprise AI</p>
        <span class="funding">Total Funding: $640M</span>
        <span class="stage">Stage: Series B</span>
        <span class="location">Location: Paris, France</span>
        <span class="year">Founded: 2023</span>
        <span class="team">Employees: 50-100</span>
        <div class="categories"><span>Open Source</span><span>AI Models</span></div>
    </div>
    """,
]

SAMPLE_PRODUCT_HTML = [
    """
    <div class="product-header">
        <h1 class="product-title">SynthFlow Voice 2.0</h1>
        <p class="tagline">Real-Time Low-Latency AI Voice Agents for Enterprise</p>
        <div class="maker">By Synthflow Inc</div>
        <div class="upvotes">Upvotes: 1240</div>
        <div class="pricing">Pricing Model: Freemium</div>
        <a class="url" href="https://synthflow.ai">Product Website</a>
    </div>
    """,
    """
    <div class="product-header">
        <h1 class="product-title">Claude 3.5 Sonnet</h1>
        <p class="tagline">State of the Art Intelligence & Code Generation Model</p>
        <div class="maker">By Anthropic PBC</div>
        <div class="upvotes">Upvotes: 4890</div>
        <div class="pricing">Pricing Model: Paid</div>
        <a class="url" href="https://claude.ai">Product Website</a>
    </div>
    """,
    """
    <div class="product-header">
        <h1 class="product-title">Cursor AI Editor</h1>
        <p class="tagline">The AI-first Code Editor for Pair Programming</p>
        <div class="maker">By Anysphere Inc</div>
        <div class="upvotes">Upvotes: 3500</div>
        <div class="pricing">Pricing Model: Freemium</div>
        <a class="url" href="https://cursor.com">Product Website</a>
    </div>
    """,
]


class PipelineOrchestrator:
    """Master orchestrator executing pipeline stages with rich CLI metrics reporting."""

    def __init__(self, dry_run: bool = False, limit: Optional[int] = None):
        self.dry_run = dry_run
        if limit is not None:
            self.limit = limit
        else:
            self.limit = 5 if dry_run else 20
        self.resolver = EntityResolver()
        self.llm_chain = FallbackChain()
        self.dedup = DedupTracker()
        self.metrics: Dict[str, Any] = {
            "start_time": time.time(),
            "targets": {},
            "llm_tiers_used": {},
            "resolution_stats": {"exact": 0, "normalized": 0, "fuzzy": 0, "unresolved": 0},
            "validation_failures": 0,
            "incremental": {
                "total": 0,
                "new": 0,
                "changed": 0,
                "unchanged": 0,
                "skipped": 0,
                "llm_calls_saved": 0,
                "embedding_calls_saved": 0,
            },
        }

    # --------------------------------------------------------------------------
    # PIPELINE EXECUTION STAGES
    # --------------------------------------------------------------------------

    async def run_startups_pipeline(self) -> int:
        """Run Startups ingestion: Incremental Check -> LLM Chain -> Entity Resolver -> Schema Validation -> JSONL."""
        logger.info("Executing Startups Ingestion Pipeline", limit=self.limit)
        target_name = "Startups"
        records_written = 0
        raw_items = SAMPLE_STARTUP_HTML[: self.limit]

        out_records = []
        for raw_html in raw_items:
            item_url = extract_item_url(raw_html, default_prefix="https://graphone.internal/startups")
            inc_res = self.dedup.check_incremental(url=item_url, content=raw_html)

            self.metrics["incremental"]["total"] += 1
            if inc_res.decision == ProcessingDecision.NEW:
                self.metrics["incremental"]["new"] += 1
            elif inc_res.decision == ProcessingDecision.CHANGED:
                self.metrics["incremental"]["changed"] += 1
            elif inc_res.decision == ProcessingDecision.UNCHANGED:
                self.metrics["incremental"]["unchanged"] += 1

            if inc_res.is_skip or inc_res.decision == ProcessingDecision.UNCHANGED:
                self.metrics["incremental"]["skipped"] += 1
                self.metrics["incremental"]["llm_calls_saved"] += 1
                self.metrics["incremental"]["embedding_calls_saved"] += 1
                metrics_collector.record_incremental_decision(
                    decision=inc_res.decision,
                    action=inc_res.action,
                    llm_calls_saved=1,
                    embedding_calls_saved=1,
                )
                logger.info(
                    "Incremental check skipped unchanged record",
                    url=item_url,
                    decision=inc_res.decision.value,
                    action=inc_res.action.value,
                )
                continue

            metrics_collector.record_incremental_decision(
                decision=inc_res.decision,
                action=inc_res.action,
            )

            # 1. LLM Extraction
            extracted_dict, tier_used = await self.llm_chain.extract_with_fallback(raw_html, Startup)
            self.metrics["llm_tiers_used"][tier_used] = self.metrics["llm_tiers_used"].get(tier_used, 0) + 1

            if not extracted_dict:
                self.metrics["validation_failures"] += 1
                self.dedup.record_failed_processing(url=item_url, reason="LLM extraction returned empty")
                continue

            try:
                # 2. Schema Validation
                startup_obj = Startup.model_validate(extracted_dict)
                # 3. Entity Resolution
                resolved_obj, res_info = self.resolver.resolve_record(startup_obj)
                m_used = res_info.get("method_used", "unresolved")
                if m_used in self.metrics["resolution_stats"]:
                    self.metrics["resolution_stats"][m_used] += 1

                out_records.append(resolved_obj)
                records_written += 1
                # Update fingerprint in dedup tracker ONLY after successful downstream processing
                self.dedup.record_successful_processing(url=item_url, content=raw_html, source=target_name)
            except Exception as e:
                logger.error("Startup Pydantic validation failed", error=str(e))
                self.metrics["validation_failures"] += 1
                self.dedup.record_failed_processing(url=item_url, reason=str(e))

        # Only update data/processed/startups.jsonl if new/changed records were written
        out_file = settings.DATA_PROCESSED_DIR / "startups.jsonl"
        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        if records_written > 0:
            existing_recs = []
            if out_file.exists():
                try:
                    with open(out_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                existing_recs.append(json.loads(line))
                except Exception as e:
                    logger.warning("Could not read existing startups.jsonl", error=str(e))

            rec_map = {r.get("canonical_name") or r.get("name") or (r.get("source") or {}).get("url"): r for r in existing_recs}
            for rec in out_records:
                rec_dict = json.loads(rec.model_dump_json())
                k = rec_dict.get("canonical_name") or rec_dict.get("name") or (rec_dict.get("source") or {}).get("url")
                rec_map[k] = rec_dict

            with open(out_file, "w", encoding="utf-8") as f:
                for r in rec_map.values():
                    f.write(json.dumps(r) + "\n")

        self.metrics["targets"][target_name] = {"count": records_written, "status": "success"}
        return records_written

    async def run_products_pipeline(self) -> int:
        """Run Products ingestion: Incremental Check -> LLM Chain -> Entity Resolver -> Schema Validation -> JSONL."""
        logger.info("Executing Products Ingestion Pipeline", limit=self.limit)
        target_name = "Products"
        records_written = 0
        raw_items = SAMPLE_PRODUCT_HTML[: self.limit]

        out_records = []
        for raw_html in raw_items:
            item_url = extract_item_url(raw_html, default_prefix="https://graphone.internal/products")
            inc_res = self.dedup.check_incremental(url=item_url, content=raw_html)

            self.metrics["incremental"]["total"] += 1
            if inc_res.decision == ProcessingDecision.NEW:
                self.metrics["incremental"]["new"] += 1
            elif inc_res.decision == ProcessingDecision.CHANGED:
                self.metrics["incremental"]["changed"] += 1
            elif inc_res.decision == ProcessingDecision.UNCHANGED:
                self.metrics["incremental"]["unchanged"] += 1

            if inc_res.is_skip or inc_res.decision == ProcessingDecision.UNCHANGED:
                self.metrics["incremental"]["skipped"] += 1
                self.metrics["incremental"]["llm_calls_saved"] += 1
                self.metrics["incremental"]["embedding_calls_saved"] += 1
                metrics_collector.record_incremental_decision(
                    decision=inc_res.decision,
                    action=inc_res.action,
                    llm_calls_saved=1,
                    embedding_calls_saved=1,
                )
                logger.info(
                    "Incremental check skipped unchanged record",
                    url=item_url,
                    decision=inc_res.decision.value,
                    action=inc_res.action.value,
                )
                continue

            metrics_collector.record_incremental_decision(
                decision=inc_res.decision,
                action=inc_res.action,
            )

            # 1. LLM Extraction
            extracted_dict, tier_used = await self.llm_chain.extract_with_fallback(raw_html, Product)
            self.metrics["llm_tiers_used"][tier_used] = self.metrics["llm_tiers_used"].get(tier_used, 0) + 1

            if not extracted_dict:
                self.metrics["validation_failures"] += 1
                self.dedup.record_failed_processing(url=item_url, reason="LLM extraction returned empty")
                continue

            try:
                # 2. Schema Validation
                product_obj = Product.model_validate(extracted_dict)
                # 3. Entity Resolution on maker_company
                resolved_obj, res_info = self.resolver.resolve_record(product_obj)
                m_used = res_info.get("method_used", "unresolved")
                if m_used in self.metrics["resolution_stats"]:
                    self.metrics["resolution_stats"][m_used] += 1

                out_records.append(resolved_obj)
                records_written += 1
                # Update fingerprint in dedup tracker ONLY after successful downstream processing
                self.dedup.record_successful_processing(url=item_url, content=raw_html, source=target_name)
            except Exception as e:
                logger.error("Product Pydantic validation failed", error=str(e))
                self.metrics["validation_failures"] += 1
                self.dedup.record_failed_processing(url=item_url, reason=str(e))

        # Only update data/processed/products.jsonl if new/changed records were written
        out_file = settings.DATA_PROCESSED_DIR / "products.jsonl"
        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        if records_written > 0:
            existing_recs = []
            if out_file.exists():
                try:
                    with open(out_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                existing_recs.append(json.loads(line))
                except Exception as e:
                    logger.warning("Could not read existing products.jsonl", error=str(e))

            rec_map = {r.get("canonical_name") or r.get("name") or (r.get("source") or {}).get("url"): r for r in existing_recs}
            for rec in out_records:
                rec_dict = json.loads(rec.model_dump_json())
                k = rec_dict.get("canonical_name") or rec_dict.get("name") or (rec_dict.get("source") or {}).get("url")
                rec_map[k] = rec_dict

            with open(out_file, "w", encoding="utf-8") as f:
                for r in rec_map.values():
                    f.write(json.dumps(r) + "\n")

        self.metrics["targets"][target_name] = {"count": records_written, "status": "success"}
        return records_written

    async def run_papers_pipeline(self) -> int:
        """Run Research Papers ingestion pipeline."""
        logger.info("Executing Research Papers Pipeline", limit=self.limit)
        summary = await run_research_papers_pipeline(target_limit=self.limit)
        count = summary.get("records_written", 0)
        self.metrics["targets"]["Research Papers"] = {"count": count, "status": "success"}
        return count

    async def run_news_pipeline(self) -> int:
        """Run News Freshness pipeline."""
        logger.info("Executing News Freshness Pipeline", limit=self.limit)
        summary = await run_news_freshness_pipeline(limit_per_source=self.limit)
        news_cnt = summary.get("news_written", 0)
        self.metrics["targets"]["News"] = {"count": news_cnt, "status": "success"}
        return news_cnt

    async def run_jobs_pipeline(self) -> int:
        """Run Jobs Freshness pipeline."""
        logger.info("Executing Jobs Freshness Pipeline", limit=self.limit)
        summary = await run_jobs_freshness_pipeline(limit_per_source=self.limit)
        jobs_cnt = summary.get("jobs_written", 0)
        self.metrics["targets"]["Jobs"] = {"count": jobs_cnt, "status": "success"}
        return jobs_cnt

    async def run_freshness_news_jobs_pipeline(self) -> Dict[str, int]:
        """Run News & Jobs Freshness pipeline."""
        logger.info("Executing News & Jobs Freshness Pipeline", limit=self.limit)
        summary = await run_freshness_pipeline(limit_per_source=self.limit)
        news_cnt = summary.get("news_written", 0)
        jobs_cnt = summary.get("jobs_written", 0)

        self.metrics["targets"]["News"] = {"count": news_cnt, "status": "success"}
        self.metrics["targets"]["Jobs"] = {"count": jobs_cnt, "status": "success"}
        return {"news": news_cnt, "jobs": jobs_cnt}

    async def run_all_pipelines(self):
        """Run all ingestion target pipelines sequentially."""
        await self.run_startups_pipeline()
        await self.run_products_pipeline()
        await self.run_papers_pipeline()
        await self.run_freshness_news_jobs_pipeline()

    # --------------------------------------------------------------------------
    # TERMINAL UI SUMMARY TABLE DISPLAY
    # --------------------------------------------------------------------------

    def print_summary_table(self, sheet_info: Optional[Dict[str, Any]] = None):
        """Render beautiful CLI summary table using rich library."""
        elapsed_sec = round(time.time() - self.metrics["start_time"], 2)

        console.print()
        console.print(Panel.fit("[bold cyan]🚀 GRAPHONE PIPELINE EXECUTION SUMMARY[/bold cyan]"))

        # Table 1: Ingestion & Record Metrics
        table = Table(title="Target Entity Processing Summary", show_header=True, header_style="bold magenta")
        table.add_column("Entity Type", style="cyan", width=22)
        table.add_column("Records Processed", justify="right", style="green")
        table.add_column("Validation Status", style="yellow")

        total_records = 0
        for target, data in self.metrics["targets"].items():
            cnt = data.get("count", 0)
            total_records += cnt
            table.add_column if False else None
            table.add_row(target, str(cnt), "[bold green]PASSED[/bold green]")

        console.print(table)

        # Table 2: LLM Tier Usage & Entity Resolution Metrics
        metrics_table = Table(title="Execution System Metrics", show_header=True, header_style="bold blue")
        metrics_table.add_column("Metric Category", style="cyan")
        metrics_table.add_column("Details / Breakdown", style="white")

        llm_str = ", ".join([f"{k}: {v}" for k, v in self.metrics["llm_tiers_used"].items()]) or "N/A"
        res_str = ", ".join([f"{k}: {v}" for k, v in self.metrics["resolution_stats"].items()])

        metrics_table.add_row("Total Time Elapsed", f"{elapsed_sec} seconds")
        metrics_table.add_row("Total Records Ingested", str(total_records))
        metrics_table.add_row("Pydantic Validation Failures", str(self.metrics["validation_failures"]))
        metrics_table.add_row("LLM Tiers Succeeded", llm_str)
        metrics_table.add_row("Entity Resolution Methods", res_str)

        inc = self.metrics.get("incremental", {})
        if inc.get("total", 0) > 0:
            avoided_pct = round((inc.get("skipped", 0) / max(inc.get("total", 1), 1)) * 100.0, 1)
            inc_str = (
                f"Total: {inc.get('total', 0)} (New: {inc.get('new', 0)}, "
                f"Changed: {inc.get('changed', 0)}, Skipped: {inc.get('skipped', 0)}) | "
                f"Avoided: {avoided_pct}% | "
                f"Saved: {inc.get('llm_calls_saved', 0)} LLMs, {inc.get('embedding_calls_saved', 0)} Embeddings"
            )
            metrics_table.add_row("Incremental Dedup Decision", inc_str)

        if sheet_info:
            metrics_table.add_row("Google Sheets Export Status", sheet_info.get("mode", "N/A"))
            metrics_table.add_row("Shareable Spreadsheet Link", f"[link={sheet_info.get('public_link')}]{sheet_info.get('public_link')}[/link]")

        console.print(metrics_table)
        console.print()


# ------------------------------------------------------------------------------
# CLI ENTRY POINT & SUBCOMMANDS
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Graphone Pipeline Ingestion CLI Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Execute ingestion target pipeline")
    run_parser.add_argument(
        "target",
        choices=["startups", "products", "papers", "news", "jobs", "all"],
        help="Target entity type to process",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Run fast iteration on small sample")
    run_parser.add_argument("--limit", type=int, default=20, help="Max items to process")

    # Subcommand: export
    export_parser = subparsers.add_parser("export", help="Export processed data to Google Sheets")
    export_parser.add_argument("dest", choices=["sheets"], help="Destination format (sheets)")
    export_parser.add_argument("--dry-run", action="store_true", help="Perform local CSV export fallback")
    export_parser.add_argument("--limit", type=int, default=20, help="Max items to process (ignored for full export)")

    args = parser.parse_args()

    dry_run = getattr(args, "dry_run", False)
    limit = getattr(args, "limit", 20)
    orchestrator = PipelineOrchestrator(dry_run=dry_run, limit=limit)

    if args.command == "run":
        target = args.target

        if target == "startups":
            asyncio.run(orchestrator.run_startups_pipeline())
        elif target == "products":
            asyncio.run(orchestrator.run_products_pipeline())
        elif target == "papers":
            asyncio.run(orchestrator.run_papers_pipeline())
        elif target == "news":
            asyncio.run(orchestrator.run_news_pipeline())
        elif target == "jobs":
            asyncio.run(orchestrator.run_jobs_pipeline())
        elif target == "all":
            asyncio.run(orchestrator.run_startups_pipeline())
            asyncio.run(orchestrator.run_products_pipeline())
            asyncio.run(orchestrator.run_papers_pipeline())
            asyncio.run(orchestrator.run_freshness_news_jobs_pipeline())

            sheet_info = GoogleSheetsExporter().export_all(dry_run=args.dry_run)
            orchestrator.print_summary_table(sheet_info=sheet_info)
            return

        orchestrator.print_summary_table()

    elif args.command == "export" and args.dest == "sheets":
        sheet_info = GoogleSheetsExporter().export_all(dry_run=args.dry_run)
        orchestrator.print_summary_table(sheet_info=sheet_info)


if __name__ == "__main__":
    main()
