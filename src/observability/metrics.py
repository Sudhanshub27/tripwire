"""
================================================================================
GRAPHONE PIPELINE: ENTERPRISE OBSERVABILITY & PROMETHEUS METRICS MODULE
================================================================================

Provides real-time system metrics, latency histograms, LLM token counters,
entity resolution merger counts, and a Prometheus-formatted /metrics endpoint output.
================================================================================
"""

from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Singleton Prometheus & Operational Metrics Collector for Graphone Pipeline."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance._init_metrics()
        return cls._instance

    def _init_metrics(self):
        self.records_ingested: Dict[str, int] = {
            "startup": 0,
            "product": 0,
            "research_paper": 0,
            "job": 0,
            "news": 0,
        }
        self.llm_tokens: Dict[str, int] = {
            "Gemini": 0,
            "Groq": 0,
            "DeepSeek": 0,
            "Heuristic": 0,
        }
        self.llm_latency_sum: Dict[str, float] = {
            "Gemini": 0.0,
            "Groq": 0.0,
            "DeepSeek": 0.0,
            "Heuristic": 0.0,
        }
        self.llm_call_count: Dict[str, int] = {
            "Gemini": 0,
            "Groq": 0,
            "DeepSeek": 0,
            "Heuristic": 0,
        }
        self.er_merges: Dict[str, int] = {
            "exact": 0,
            "normalized": 0,
            "fuzzy": 0,
            "unresolved": 0,
        }
        # Incremental processing counters
        self.incremental_total: int = 0
        self.incremental_new: int = 0
        self.incremental_changed: int = 0
        self.incremental_unchanged: int = 0
        self.incremental_skipped: int = 0
        self.llm_calls_saved: int = 0
        self.embedding_calls_saved: int = 0

    def record_ingested_entity(self, entity_type: str, count: int = 1):
        """Record newly ingested entity count."""
        key = entity_type.lower()
        if key in self.records_ingested:
            self.records_ingested[key] += count
        else:
            self.records_ingested[key] = count

    def record_llm_call(self, provider_family: str, latency_seconds: float, token_count: int):
        """Record LLM call latency and tokens."""
        fam = provider_family.capitalize()
        if fam not in self.llm_tokens:
            fam = "Heuristic"
        self.llm_tokens[fam] += token_count
        self.llm_latency_sum[fam] += latency_seconds
        self.llm_call_count[fam] += 1

    def record_er_merge(self, method: str):
        """Record entity resolution merger method."""
        m = method.lower()
        if m in self.er_merges:
            self.er_merges[m] += 1
        else:
            self.er_merges["unresolved"] += 1

    def record_incremental_decision(
        self,
        decision: Any,
        action: Optional[Any] = None,
        llm_calls_saved: int = 0,
        embedding_calls_saved: int = 0,
    ):
        """Record incremental decision telemetry and saved resource calls."""
        self.incremental_total += 1
        d_raw = decision.value if hasattr(decision, "value") else str(decision)
        d = d_raw.upper()
        if d == "NEW":
            self.incremental_new += 1
        elif d == "CHANGED":
            self.incremental_changed += 1
        elif d == "UNCHANGED":
            self.incremental_unchanged += 1

        a_raw = action.value if hasattr(action, "value") else (str(action) if action is not None else None)
        act = str(a_raw or ("SKIP" if d == "UNCHANGED" else "PROCESS")).upper()
        if act == "SKIP":
            self.incremental_skipped += 1

        self.llm_calls_saved += llm_calls_saved
        self.embedding_calls_saved += embedding_calls_saved

    def reset_incremental_metrics(self):
        """Reset incremental counters (useful for isolated benchmark runs or tests)."""
        self.incremental_total = 0
        self.incremental_new = 0
        self.incremental_changed = 0
        self.incremental_unchanged = 0
        self.incremental_skipped = 0
        self.llm_calls_saved = 0
        self.embedding_calls_saved = 0

    def get_incremental_stats(self) -> Dict[str, Any]:
        """Return snapshot of incremental processing metrics including calculated work avoided percentage."""
        work_avoided_pct = (
            round((self.incremental_skipped / self.incremental_total) * 100.0, 1)
            if self.incremental_total > 0
            else 0.0
        )
        return {
            "incremental_total": self.incremental_total,
            "incremental_new": self.incremental_new,
            "incremental_changed": self.incremental_changed,
            "incremental_unchanged": self.incremental_unchanged,
            "incremental_skipped": self.incremental_skipped,
            "llm_calls_saved": self.llm_calls_saved,
            "embedding_calls_saved": self.embedding_calls_saved,
            "work_avoided_pct": work_avoided_pct,
        }

    def generate_prometheus_format(self) -> str:
        """Generate Prometheus exposition text format for /metrics scraping."""
        lines: List[str] = [
            "# HELP graphone_records_ingested_total Total count of processed entity records by type.",
            "# TYPE graphone_records_ingested_total counter",
        ]
        for etype, count in self.records_ingested.items():
            lines.append(f'graphone_records_ingested_total{{entity_type="{etype}"}} {count}')

        lines.extend([
            "# HELP graphone_llm_tokens_total Total tokens processed per LLM provider family.",
            "# TYPE graphone_llm_tokens_total counter",
        ])
        for provider, tokens in self.llm_tokens.items():
            lines.append(f'graphone_llm_tokens_total{{provider="{provider}"}} {tokens}')

        lines.extend([
            "# HELP graphone_llm_calls_total Total LLM calls per provider family.",
            "# TYPE graphone_llm_calls_total counter",
        ])
        for provider, calls in self.llm_call_count.items():
            lines.append(f'graphone_llm_calls_total{{provider="{provider}"}} {calls}')

        lines.extend([
            "# HELP graphone_llm_avg_latency_seconds Average latency per provider call in seconds.",
            "# TYPE graphone_llm_avg_latency_seconds gauge",
        ])
        for provider, ccount in self.llm_call_count.items():
            avg_lat = round(self.llm_latency_sum[provider] / ccount, 4) if ccount > 0 else 0.0
            lines.append(f'graphone_llm_avg_latency_seconds{{provider="{provider}"}} {avg_lat}')

        lines.extend([
            "# HELP graphone_entity_resolution_merges_total Deduplication merges by resolution method.",
            "# TYPE graphone_entity_resolution_merges_total counter",
        ])
        for method, count in self.er_merges.items():
            lines.append(f'graphone_entity_resolution_merges_total{{method="{method}"}} {count}')

        # Incremental Processing Prometheus Counters
        lines.extend([
            "# HELP graphone_incremental_total Total incremental processing checks evaluated.",
            "# TYPE graphone_incremental_total counter",
            f"graphone_incremental_total {self.incremental_total}",
            "# HELP graphone_incremental_new Total new documents evaluated.",
            "# TYPE graphone_incremental_new counter",
            f"graphone_incremental_new {self.incremental_new}",
            "# HELP graphone_incremental_changed Total documents with changed content.",
            "# TYPE graphone_incremental_changed counter",
            f"graphone_incremental_changed {self.incremental_changed}",
            "# HELP graphone_incremental_unchanged Total documents with unchanged content.",
            "# TYPE graphone_incremental_unchanged counter",
            f"graphone_incremental_unchanged {self.incremental_unchanged}",
            "# HELP graphone_incremental_skipped Total documents skipped from reprocessing.",
            "# TYPE graphone_incremental_skipped counter",
            f"graphone_incremental_skipped {self.incremental_skipped}",
            "# HELP graphone_llm_calls_saved_total Total LLM calls saved via incremental skipping.",
            "# TYPE graphone_llm_calls_saved_total counter",
            f"graphone_llm_calls_saved_total {self.llm_calls_saved}",
            "# HELP graphone_embedding_calls_saved_total Total embedding calls saved via incremental skipping.",
            "# TYPE graphone_embedding_calls_saved_total counter",
            f"graphone_embedding_calls_saved_total {self.embedding_calls_saved}",
        ])

        return "\n".join(lines) + "\n"


metrics_collector = MetricsCollector()
