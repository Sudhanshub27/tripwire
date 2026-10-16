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
        # Entity resolution metrics (Phase 3)
        self.er_total: int = 0
        self.er_auto_matched: int = 0
        self.er_needs_review: int = 0
        self.er_no_match: int = 0
        self.er_review_approved: int = 0
        self.er_review_rejected: int = 0
        self.er_confidence_sum: float = 0.0
        self.er_confidence_count: int = 0
        self.er_latency_sum: float = 0.0
        self.er_latency_count: int = 0
        # Incremental processing counters
        self.incremental_total: int = 0
        self.incremental_new: int = 0
        self.incremental_changed: int = 0
        self.incremental_unchanged: int = 0
        self.incremental_skipped: int = 0
        self.llm_calls_saved: int = 0
        self.embedding_calls_saved: int = 0
        # Graph query engine metrics (Phase 4)
        self.graph_queries_total: Dict[str, int] = {
            "node_lookup": 0,
            "neighbors": 0,
            "traverse": 0,
            "path": 0,
            "subgraph": 0,
            "relationship": 0,
        }
        self.graph_query_errors: int = 0
        self.graph_query_latency_sum: float = 0.0
        self.graph_query_latency_count: int = 0
        self.graph_nodes_visited: int = 0
        self.graph_nodes_returned: int = 0
        self.graph_results_truncated: int = 0
        # Hybrid retrieval metrics (Phase 5)
        self.hybrid_queries_total: int = 0
        self.hybrid_query_errors: int = 0
        self.hybrid_query_latency_sum: float = 0.0
        self.hybrid_query_latency_count: int = 0
        self.hybrid_vector_candidates: int = 0
        self.hybrid_graph_candidates: int = 0
        self.hybrid_merged_candidates: int = 0
        self.hybrid_final_results: int = 0
        self.hybrid_graph_expansions_truncated: int = 0
        # Citation-Aware RAG metrics (Phase 6)
        self.rag_queries_total: int = 0
        self.rag_citation_validations_total: int = 0
        self.rag_invalid_citations_total: int = 0
        self.rag_unsupported_claims_total: int = 0
        self.rag_abstentions_total: int = 0
        self.rag_validation_failures_total: int = 0
        self.rag_query_latency_sum: float = 0.0
        self.rag_query_latency_count: int = 0
        self.rag_validation_latency_sum: float = 0.0
        self.rag_validation_latency_count: int = 0

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

    def record_er_resolution(self, decision: Any, confidence: float, latency_seconds: float = 0.0):
        """Record entity resolution decision, confidence, and latency."""
        self.er_total += 1
        d_raw = decision.value if hasattr(decision, "value") else str(decision)
        d = d_raw.upper()
        if d in ("AUTO_MATCH", "MATCH"):
            self.er_auto_matched += 1
        elif d in ("NEEDS_REVIEW", "REVIEW"):
            self.er_needs_review += 1
        elif d in ("NO_MATCH", "NEW", "UNRESOLVED"):
            self.er_no_match += 1

        self.er_confidence_sum += float(confidence)
        self.er_confidence_count += 1
        if latency_seconds > 0.0:
            self.er_latency_sum += float(latency_seconds)
            self.er_latency_count += 1

    def record_er_review_action(self, action: str):
        """Record human review action (approve / reject)."""
        act = action.lower()
        if act in ("approve", "approved"):
            self.er_review_approved += 1
        elif act in ("reject", "rejected"):
            self.er_review_rejected += 1

    def reset_er_metrics(self):
        """Reset entity resolution metric counters."""
        self.er_total = 0
        self.er_auto_matched = 0
        self.er_needs_review = 0
        self.er_no_match = 0
        self.er_review_approved = 0
        self.er_review_rejected = 0
        self.er_confidence_sum = 0.0
        self.er_confidence_count = 0
        self.er_latency_sum = 0.0
        self.er_latency_count = 0
        self.er_merges = {
            "exact": 0,
            "normalized": 0,
            "fuzzy": 0,
            "unresolved": 0,
        }

    def get_er_metrics(self) -> Dict[str, Any]:
        """Return snapshot of entity resolution metrics."""
        avg_conf = (
            round(self.er_confidence_sum / self.er_confidence_count, 4)
            if self.er_confidence_count > 0
            else 0.0
        )
        avg_lat = (
            round((self.er_latency_sum / self.er_latency_count) * 1000.0, 2)
            if self.er_latency_count > 0
            else 0.0
        )
        queue_size = 0
        try:
            from src.resolution.review_queue import review_queue
            queue_size = len(review_queue.get_pending_reviews())
        except Exception:
            queue_size = max(0, self.er_needs_review - self.er_review_approved - self.er_review_rejected)

        return {
            "entity_resolution_total": self.er_total,
            "entity_resolution_auto_matched": self.er_auto_matched,
            "entity_resolution_needs_review": self.er_needs_review,
            "entity_resolution_no_match": self.er_no_match,
            "entity_resolution_review_approved": self.er_review_approved,
            "entity_resolution_review_rejected": self.er_review_rejected,
            "average_confidence": avg_conf,
            "review_queue_size": queue_size,
            "resolution_latency_avg_ms": avg_lat,
        }

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

    def record_graph_query(
        self,
        query_type: str,
        latency_seconds: float,
        nodes_visited: int = 0,
        nodes_returned: int = 0,
        is_truncated: bool = False,
        is_error: bool = False,
    ):
        """Record graph query execution metrics."""
        qtype = query_type.lower()
        if qtype in self.graph_queries_total:
            self.graph_queries_total[qtype] += 1
        else:
            self.graph_queries_total[qtype] = 1

        if is_error:
            self.graph_query_errors += 1

        self.graph_query_latency_sum += float(latency_seconds)
        self.graph_query_latency_count += 1
        self.graph_nodes_visited += int(nodes_visited)
        self.graph_nodes_returned += int(nodes_returned)
        if is_truncated:
            self.graph_results_truncated += 1

    def reset_graph_metrics(self):
        """Reset graph query metrics counters."""
        self.graph_queries_total = {
            "node_lookup": 0,
            "neighbors": 0,
            "traverse": 0,
            "path": 0,
            "subgraph": 0,
            "relationship": 0,
        }
        self.graph_query_errors = 0
        self.graph_query_latency_sum = 0.0
        self.graph_query_latency_count = 0
        self.graph_nodes_visited = 0
        self.graph_nodes_returned = 0
        self.graph_results_truncated = 0

    def get_graph_metrics(self) -> Dict[str, Any]:
        """Return snapshot of graph query metrics."""
        avg_lat = (
            round((self.graph_query_latency_sum / self.graph_query_latency_count) * 1000.0, 2)
            if self.graph_query_latency_count > 0
            else 0.0
        )
        total_queries = sum(self.graph_queries_total.values())
        return {
            "graph_queries_total": total_queries,
            "graph_queries_by_type": dict(self.graph_queries_total),
            "graph_query_errors": self.graph_query_errors,
            "graph_query_avg_latency_ms": avg_lat,
            "graph_nodes_visited": self.graph_nodes_visited,
            "graph_nodes_returned": self.graph_nodes_returned,
            "graph_results_truncated": self.graph_results_truncated,
        }

    def record_hybrid_query(
        self,
        latency_seconds: float,
        vector_candidates: int = 0,
        graph_candidates: int = 0,
        merged_candidates: int = 0,
        final_results: int = 0,
        is_graph_truncated: bool = False,
        is_error: bool = False,
    ):
        """Record hybrid vector + graph retrieval execution telemetry."""
        self.hybrid_queries_total += 1
        if is_error:
            self.hybrid_query_errors += 1

        self.hybrid_query_latency_sum += float(latency_seconds)
        self.hybrid_query_latency_count += 1
        self.hybrid_vector_candidates += int(vector_candidates)
        self.hybrid_graph_candidates += int(graph_candidates)
        self.hybrid_merged_candidates += int(merged_candidates)
        self.hybrid_final_results += int(final_results)
        if is_graph_truncated:
            self.hybrid_graph_expansions_truncated += 1

    def reset_hybrid_metrics(self):
        """Reset hybrid retrieval metrics counters."""
        self.hybrid_queries_total = 0
        self.hybrid_query_errors = 0
        self.hybrid_query_latency_sum = 0.0
        self.hybrid_query_latency_count = 0
        self.hybrid_vector_candidates = 0
        self.hybrid_graph_candidates = 0
        self.hybrid_merged_candidates = 0
        self.hybrid_final_results = 0
        self.hybrid_graph_expansions_truncated = 0

    def get_hybrid_metrics(self) -> Dict[str, Any]:
        """Return snapshot of hybrid retrieval metrics."""
        avg_lat = (
            round((self.hybrid_query_latency_sum / self.hybrid_query_latency_count) * 1000.0, 2)
            if self.hybrid_query_latency_count > 0
            else 0.0
        )
        return {
            "hybrid_queries_total": self.hybrid_queries_total,
            "hybrid_query_errors": self.hybrid_query_errors,
            "hybrid_query_avg_latency_ms": avg_lat,
            "hybrid_vector_candidates_total": self.hybrid_vector_candidates,
            "hybrid_graph_candidates_total": self.hybrid_graph_candidates,
            "hybrid_merged_candidates_total": self.hybrid_merged_candidates,
            "hybrid_final_results_total": self.hybrid_final_results,
            "hybrid_graph_expansions_truncated": self.hybrid_graph_expansions_truncated,
        }

    def record_rag_query(
        self,
        latency_seconds: float,
        citations_count: int = 0,
        invalid_citations_count: int = 0,
        unsupported_claims_count: int = 0,
        is_abstained: bool = False,
        validation_status: str = "VALID",
    ):
        """Record Citation-Aware RAG query telemetry and citation outcomes."""
        self.rag_queries_total += 1
        self.rag_query_latency_sum += float(latency_seconds)
        self.rag_query_latency_count += 1
        self.rag_citation_validations_total += int(citations_count)
        self.rag_invalid_citations_total += int(invalid_citations_count)
        self.rag_unsupported_claims_total += int(unsupported_claims_count)
        if is_abstained:
            self.rag_abstentions_total += 1
        if validation_status == "VALIDATION_ERROR":
            self.rag_validation_failures_total += 1

    def record_citation_validation(
        self,
        is_valid: bool,
        is_semantic_supported: bool,
        is_error: bool = False,
        latency_seconds: float = 0.0,
    ):
        """Record standalone citation validation execution."""
        self.rag_citation_validations_total += 1
        if not is_valid:
            self.rag_invalid_citations_total += 1
        if not is_semantic_supported:
            self.rag_unsupported_claims_total += 1
        if is_error:
            self.rag_validation_failures_total += 1
        if latency_seconds > 0:
            self.rag_validation_latency_sum += float(latency_seconds)
            self.rag_validation_latency_count += 1

    def reset_rag_metrics(self):
        """Reset Citation-Aware RAG metrics counters."""
        self.rag_queries_total = 0
        self.rag_citation_validations_total = 0
        self.rag_invalid_citations_total = 0
        self.rag_unsupported_claims_total = 0
        self.rag_abstentions_total = 0
        self.rag_validation_failures_total = 0
        self.rag_query_latency_sum = 0.0
        self.rag_query_latency_count = 0
        self.rag_validation_latency_sum = 0.0
        self.rag_validation_latency_count = 0

    def get_rag_metrics(self) -> Dict[str, Any]:
        """Return snapshot of Citation-Aware RAG metrics."""
        avg_q_lat = (
            round((self.rag_query_latency_sum / self.rag_query_latency_count) * 1000.0, 2)
            if self.rag_query_latency_count > 0
            else 0.0
        )
        avg_v_lat = (
            round((self.rag_validation_latency_sum / self.rag_validation_latency_count) * 1000.0, 2)
            if self.rag_validation_latency_count > 0
            else 0.0
        )
        return {
            "rag_queries_total": self.rag_queries_total,
            "rag_citation_validations_total": self.rag_citation_validations_total,
            "rag_invalid_citations_total": self.rag_invalid_citations_total,
            "rag_unsupported_claims_total": self.rag_unsupported_claims_total,
            "rag_abstentions_total": self.rag_abstentions_total,
            "rag_validation_failures_total": self.rag_validation_failures_total,
            "rag_query_avg_latency_ms": avg_q_lat,
            "rag_validation_avg_latency_ms": avg_v_lat,
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

        # Entity Resolution 2.0 Prometheus Metrics (Phase 3)
        er_stats = self.get_er_metrics()
        lines.extend([
            "# HELP graphone_entity_resolution_total Total entity resolution evaluations.",
            "# TYPE graphone_entity_resolution_total counter",
            f"graphone_entity_resolution_total {self.er_total}",
            "# HELP graphone_entity_resolution_auto_matched Total auto-matched entities.",
            "# TYPE graphone_entity_resolution_auto_matched counter",
            f"graphone_entity_resolution_auto_matched {self.er_auto_matched}",
            "# HELP graphone_entity_resolution_needs_review Total entities flagged for human review.",
            "# TYPE graphone_entity_resolution_needs_review counter",
            f"graphone_entity_resolution_needs_review {self.er_needs_review}",
            "# HELP graphone_entity_resolution_no_match Total distinct entities with no match found.",
            "# TYPE graphone_entity_resolution_no_match counter",
            f"graphone_entity_resolution_no_match {self.er_no_match}",
            "# HELP graphone_entity_resolution_review_approved Total reviews approved by human reviewers.",
            "# TYPE graphone_entity_resolution_review_approved counter",
            f"graphone_entity_resolution_review_approved {self.er_review_approved}",
            "# HELP graphone_entity_resolution_review_rejected Total reviews rejected by human reviewers.",
            "# TYPE graphone_entity_resolution_review_rejected counter",
            f"graphone_entity_resolution_review_rejected {self.er_review_rejected}",
            "# HELP graphone_entity_resolution_avg_confidence Average confidence score across resolutions.",
            "# TYPE graphone_entity_resolution_avg_confidence gauge",
            f"graphone_entity_resolution_avg_confidence {er_stats['average_confidence']}",
            "# HELP graphone_entity_resolution_review_queue_size Current pending review queue size.",
            "# TYPE graphone_entity_resolution_review_queue_size gauge",
            f"graphone_entity_resolution_review_queue_size {er_stats['review_queue_size']}",
            "# HELP graphone_entity_resolution_avg_latency_seconds Average resolution latency in seconds.",
            "# TYPE graphone_entity_resolution_avg_latency_seconds gauge",
            f"graphone_entity_resolution_avg_latency_seconds {round(er_stats['resolution_latency_avg_ms'] / 1000.0, 4)}",
        ])

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

        # Graph Query Engine Prometheus Metrics (Phase 4)
        graph_stats = self.get_graph_metrics()
        lines.extend([
            "# HELP graphone_graph_queries_total Total graph queries executed by query type.",
            "# TYPE graphone_graph_queries_total counter",
        ])
        for qtype, qcount in self.graph_queries_total.items():
            lines.append(f'graphone_graph_queries_total{{query_type="{qtype}"}} {qcount}')

        lines.extend([
            "# HELP graphone_graph_query_errors_total Total graph queries resulting in errors or missing roots.",
            "# TYPE graphone_graph_query_errors_total counter",
            f"graphone_graph_query_errors_total {self.graph_query_errors}",
            "# HELP graphone_graph_query_avg_latency_seconds Average graph query execution latency in seconds.",
            "# TYPE graphone_graph_query_avg_latency_seconds gauge",
            f"graphone_graph_query_avg_latency_seconds {round(graph_stats['graph_query_avg_latency_ms'] / 1000.0, 4)}",
            "# HELP graphone_graph_nodes_visited_total Total graph nodes visited during traversals.",
            "# TYPE graphone_graph_nodes_visited_total counter",
            f"graphone_graph_nodes_visited_total {self.graph_nodes_visited}",
            "# HELP graphone_graph_nodes_returned_total Total graph nodes returned in query responses.",
            "# TYPE graphone_graph_nodes_returned_total counter",
            f"graphone_graph_nodes_returned_total {self.graph_nodes_returned}",
            "# HELP graphone_graph_results_truncated_total Total query executions where results were truncated.",
            "# TYPE graphone_graph_results_truncated_total counter",
            f"graphone_graph_results_truncated_total {self.graph_results_truncated}",
        ])

        # Hybrid Retrieval Prometheus Metrics (Phase 5)
        hybrid_stats = self.get_hybrid_metrics()
        lines.extend([
            "# HELP graphone_hybrid_queries_total Total hybrid vector+graph queries executed.",
            "# TYPE graphone_hybrid_queries_total counter",
            f"graphone_hybrid_queries_total {self.hybrid_queries_total}",
            "# HELP graphone_hybrid_query_errors_total Total hybrid queries resulting in errors.",
            "# TYPE graphone_hybrid_query_errors_total counter",
            f"graphone_hybrid_query_errors_total {self.hybrid_query_errors}",
            "# HELP graphone_hybrid_query_avg_latency_seconds Average hybrid query execution latency in seconds.",
            "# TYPE graphone_hybrid_query_avg_latency_seconds gauge",
            f"graphone_hybrid_query_avg_latency_seconds {round(hybrid_stats['hybrid_query_avg_latency_ms'] / 1000.0, 4)}",
            "# HELP graphone_hybrid_vector_candidates_total Total vector candidates retrieved in hybrid searches.",
            "# TYPE graphone_hybrid_vector_candidates_total counter",
            f"graphone_hybrid_vector_candidates_total {self.hybrid_vector_candidates}",
            "# HELP graphone_hybrid_graph_candidates_total Total graph candidates discovered in hybrid expansions.",
            "# TYPE graphone_hybrid_graph_candidates_total counter",
            f"graphone_hybrid_graph_candidates_total {self.hybrid_graph_candidates}",
            "# HELP graphone_hybrid_merged_candidates_total Total candidates merged across both pathways.",
            "# TYPE graphone_hybrid_merged_candidates_total counter",
            f"graphone_hybrid_merged_candidates_total {self.hybrid_merged_candidates}",
            "# HELP graphone_hybrid_final_results_total Total final ranked results returned by hybrid retrieval.",
            "# TYPE graphone_hybrid_final_results_total counter",
            f"graphone_hybrid_final_results_total {self.hybrid_final_results}",
            "# HELP graphone_hybrid_graph_truncated_total Total hybrid queries where graph expansions were capped.",
            "# TYPE graphone_hybrid_graph_truncated_total counter",
            f"graphone_hybrid_graph_truncated_total {self.hybrid_graph_expansions_truncated}",
        ])

        # Citation-Aware RAG Prometheus Metrics (Phase 6)
        rag_stats = self.get_rag_metrics()
        lines.extend([
            "# HELP graphone_rag_queries_total Total Citation-Aware RAG queries executed.",
            "# TYPE graphone_rag_queries_total counter",
            f"graphone_rag_queries_total {self.rag_queries_total}",
            "# HELP graphone_rag_citation_validations_total Total citation validations evaluated.",
            "# TYPE graphone_rag_citation_validations_total counter",
            f"graphone_rag_citation_validations_total {self.rag_citation_validations_total}",
            "# HELP graphone_rag_invalid_citations_total Total invalid citations detected.",
            "# TYPE graphone_rag_invalid_citations_total counter",
            f"graphone_rag_invalid_citations_total {self.rag_invalid_citations_total}",
            "# HELP graphone_rag_unsupported_claims_total Total unsupported factual claims flagged.",
            "# TYPE graphone_rag_unsupported_claims_total counter",
            f"graphone_rag_unsupported_claims_total {self.rag_unsupported_claims_total}",
            "# HELP graphone_rag_abstentions_total Total queries where RAG abstained due to insufficient evidence.",
            "# TYPE graphone_rag_abstentions_total counter",
            f"graphone_rag_abstentions_total {self.rag_abstentions_total}",
            "# HELP graphone_rag_validation_failures_total Total citation validation errors.",
            "# TYPE graphone_rag_validation_failures_total counter",
            f"graphone_rag_validation_failures_total {self.rag_validation_failures_total}",
            "# HELP graphone_rag_query_avg_latency_seconds Average RAG query latency in seconds.",
            "# TYPE graphone_rag_query_avg_latency_seconds gauge",
            f"graphone_rag_query_avg_latency_seconds {round(rag_stats['rag_query_avg_latency_ms'] / 1000.0, 4)}",
            "# HELP graphone_rag_validation_avg_latency_seconds Average validation latency in seconds.",
            "# TYPE graphone_rag_validation_avg_latency_seconds gauge",
            f"graphone_rag_validation_avg_latency_seconds {round(rag_stats['rag_validation_avg_latency_ms'] / 1000.0, 4)}",
        ])

        return "\n".join(lines) + "\n"


metrics_collector = MetricsCollector()
