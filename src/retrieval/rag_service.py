"""
================================================================================
GRAPHONE PIPELINE: CITATION-AWARE RAG SERVICE
================================================================================
Orchestrates:
1. Hybrid retrieval (Dense vector similarity + Knowledge Graph multi-hop BFS).
2. Provenance-tracked chunk assembly with stable document and chunk IDs.
3. Evidence-aware response policy & abstention handling.
4. Grounded answer generation with structured [1], [2] citation markers.
5. Multi-stage citation and semantic claim validation.
6. Execution metrics and telemetry recording.
================================================================================
"""

import time
from typing import Any, List, Optional, Set, Tuple

import structlog

from src.observability.metrics import metrics_collector
from src.retrieval.citation_validator import (
    CitationValidator,
    citation_validator,
    clean_tokens,
)
from src.retrieval.models import (
    Citation,
    HybridRetrievalRequest,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievalCandidate,
    SourceChunkProvenance,
)
from src.retrieval.service import HybridRetrievalService, hybrid_retrieval_service

logger = structlog.get_logger(__name__)

STANDARD_ABSTENTION_MESSAGE = (
    "I do not have sufficient verifiable evidence in the retrieved sources to answer this question. "
    "The available sources are insufficient to support a grounded factual answer."
)



class RAGService:
    """
    Production-grade Citation-Aware RAG Engine.
    Combines hybrid retrieval with grounded provenance-backed generation and validation.
    """

    def __init__(
        self,
        retrieval_svc: Optional[HybridRetrievalService] = None,
        validator: Optional[CitationValidator] = None,
        llm_chain: Optional[Any] = None,
    ):
        self.retrieval_service = retrieval_svc or hybrid_retrieval_service
        self.validator = validator or citation_validator
        self.llm_chain = llm_chain

    def answer_query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """
        Execute full grounded citation-aware RAG pipeline:
        Retrieval -> Provenance Assembly -> Generation -> Citation Validation.
        """
        t_start = time.perf_counter()
        clean_query = request.query.strip()

        # Step 1: Hybrid Retrieval Stage
        retrieval_req = HybridRetrievalRequest(
            query=clean_query,
            candidate_type=request.candidate_type,
            vector_limit=request.vector_limit,
            graph_depth=request.graph_depth,
            max_graph_candidates=request.max_graph_candidates,
            vector_weight=request.vector_weight,
            graph_weight=request.graph_weight,
            ranking_strategy=request.ranking_strategy,
            min_score=request.min_score,
            limit=request.limit,
        )

        retrieval_res = self.retrieval_service.search_hybrid(retrieval_req)

        # Step 2: Provenance-tracked Chunk Assembly
        sources = self._assemble_provenance_sources(retrieval_res.results)

        # Step 3: Evidence Policy & Sufficiency Check
        is_sufficient, reason = self._check_evidence_sufficiency(clean_query, sources)

        # Handle Abstention if evidence is insufficient
        if not is_sufficient:
            t_gen = time.perf_counter()
            generation_ms = (t_gen - t_start) * 1000.0

            # Validate the abstention outcome
            val_res = self.validator.validate(
                answer=STANDARD_ABSTENTION_MESSAGE,
                sources=sources,
                citations=[],
                semantic_validation=request.semantic_validation,
                validator_type=request.validator_type,
            )
            t_end = time.perf_counter()
            validation_ms = (t_end - t_gen) * 1000.0
            total_ms = (t_end - t_start) * 1000.0

            metrics_collector.record_rag_query(
                latency_seconds=total_ms / 1000.0,
                citations_count=0,
                invalid_citations_count=0,
                unsupported_claims_count=0,
                is_abstained=True,
                validation_status=val_res.status.value,
            )

            return RAGQueryResponse(
                query=clean_query,
                answer=STANDARD_ABSTENTION_MESSAGE,
                citations=[],
                sources=sources,
                abstained=True,
                validation_status=val_res.status,
                validation_result=val_res,
                retrieval_response=retrieval_res,
                execution_ms=round(total_ms, 3),
                generation_ms=round(generation_ms, 3),
                validation_ms=round(validation_ms, 3),
            )

        # Step 4: Generation Stage
        t_gen_start = time.perf_counter()
        answer, citations = self._generate_grounded_answer(clean_query, sources)
        t_gen_end = time.perf_counter()
        generation_ms = (t_gen_end - t_gen_start) * 1000.0

        # Step 5: Citation & Evidence Validation Stage
        t_val_start = time.perf_counter()
        val_res = self.validator.validate(
            answer=answer,
            sources=sources,
            citations=citations,
            semantic_validation=request.semantic_validation,
            validator_type=request.validator_type,
        )
        t_val_end = time.perf_counter()
        validation_ms = (t_val_end - t_val_start) * 1000.0
        total_ms = (t_val_end - t_start) * 1000.0

        # Record system observability metrics
        metrics_collector.record_rag_query(
            latency_seconds=total_ms / 1000.0,
            citations_count=val_res.total_citations,
            invalid_citations_count=val_res.invalid_citations,
            unsupported_claims_count=val_res.unsupported_claims,
            is_abstained=False,
            validation_status=val_res.status.value,
        )

        return RAGQueryResponse(
            query=clean_query,
            answer=answer,
            citations=citations,
            sources=sources,
            abstained=False,
            validation_status=val_res.status,
            validation_result=val_res,
            retrieval_response=retrieval_res,
            execution_ms=round(total_ms, 3),
            generation_ms=round(generation_ms, 3),
            validation_ms=round(validation_ms, 3),
        )

    def _assemble_provenance_sources(
        self, candidates: List[RetrievalCandidate]
    ) -> List[SourceChunkProvenance]:
        """
        Transform retrieved candidates into verified, provenance-rich source chunks.
        Supports duplicate chunks without corrupting citation indices.
        """
        sources: List[SourceChunkProvenance] = []

        for idx, cand in enumerate(candidates, start=1):
            payload = cand.payload or {}

            # Stable document ID
            doc_id = payload.get("id") or cand.id
            if not doc_id.startswith("doc-"):
                doc_id = f"doc-{cand.candidate_type}-{doc_id.lstrip('node-')}"

            # Stable chunk ID
            chunk_id = f"{doc_id}:chunk-0"

            # Provenance metadata - preserve explicit absence, never fabricate
            source_url = None
            if isinstance(payload.get("source"), dict) and payload["source"].get("url"):
                source_url = payload["source"]["url"]
            elif payload.get("pdf_url"):
                source_url = payload["pdf_url"]
            elif payload.get("url"):
                source_url = payload["url"]

            page_number = payload.get("page_number")
            section = payload.get("section")

            # Extract verifiable content excerpt
            text_parts = []
            if cand.title:
                text_parts.append(f"{cand.title}:")
            if cand.text:
                text_parts.append(cand.text)
            elif payload.get("description"):
                text_parts.append(str(payload.get("description")))
            elif payload.get("tagline"):
                text_parts.append(str(payload.get("tagline")))
            elif payload.get("abstract"):
                text_parts.append(str(payload.get("abstract")))

            # Include key verifiable attributes from payload
            if payload.get("location"):
                text_parts.append(f"Location: {payload['location']}.")
            if payload.get("total_funding"):
                text_parts.append(f"Funding: {payload['total_funding']}.")
            if payload.get("stage"):
                text_parts.append(f"Stage: {payload['stage']}.")
            if payload.get("founding_year"):
                text_parts.append(f"Founded: {payload['founding_year']}.")
            if payload.get("maker_company"):
                text_parts.append(f"Maker: {payload['maker_company']}.")
            if payload.get("authors"):
                text_parts.append(f"Authors: {', '.join(payload['authors'])}.")
            if cand.path_evidence:
                text_parts.append(f"Relationships: {'; '.join(cand.path_evidence)}.")

            full_text = " ".join(text_parts).strip()
            if not full_text:
                full_text = f"{cand.title} ({cand.candidate_type})"

            provenance = SourceChunkProvenance(
                index=idx,
                source_id=cand.id,
                document_id=doc_id,
                chunk_id=chunk_id,
                title=cand.title,
                source_url=source_url,
                page_number=page_number,
                section=section,
                text=full_text,
                retrieval_method=cand.source.value,
                retrieval_score=cand.final_score,
                candidate_type=cand.candidate_type,
                metadata={k: v for k, v in payload.items() if k not in ("vector", "embedding")},
            )
            sources.append(provenance)

        return sources

    def _check_evidence_sufficiency(
        self, query: str, sources: List[SourceChunkProvenance]
    ) -> Tuple[bool, str]:
        """
        Evidence-aware response policy:
        Decides whether retrieved context contains sufficient evidence to answer,
        or whether the system must abstain.
        """
        if not sources:
            return False, "No sources retrieved"

        query_tokens = clean_tokens(query)
        if not query_tokens:
            return False, "Query contains no content tokens"

        # Check total lexical/entity overlap across all retrieved sources
        all_source_tokens: Set[str] = set()
        for s in sources:
            all_source_tokens.update(clean_tokens(s.title))
            all_source_tokens.update(clean_tokens(s.text))

        overlap = query_tokens.intersection(all_source_tokens)

        # Overlap ratio relative to query tokens
        overlap_ratio = len(overlap) / len(query_tokens)

        # If multi-token query has fewer than 2 matching terms or overlap ratio is too low
        if len(query_tokens) >= 3 and len(overlap) < 2:
            return False, f"Insufficient query evidence: only {len(overlap)} of {len(query_tokens)} terms match"

        if overlap_ratio < 0.30:
            return False, f"Low query term coverage ({round(overlap_ratio, 2)} < 0.30)"

        top_score = max((s.retrieval_score for s in sources), default=0.0)
        if top_score < 0.25 and overlap_ratio < 0.50:
            return False, f"Low retrieval confidence ({top_score}) with low evidence overlap"

        return True, f"Found {len(overlap)} matching evidence terms: {list(overlap)[:3]}"

    def _generate_grounded_answer(
        self, query: str, sources: List[SourceChunkProvenance]
    ) -> Tuple[str, List[Citation]]:
        """
        Generate grounded answer with [1], [2] citation markers mapped to source chunks.
        """
        # If external LLM chain available, invoke it
        if self.llm_chain and hasattr(self.llm_chain, "generate_with_citations"):
            try:
                return self.llm_chain.generate_with_citations(query, sources)
            except Exception as e:
                logger.warning("LLM generation failed, using deterministic grounded generator", error=str(e))

        # Deterministic Grounded Generation Engine
        return self._deterministic_grounded_generation(query, sources)

    def _deterministic_grounded_generation(
        self, query: str, sources: List[SourceChunkProvenance]
    ) -> Tuple[str, List[Citation]]:
        """
        Deterministic, factually grounded generation engine that synthesizes statements
        directly from retrieved sources and attaches valid [1], [2] citations.
        """
        query_tokens = clean_tokens(query)

        # Match sources relevant to query tokens
        matched_sources: List[Tuple[SourceChunkProvenance, float]] = []
        for s in sources:
            s_tokens = clean_tokens(s.title + " " + s.text)
            overlap = query_tokens.intersection(s_tokens)
            score = len(overlap) / max(1, len(query_tokens))
            if overlap:
                matched_sources.append((s, score))

        # Sort by relevance to query
        matched_sources.sort(key=lambda x: -x[1])
        relevant_sources = [s for s, _ in matched_sources[:3]] if matched_sources else sources[:2]

        answer_sentences: List[str] = []
        citations: List[Citation] = []

        for s in relevant_sources:
            payload = s.metadata
            title = s.title
            idx = s.index

            # Construct clean, factual grounded claim from verified source attributes
            claim_parts: List[str] = []
            if payload.get("tagline"):
                claim_parts.append(f"{title} is described as {payload['tagline']}")
            elif payload.get("description"):
                desc = payload["description"].rstrip(".")
                claim_parts.append(f"{title} focuses on {desc}")
            else:
                claim_parts.append(f"{title} is an entity in the pipeline")

            if payload.get("stage") and payload.get("total_funding"):
                claim_parts.append(f"with {payload['total_funding']} in {payload['stage']} funding")
            elif payload.get("total_funding"):
                claim_parts.append(f"with total funding of {payload['total_funding']}")

            if payload.get("location"):
                claim_parts.append(f"based in {payload['location']}")

            if payload.get("maker_company"):
                claim_parts.append(f"developed by {payload['maker_company']}")

            if payload.get("authors"):
                authors_str = ", ".join(payload["authors"][:2])
                claim_parts.append(f"authored by {authors_str}")

            sentence = " ".join(claim_parts) + f" [{idx}]."
            answer_sentences.append(sentence)

            citations.append(
                Citation(
                    citation_id=idx,
                    source_id=s.source_id,
                    chunk_id=s.chunk_id,
                    document_id=s.document_id,
                    claim_text=sentence,
                    text_span=s.text[:200],
                    reference_valid=True,
                    claim_supported=True,
                )
            )

        # If multiple sources support related aspects, construct combined answer
        final_answer = " ".join(answer_sentences)
        return final_answer, citations


# Global singleton instance
rag_service = RAGService()
