"""
================================================================================
GRAPHONE PIPELINE: GRAPH + VECTOR HYBRID RETRIEVAL DATA MODELS & SCHEMAS
================================================================================
Validated Pydantic models for hybrid retrieval requests, candidates, evidence,
ranking explanations, and structured query responses.
================================================================================
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.graph.models import Direction


class RetrievalSource(str, Enum):
    """Source pathway that produced a retrieval candidate."""

    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    BOTH = "BOTH"


class RankingStrategy(str, Enum):
    """Supported ranking algorithms for candidate score combination."""

    WEIGHTED = "weighted"
    RRF = "rrf"


class PathEvidence(BaseModel):
    """Structured evidence of a graph path leading from a seed to a discovered candidate."""

    model_config = ConfigDict(extra="allow")

    seed_id: str = Field(..., description="Root seed node identifier")
    seed_title: str = Field(..., description="Root seed entity title / name")
    hops: int = Field(default=1, ge=1, description="Number of relationship hops from seed")
    relations: List[str] = Field(default_factory=list, description="Relationship types traversed")
    description: str = Field(..., description="Human-readable path expression e.g. 'A --[PRODUCES]--> B'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Cumulative edge confidence")


class RetrievalCandidate(BaseModel):
    """A ranked candidate entity resulting from hybrid, vector, or graph retrieval."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Stable unique candidate identifier")
    candidate_type: str = Field(..., description="Entity type: startup, product, research_paper, job, news")
    title: str = Field(..., description="Display or canonical entity title")
    text: Optional[str] = Field(default=None, description="Indexed descriptive text or summary")
    vector_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score from vector retrieval (if applicable)",
    )
    graph_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Graph relevance score from graph traversal (if applicable)",
    )
    graph_distance: Optional[int] = Field(
        default=None,
        ge=0,
        description="Shortest traversal distance from vector seed (0 for seed itself)",
    )
    path_evidence: List[str] = Field(
        default_factory=list,
        description="List of relationship paths connecting this candidate to seeds",
    )
    sources: List[RetrievalSource] = Field(
        default_factory=list,
        description="All retrieval pathways that surfaced this candidate",
    )
    source: RetrievalSource = Field(
        default=RetrievalSource.VECTOR,
        description="Primary retrieval source classification: VECTOR, GRAPH, or BOTH",
    )
    final_score: float = Field(
        ...,
        ge=0.0,
        description="Final deterministic ranking score",
    )
    ranking_explanation: str = Field(
        ...,
        description="Human-readable explanation of ranking score derivation",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Underlying entity attributes and metadata",
    )


class HybridRetrievalRequest(BaseModel):
    """Validated request parameters for hybrid retrieval endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language semantic search query",
    )
    candidate_type: Optional[str] = Field(
        default="all",
        description="Optional entity type filter: all, startup, product, research_paper, job, news",
    )
    vector_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum candidates retrieved in initial vector search stage",
    )
    graph_depth: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum traversal depth from vector seed entities (hops)",
    )
    max_graph_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum total candidates surfaced through graph expansion",
    )
    edge_types: Optional[List[str]] = Field(
        default=None,
        description="Optional list of edge relationship types to traverse (e.g. ['PRODUCES'])",
    )
    direction: Direction = Field(
        default=Direction.BOTH,
        description="Edge traversal direction: outgoing, incoming, both",
    )
    vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Relative weight for vector similarity score in weighted fusion",
    )
    graph_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Relative weight for graph relevance score in weighted fusion",
    )
    ranking_strategy: RankingStrategy = Field(
        default=RankingStrategy.WEIGHTED,
        description="Ranking combination strategy: 'weighted' or 'rrf'",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="Smoothing constant for Reciprocal Rank Fusion",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for final results",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of final ranked results to return",
    )


class HybridRetrievalResponse(BaseModel):
    """Structured response container for hybrid retrieval queries."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(..., description="Echo of input query")
    candidate_type: Optional[str] = Field(default="all", description="Entity filter applied")
    total_candidates: int = Field(..., ge=0, description="Count of returned candidates")
    vector_candidates_count: int = Field(..., ge=0, description="Raw candidates from vector search")
    graph_candidates_count: int = Field(..., ge=0, description="Raw candidates from graph expansion")
    merged_candidates_count: int = Field(..., ge=0, description="Deduplicated candidate count before limit")
    returned_count: int = Field(..., ge=0, description="Candidates returned after limit and min_score")
    results: List[RetrievalCandidate] = Field(default_factory=list, description="Ranked candidate results")
    execution_ms: float = Field(default=0.0, ge=0.0, description="Total execution duration in milliseconds")
    is_truncated: bool = Field(default=False, description="True if results were capped by limit")
    ranking_strategy: str = Field(default="weighted", description="Applied ranking strategy")
    message: Optional[str] = Field(default=None, description="Contextual message or status notes")


# ==============================================================================
# PHASE 6: CITATION-AWARE RAG DATA MODELS & PROVENANCE SCHEMAS
# ==============================================================================


class ValidationStatus(str, Enum):
    """Validation outcome status for RAG generation and citations."""

    VALID = "VALID"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class SourceChunkProvenance(BaseModel):
    """Verifiable provenance and content metadata for a retrieved document chunk."""

    model_config = ConfigDict(extra="allow")

    index: int = Field(..., ge=1, description="1-based citation index e.g. 1 for [1]")
    source_id: str = Field(..., description="Stable unique entity / source identifier")
    document_id: str = Field(..., description="Stable document identifier")
    chunk_id: str = Field(..., description="Stable chunk identifier (doc:chunk-N)")
    title: str = Field(..., description="Display title or entity name")
    source_url: Optional[str] = Field(default=None, description="Verified source URL or path (None if unavailable)")
    page_number: Optional[int] = Field(default=None, description="Source page number (None if unavailable)")
    section: Optional[str] = Field(default=None, description="Document section or header (None if unavailable)")
    text: str = Field(..., description="Content text excerpt of the retrieved chunk")
    retrieval_method: str = Field(default="hybrid", description="Method used: vector, graph, hybrid, both")
    retrieval_score: float = Field(default=0.0, ge=0.0, description="Retrieval ranking score")
    candidate_type: Optional[str] = Field(default=None, description="Entity type: startup, product, research_paper, etc.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional verified metadata attributes")


class Citation(BaseModel):
    """Extracted citation connecting a factual claim or span to a retrieved source chunk."""

    model_config = ConfigDict(extra="allow")

    citation_id: int = Field(..., ge=1, description="1-based citation marker number e.g. 1 for [1]")
    source_id: str = Field(..., description="Identifier of the referenced source chunk")
    chunk_id: Optional[str] = Field(default=None, description="Stable chunk identifier")
    document_id: Optional[str] = Field(default=None, description="Stable document identifier")
    claim_text: Optional[str] = Field(default=None, description="Specific factual claim or sentence supported")
    text_span: Optional[str] = Field(default=None, description="Exact or relevant supporting text excerpt from chunk")
    reference_valid: bool = Field(default=True, description="True if citation resolves to a retrieved source chunk")
    claim_supported: Optional[bool] = Field(default=None, description="True if cited chunk semantically supports claim")


class CitationValidationResult(BaseModel):
    """Detailed report produced by citation and claim-support validation."""

    model_config = ConfigDict(extra="allow")

    status: ValidationStatus = Field(..., description="Overall validation status")
    reference_valid: bool = Field(..., description="True if all citation references are valid")
    semantic_valid: bool = Field(..., description="True if cited sources semantically support the associated claims")
    total_citations: int = Field(default=0, ge=0, description="Total citation markers found in answer")
    valid_citations: int = Field(default=0, ge=0, description="Number of valid citations resolving to sources")
    invalid_citations: int = Field(default=0, ge=0, description="Number of invalid / unresolvable citations")
    unsupported_claims: int = Field(default=0, ge=0, description="Number of claims failing semantic support")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic validation warnings")
    details: Dict[str, Any] = Field(default_factory=dict, description="Fine-grained claim-level validation details")


class RAGQueryRequest(BaseModel):
    """Validated request schema for citation-aware grounded question answering."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language question to answer with citations",
    )
    candidate_type: Optional[str] = Field(
        default="all",
        description="Optional entity type filter: all, startup, product, research_paper, job, news",
    )
    vector_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum candidates retrieved in initial vector search stage",
    )
    graph_depth: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum traversal depth from vector seed entities (hops)",
    )
    max_graph_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum total candidates surfaced through graph expansion",
    )
    vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Relative weight for vector similarity score in weighted fusion",
    )
    graph_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Relative weight for graph relevance score in weighted fusion",
    )
    ranking_strategy: RankingStrategy = Field(
        default=RankingStrategy.WEIGHTED,
        description="Ranking combination strategy: 'weighted' or 'rrf'",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for retrieved results",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top retrieved context chunks provided for citation generation",
    )
    semantic_validation: bool = Field(
        default=True,
        description="Whether to perform semantic claim-to-evidence support validation",
    )
    validator_type: str = Field(
        default="heuristic",
        description="Validator mode: 'heuristic', 'llm', or 'hybrid'",
    )


class RAGQueryResponse(BaseModel):
    """Structured response container for citation-aware grounded RAG answers."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(..., description="Echo of input query")
    answer: str = Field(..., description="Grounded answer containing [1], [2] citations or abstention")
    citations: List[Citation] = Field(default_factory=list, description="Extracted citations with claim mappings")
    sources: List[SourceChunkProvenance] = Field(default_factory=list, description="Provenance-tracked retrieved sources")
    abstained: bool = Field(default=False, description="True if system abstained due to insufficient evidence")
    validation_status: ValidationStatus = Field(..., description="Overall citation validation status")
    validation_result: Optional[CitationValidationResult] = Field(default=None, description="Detailed validation breakdown")
    retrieval_response: Optional[HybridRetrievalResponse] = Field(default=None, description="Underlying retrieval details")
    execution_ms: float = Field(default=0.0, ge=0.0, description="Total execution duration in milliseconds")
    generation_ms: float = Field(default=0.0, ge=0.0, description="Answer generation duration in milliseconds")
    validation_ms: float = Field(default=0.0, ge=0.0, description="Citation validation duration in milliseconds")


class CitationValidationRequest(BaseModel):
    """Request schema for validating citations against a given answer and sources."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, description="Answer text containing [1], [2] citation markers")
    sources: List[SourceChunkProvenance] = Field(..., description="Retrieved source chunks against which to validate")
    citations: Optional[List[Citation]] = Field(default=None, description="Optional pre-extracted citations")
    semantic_validation: bool = Field(default=True, description="Enable semantic claim entailment checking")
    validator_type: str = Field(default="heuristic", description="Validator mode: 'heuristic', 'llm', or 'hybrid'")


class CitationValidationResponse(BaseModel):
    """Response schema for citation validation endpoint."""

    model_config = ConfigDict(extra="allow")

    validation_status: ValidationStatus = Field(..., description="Overall validation status outcome")
    validation_result: CitationValidationResult = Field(..., description="Structured validation report")
    citations: List[Citation] = Field(default_factory=list, description="Validated citations")
    execution_ms: float = Field(default=0.0, ge=0.0, description="Validation duration in milliseconds")
