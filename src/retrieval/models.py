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
