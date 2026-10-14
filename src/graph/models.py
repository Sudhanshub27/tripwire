"""
================================================================================
GRAPHONE PIPELINE: GRAPH QUERY DATA MODELS & SCHEMAS
================================================================================
Validated Pydantic models for graph nodes, edges, queries, traversals,
path discovery, subgraphs, and structured query responses.
================================================================================
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Direction(str, Enum):
    """Direction of edge traversal relative to the active node."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class QueryType(str, Enum):
    """Supported graph query operation types."""

    NODE_LOOKUP = "node_lookup"
    NEIGHBORS = "neighbors"
    TRAVERSE = "traverse"
    PATH = "path"
    SUBGRAPH = "subgraph"
    RELATIONSHIP = "relationship"


class GraphNode(BaseModel):
    """Graph node representation with stable identifier and typed metadata."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Globally unique stable node identifier")
    label: str = Field(..., description="Entity label type (Startup, Product, ResearchPaper, Job, News)")
    name: str = Field(..., description="Canonical or display entity name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary typed node properties")

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience dictionary-like property access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.properties.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to standard dict representation."""
        base = {
            "id": self.id,
            "label": self.label,
            "name": self.name,
        }
        base.update(self.properties)
        return base


class GraphEdge(BaseModel):
    """Graph directed relationship triple (source -> relation -> target)."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relation: str = Field(..., description="Relationship type (PRODUCES, POSTED_BY, MENTIONS, etc.)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction / linkage confidence [0.0, 1.0]")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary edge metadata")

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience dictionary-like property access."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.properties.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to standard dict representation."""
        base = {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": self.confidence,
        }
        base.update(self.properties)
        return base


class PathStep(BaseModel):
    """Single step along an entity path."""

    from_node: str = Field(..., description="Origin node identifier for this hop")
    relation: str = Field(..., description="Traversed edge relationship type")
    direction: str = Field(..., description="Hop direction: outgoing or incoming")
    to_node: str = Field(..., description="Destination node identifier for this hop")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Edge confidence")


class PathResult(BaseModel):
    """Discovered path between two entities in the graph."""

    found: bool = Field(..., description="True if a connecting path was discovered")
    length: int = Field(default=0, ge=0, description="Number of relationship hops in the path")
    path: List[str] = Field(default_factory=list, description="Cypher-like token sequence representation")
    nodes: List[GraphNode] = Field(default_factory=list, description="Ordered nodes along the path")
    edges: List[GraphEdge] = Field(default_factory=list, description="Ordered edges along the path")
    steps: List[PathStep] = Field(default_factory=list, description="Structured step-by-step path transitions")


# ==============================================================================
# Query Request Models
# ==============================================================================


class NeighborQueryRequest(BaseModel):
    """Parameters for direct neighborhood discovery."""

    relation: Optional[str] = Field(default=None, description="Optional relationship filter (e.g. PRODUCES)")
    node_label: Optional[str] = Field(default=None, description="Optional neighbor node label filter (e.g. Product)")
    direction: Direction = Field(default=Direction.BOTH, description="Direction relative to target node")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Key-value property filters for matching")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum neighbors to return")


class TraverseQueryRequest(BaseModel):
    """Parameters for multi-hop graph traversal."""

    root_id: str = Field(..., description="Seed node identifier or name to traverse from")
    max_depth: int = Field(default=2, ge=1, le=10, description="Maximum traversal depth (hops)")
    relation: Optional[str] = Field(default=None, description="Optional relationship type filter")
    node_label: Optional[str] = Field(default=None, description="Optional node label filter")
    direction: Direction = Field(default=Direction.BOTH, description="Traversal direction: outgoing, incoming, both")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Node property filter constraints")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum total nodes to return")


class PathQueryRequest(BaseModel):
    """Parameters for shortest path discovery between two nodes."""

    from_node: str = Field(..., description="Source node identifier or name")
    to_node: str = Field(..., description="Destination node identifier or name")
    max_depth: int = Field(default=5, ge=1, le=10, description="Maximum search depth for path discovery")
    relation: Optional[str] = Field(default=None, description="Optional relationship type constraint")
    direction: Direction = Field(default=Direction.BOTH, description="Search direction")


class SubgraphQueryRequest(BaseModel):
    """Parameters for neighborhood subgraph extraction."""

    node_ids: List[str] = Field(..., min_length=1, description="List of seed node identifiers or names")
    hops: int = Field(default=1, ge=0, le=5, description="Expansion hops around seed nodes (0 = seed nodes only)")
    relation: Optional[str] = Field(default=None, description="Optional relationship type constraint")
    node_label: Optional[str] = Field(default=None, description="Optional node label constraint")
    direction: Direction = Field(default=Direction.BOTH, description="Expansion direction")
    limit: int = Field(default=200, ge=1, le=1000, description="Maximum subgraph nodes")


class RelationshipQueryRequest(BaseModel):
    """Parameters for global relationship / triple pattern search."""

    relation: Optional[str] = Field(default=None, description="Relationship type filter (e.g. PRODUCES)")
    source_label: Optional[str] = Field(default=None, description="Source node label constraint")
    target_label: Optional[str] = Field(default=None, description="Target node label constraint")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum relationships to return")


# ==============================================================================
# Query Response Models
# ==============================================================================


class QueryResult(BaseModel):
    """Structured, validated graph query response container."""

    query_type: QueryType = Field(..., description="Executed query operation type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Echo of input query parameters")
    nodes: List[GraphNode] = Field(default_factory=list, description="Discovered or matched nodes (deduplicated)")
    edges: List[GraphEdge] = Field(default_factory=list, description="Connecting or traversed edges (deduplicated)")
    paths: Optional[List[PathResult]] = Field(default=None, description="Path discovery results if applicable")
    count: int = Field(default=0, ge=0, description="Primary result entity count")
    nodes_count: int = Field(default=0, ge=0, description="Total matched node count")
    edges_count: int = Field(default=0, ge=0, description="Total matched edge count")
    visited_nodes_count: int = Field(default=0, ge=0, description="Count of distinct nodes explored during traversal")
    execution_ms: float = Field(default=0.0, ge=0.0, description="Query execution duration in milliseconds")
    is_truncated: bool = Field(default=False, description="True if results were capped by a limit constraint")
    message: Optional[str] = Field(default=None, description="Optional contextual message or warning")
