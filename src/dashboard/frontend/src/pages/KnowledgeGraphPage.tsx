import React, { useEffect, useState, useMemo } from "react";
import {
  Share2,
  Code,
  Database,
  Check,
  Copy,
  ExternalLink,
  RefreshCw,
  Search,
  ArrowRight,
  GitBranch,
  Layers,
  ZoomIn,
  RotateCcw,
  Sparkles,
} from "lucide-react";

interface GraphNode {
  id: string;
  label: string;
  name: string;
  location?: string;
  funding?: string;
  stage?: string;
  maker?: string;
  pricing?: string;
  company?: string;
  salary?: string;
  authors?: string[];
  journal?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  confidence: number;
}

interface NeighborItem {
  neighbor_id: string;
  relation: string;
  direction: "outgoing" | "incoming";
  confidence: number;
  node?: GraphNode;
}

export const KnowledgeGraphPage: React.FC = () => {
  const [fullGraph, setFullGraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] }>({
    nodes: [],
    edges: [],
  });
  const [subgraph, setSubgraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [cypherScript, setCypherScript] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  // Traversal & Query State
  const [selectedEntityId, setSelectedEntityId] = useState<string>("");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [currentHops, setCurrentHops] = useState<number>(1);
  const [relationFilter, setRelationFilter] = useState<string>("ALL");
  const [neighborsList, setNeighborsList] = useState<NeighborItem[]>([]);
  const [traversalLoading, setTraversalLoading] = useState<boolean>(false);

  // Path Discovery State
  const [pathFrom, setPathFrom] = useState<string>("");
  const [pathTo, setPathTo] = useState<string>("");
  const [pathResult, setPathResult] = useState<{
    found: boolean;
    length: number;
    path: string[];
    nodes: GraphNode[];
    edges: GraphEdge[];
  } | null>(null);
  const [pathLoading, setPathLoading] = useState<boolean>(false);

  const fetchFullGraph = async () => {
    setLoading(true);
    try {
      const [gRes, cRes] = await Promise.all([
        fetch("/api/graph"),
        fetch("/api/graph/export"),
      ]);
      if (gRes.ok) {
        const d = await gRes.json();
        setFullGraph({ nodes: d.nodes || [], edges: d.edges || [] });
      }
      if (cRes.ok) setCypherScript(await cRes.text());
    } catch (e) {
      console.error("Failed to load Knowledge Graph", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFullGraph();
  }, []);

  // Fetch Neighbors or Traversal for Selected Entity
  const inspectEntity = async (entityId: string, hops: number = 1, rel: string = relationFilter) => {
    if (!entityId) return;
    setTraversalLoading(true);
    setSelectedEntityId(entityId);
    setCurrentHops(hops);

    try {
      const relQuery = rel !== "ALL" ? `&relation=${encodeURIComponent(rel)}` : "";

      if (hops === 1) {
        const res = await fetch(`/api/graph/neighbors/${encodeURIComponent(entityId)}?direction=both${relQuery}`);
        if (res.ok) {
          const data = await res.json();
          setSelectedNode(data.node || null);
          setNeighborsList(data.neighbors || []);
          setSubgraph({ nodes: data.nodes || [], edges: data.edges || [] });
        }
      } else {
        const res = await fetch(`/api/graph/traverse/${encodeURIComponent(entityId)}?hops=${hops}${relQuery}`);
        if (res.ok) {
          const data = await res.json();
          setSelectedNode(data.root || null);
          setSubgraph({ nodes: data.nodes || [], edges: data.edges || [] });

          // Also update neighbors list for the root
          const nRes = await fetch(`/api/graph/neighbors/${encodeURIComponent(entityId)}?direction=both${relQuery}`);
          if (nRes.ok) {
            const nData = await nRes.json();
            setNeighborsList(nData.neighbors || []);
          }
        }
      }
    } catch (e) {
      console.error("Failed to traverse graph for entity", e);
    } finally {
      setTraversalLoading(false);
    }
  };

  // Expand +1 hop
  const handleExpandOneHop = () => {
    if (!selectedEntityId) return;
    const nextHops = Math.min(currentHops + 1, 5);
    inspectEntity(selectedEntityId, nextHops);
  };

  // Expand +2 hops
  const handleExpandTwoHops = () => {
    if (!selectedEntityId) return;
    const nextHops = Math.min(currentHops + 2, 5);
    inspectEntity(selectedEntityId, nextHops);
  };

  // Reset Subgraph to Full View
  const handleResetGraph = () => {
    setSubgraph(null);
    setSelectedNode(null);
    setSelectedEntityId("");
    setCurrentHops(1);
    setNeighborsList([]);
  };

  // Discover Path
  const handleFindPath = async () => {
    if (!pathFrom || !pathTo) return;
    setPathLoading(true);
    try {
      const relQuery = relationFilter !== "ALL" ? `&relation=${encodeURIComponent(relationFilter)}` : "";
      const res = await fetch(
        `/api/graph/path?from=${encodeURIComponent(pathFrom)}&to=${encodeURIComponent(pathTo)}${relQuery}`
      );
      if (res.ok) {
        const data = await res.json();
        setPathResult(data);
        if (data.found && data.nodes?.length > 0) {
          setSubgraph({ nodes: data.nodes, edges: data.edges });
        }
      } else {
        setPathResult({ found: false, length: 0, path: [], nodes: [], edges: [] });
      }
    } catch (e) {
      console.error("Failed to discover path", e);
    } finally {
      setPathLoading(false);
    }
  };

  const handleCopyCypher = () => {
    navigator.clipboard.writeText(cypherScript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getLabelColor = (label: string) => {
    switch (label) {
      case "Startup":
        return "bg-[#7ee787]/10 text-[#7ee787] border-[#7ee787]/30";
      case "Product":
        return "bg-[#6e9fe0]/10 text-[#6e9fe0] border-[#6e9fe0]/30";
      case "ResearchPaper":
        return "bg-[#e8b339]/10 text-[#e8b339] border-[#e8b339]/30";
      case "Job":
        return "bg-[#e5534b]/10 text-[#e5534b] border-[#e5534b]/30";
      default:
        return "bg-[#8b8f94]/10 text-[#8b8f94] border-[#8b8f94]/30";
    }
  };

  const activeNodes = subgraph ? subgraph.nodes : fullGraph.nodes;
  const activeEdges = useMemo(() => {
    const edgesList = subgraph ? subgraph.edges : fullGraph.edges;
    if (relationFilter === "ALL") return edgesList;
    return edgesList.filter((e) => e.relation.toUpperCase() === relationFilter.toUpperCase());
  }, [subgraph, fullGraph.edges, relationFilter]);

  // List of distinct relationship types
  const relationshipTypes = useMemo(() => {
    const types = new Set<string>();
    fullGraph.edges.forEach((e) => {
      if (e.relation) types.add(e.relation.toUpperCase());
    });
    return Array.from(types);
  }, [fullGraph.edges]);

  return (
    <div className="space-y-6 select-none">
      {/* Header Bar */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold font-mono text-[#e6e6e6] flex items-center gap-2">
            <Share2 className="w-5 h-5 text-[#7ee787]" />
            KNOWLEDGE GRAPH & QUERY ENGINE
          </h1>
          <p className="text-xs text-[#8b8f94] mt-1 font-mono">
            Direct neighbors, relation filtering, bounded multi-hop traversal, and path discovery
          </p>
        </div>

        <div className="flex items-center gap-3">
          <a
            href="/metrics"
            target="_blank"
            rel="noopener noreferrer"
            className="h-8 px-3 rounded bg-[#0a0b0d] hover:bg-[#1f2124] border border-[#1f2124] text-[#7ee787] font-mono text-xs flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span>Open /metrics</span>
          </a>

          <button
            onClick={fetchFullGraph}
            className="h-8 px-3 rounded bg-[#131417] hover:bg-[#1f2124] border border-[#1f2124] text-[#e6e6e6] font-mono text-xs flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Top Stat Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="p-4 rounded bg-[#131417] border border-[#1f2124]">
          <span className="text-xs text-[#8b8f94] font-mono">Total Graph Nodes</span>
          <div className="text-2xl font-bold text-[#e6e6e6] mt-1 font-mono">
            {fullGraph.nodes.length}
          </div>
        </div>
        <div className="p-4 rounded bg-[#131417] border border-[#1f2124]">
          <span className="text-xs text-[#8b8f94] font-mono">Total Relational Edges</span>
          <div className="text-2xl font-bold text-[#7ee787] mt-1 font-mono">
            {fullGraph.edges.length}
          </div>
        </div>
        <div className="p-4 rounded bg-[#131417] border border-[#1f2124]">
          <span className="text-xs text-[#8b8f94] font-mono">Visible Subgraph Nodes</span>
          <div className="text-2xl font-bold text-[#6e9fe0] mt-1 font-mono">
            {activeNodes.length}
          </div>
        </div>
        <div className="p-4 rounded bg-[#131417] border border-[#1f2124]">
          <span className="text-xs text-[#8b8f94] font-mono">Visible Subgraph Edges</span>
          <div className="text-2xl font-bold text-[#e8b339] mt-1 font-mono">
            {activeEdges.length}
          </div>
        </div>
      </div>

      {/* Traversal & Query Toolbar */}
      <div className="p-4 bg-[#131417] border border-[#1f2124] rounded space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          {/* Entity Selector */}
          <div className="flex items-center gap-2 flex-1">
            <Search className="w-4 h-4 text-[#7ee787]" />
            <span className="text-xs font-mono text-[#8b8f94] whitespace-nowrap">Focus Entity:</span>
            <select
              value={selectedEntityId}
              onChange={(e) => {
                const val = e.target.value;
                if (val) inspectEntity(val, 1);
                else handleResetGraph();
              }}
              className="h-8 px-2.5 bg-[#0a0b0d] border border-[#1f2124] focus:border-[#7ee787] text-[#e6e6e6] rounded font-mono text-xs outline-none cursor-pointer flex-1 max-w-md"
            >
              <option value="">-- SELECT AN ENTITY TO EXPLORE --</option>
              {fullGraph.nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  [{n.label}] {n.name}
                </option>
              ))}
            </select>
          </div>

          {/* Hop Controls */}
          {selectedEntityId && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#6e9fe0]/15 text-[#6e9fe0] border border-[#6e9fe0]/30">
                DEPTH: {currentHops} {currentHops === 1 ? "HOP" : "HOPS"} (MAX 5)
              </span>

              <button
                disabled={traversalLoading}
                onClick={() => inspectEntity(selectedEntityId, 1)}
                className="px-2.5 py-1 bg-[#0a0b0d] hover:bg-[#1f2124] border border-[#1f2124] text-[#e6e6e6] rounded text-xs font-mono flex items-center gap-1 cursor-pointer transition-colors"
              >
                <Layers className="w-3 h-3 text-[#7ee787]" />
                1 Hop
              </button>

              <button
                disabled={traversalLoading || currentHops >= 5}
                onClick={handleExpandOneHop}
                className="px-2.5 py-1 bg-[#7ee787]/15 hover:bg-[#7ee787]/25 text-[#7ee787] border border-[#7ee787]/30 rounded text-xs font-mono font-bold flex items-center gap-1 cursor-pointer transition-colors disabled:opacity-40"
              >
                <ZoomIn className="w-3 h-3" />
                +1 Hop
              </button>

              <button
                disabled={traversalLoading || currentHops >= 4}
                onClick={handleExpandTwoHops}
                className="px-2.5 py-1 bg-[#6e9fe0]/15 hover:bg-[#6e9fe0]/25 text-[#6e9fe0] border border-[#6e9fe0]/30 rounded text-xs font-mono font-bold flex items-center gap-1 cursor-pointer transition-colors disabled:opacity-40"
              >
                <GitBranch className="w-3 h-3" />
                +2 Hops
              </button>

              <button
                onClick={handleResetGraph}
                className="px-2.5 py-1 bg-[#0a0b0d] hover:bg-[#1f2124] text-[#8b8f94] hover:text-[#e6e6e6] border border-[#1f2124] rounded text-xs font-mono flex items-center gap-1 cursor-pointer transition-colors"
              >
                <RotateCcw className="w-3 h-3" />
                Reset
              </button>
            </div>
          )}
        </div>

        {/* Relationship Type Filter Bar */}
        <div className="flex items-center gap-2 flex-wrap pt-2 border-t border-[#1f2124]/60">
          <span className="text-[11px] font-mono text-[#8b8f94]">Relationship Filter:</span>
          <button
            onClick={() => {
              setRelationFilter("ALL");
              if (selectedEntityId) inspectEntity(selectedEntityId, currentHops, "ALL");
            }}
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border transition-colors cursor-pointer ${
              relationFilter === "ALL"
                ? "bg-[#7ee787]/20 text-[#7ee787] border-[#7ee787]/40"
                : "bg-[#0a0b0d] text-[#8b8f94] border-[#1f2124] hover:text-[#e6e6e6]"
            }`}
          >
            ALL RELATIONS ({fullGraph.edges.length})
          </button>
          {relationshipTypes.map((rel) => {
            const count = fullGraph.edges.filter((e) => e.relation.toUpperCase() === rel).length;
            return (
              <button
                key={rel}
                onClick={() => {
                  setRelationFilter(rel);
                  if (selectedEntityId) inspectEntity(selectedEntityId, currentHops, rel);
                }}
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border transition-colors cursor-pointer ${
                  relationFilter === rel
                    ? "bg-[#6e9fe0]/20 text-[#6e9fe0] border-[#6e9fe0]/40"
                    : "bg-[#0a0b0d] text-[#8b8f94] border-[#1f2124] hover:text-[#e6e6e6]"
                }`}
              >
                {rel} ({count})
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Entity Inspector & Neighbors Card */}
      {selectedNode && (
        <div className="p-4 bg-[#131417] border border-[#7ee787]/30 rounded space-y-3">
          <div className="flex items-center justify-between border-b border-[#1f2124] pb-2">
            <div className="flex items-center gap-2">
              <span
                className={`px-2 py-0.5 rounded border text-[10px] uppercase font-bold ${getLabelColor(
                  selectedNode.label
                )}`}
              >
                {selectedNode.label}
              </span>
              <h3 className="text-sm font-bold font-mono text-[#e6e6e6]">{selectedNode.name}</h3>
              <span className="text-[10px] font-mono text-[#8b8f94]">({selectedNode.id})</span>
            </div>
            <span className="text-xs font-mono text-[#7ee787] font-bold">
              {neighborsList.length} Direct {neighborsList.length === 1 ? "Neighbor" : "Neighbors"}
            </span>
          </div>

          {/* Node Attributes */}
          <div className="flex items-center gap-4 text-xs font-mono text-[#8b8f94] flex-wrap">
            {selectedNode.location && (
              <span>Location: <strong className="text-[#e6e6e6]">{selectedNode.location}</strong></span>
            )}
            {selectedNode.funding && (
              <span>Funding: <strong className="text-[#7ee787]">{selectedNode.funding}</strong></span>
            )}
            {selectedNode.stage && (
              <span>Stage: <strong className="text-[#6e9fe0]">{selectedNode.stage}</strong></span>
            )}
            {selectedNode.maker && (
              <span>Maker: <strong className="text-[#e6e6e6]">{selectedNode.maker}</strong></span>
            )}
            {selectedNode.pricing && (
              <span>Pricing: <strong className="text-[#e8b339]">{selectedNode.pricing}</strong></span>
            )}
            {selectedNode.company && (
              <span>Company: <strong className="text-[#e6e6e6]">{selectedNode.company}</strong></span>
            )}
            {selectedNode.salary && (
              <span>Salary: <strong className="text-[#7ee787]">{selectedNode.salary}</strong></span>
            )}
          </div>

          {/* Direct Neighbors Pills */}
          {neighborsList.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#8b8f94]">
                Direct Neighbors (Click to pivot &amp; inspect):
              </span>
              <div className="flex items-center gap-2 flex-wrap">
                {neighborsList.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => inspectEntity(item.neighbor_id, 1)}
                    className="p-1.5 px-2.5 rounded bg-[#0a0b0d] hover:bg-[#1f2124] border border-[#1f2124] flex items-center gap-1.5 text-xs font-mono transition-colors cursor-pointer"
                  >
                    <span className="text-[10px] text-[#7ee787] font-bold">
                      {item.direction === "outgoing" ? "→" : "←"} [{item.relation}]
                    </span>
                    <span className="text-[#e6e6e6] font-medium">
                      {item.node?.name || item.neighbor_id}
                    </span>
                    <span
                      className={`text-[9px] px-1 py-0.2 rounded border font-mono ${getLabelColor(
                        item.node?.label || "Node"
                      )}`}
                    >
                      {item.node?.label || "Node"}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Path Discovery Section */}
      <div className="p-4 bg-[#131417] border border-[#1f2124] rounded space-y-3">
        <div className="flex items-center justify-between border-b border-[#1f2124] pb-2">
          <h2 className="text-xs font-bold font-mono text-[#e6e6e6] flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-[#6e9fe0]" />
            SHORTEST PATH DISCOVERY BETWEEN TWO ENTITIES
          </h2>
        </div>

        <div className="flex flex-col md:flex-row items-center gap-3">
          <div className="flex-1 w-full">
            <label className="text-[10px] font-mono text-[#8b8f94] block mb-1">From Entity:</label>
            <select
              value={pathFrom}
              onChange={(e) => setPathFrom(e.target.value)}
              className="w-full h-8 px-2 bg-[#0a0b0d] border border-[#1f2124] text-[#e6e6e6] rounded font-mono text-xs outline-none"
            >
              <option value="">-- SELECT SOURCE ENTITY --</option>
              {fullGraph.nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  [{n.label}] {n.name}
                </option>
              ))}
            </select>
          </div>

          <ArrowRight className="w-4 h-4 text-[#8b8f94] hidden md:block mt-4" />

          <div className="flex-1 w-full">
            <label className="text-[10px] font-mono text-[#8b8f94] block mb-1">To Entity:</label>
            <select
              value={pathTo}
              onChange={(e) => setPathTo(e.target.value)}
              className="w-full h-8 px-2 bg-[#0a0b0d] border border-[#1f2124] text-[#e6e6e6] rounded font-mono text-xs outline-none"
            >
              <option value="">-- SELECT TARGET ENTITY --</option>
              {fullGraph.nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  [{n.label}] {n.name}
                </option>
              ))}
            </select>
          </div>

          <button
            disabled={!pathFrom || !pathTo || pathLoading}
            onClick={handleFindPath}
            className="w-full md:w-auto mt-4 md:mt-0 h-8 px-4 bg-[#6e9fe0]/15 hover:bg-[#6e9fe0]/25 text-[#6e9fe0] border border-[#6e9fe0]/30 rounded text-xs font-mono font-bold flex items-center justify-center gap-1.5 transition-colors cursor-pointer disabled:opacity-40"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Discover Path</span>
          </button>
        </div>

        {/* Path Result Display */}
        {pathResult && (
          <div className="p-3 bg-[#0a0b0d] border border-[#1f2124] rounded text-xs font-mono">
            {pathResult.found ? (
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-[#7ee787] font-bold">Path Found!</span>
                  <span className="text-[#8b8f94]">({pathResult.length} hops)</span>
                </div>
                <div className="flex items-center gap-2 flex-wrap pt-1">
                  {pathResult.nodes.map((node, i) => (
                    <React.Fragment key={node.id}>
                      <span
                        className={`px-2 py-0.5 rounded border text-[11px] font-bold ${getLabelColor(
                          node.label
                        )}`}
                      >
                        {node.name}
                      </span>
                      {i < pathResult.edges.length && (
                        <span className="text-[#7ee787] font-bold text-[10px]">
                          -[{pathResult.edges[i].relation}]-&gt;
                        </span>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            ) : (
              <span className="text-[#e5534b]">
                No relational path discovered between selected entities within 5 hops.
              </span>
            )}
          </div>
        )}
      </div>

      {/* Main Two-Column View: Graph Triples & Neo4j Cypher Script */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Knowledge Triples List */}
        <div className="p-5 rounded bg-[#131417] border border-[#1f2124] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1f2124] pb-3">
            <h2 className="text-xs font-bold font-mono text-[#e6e6e6] flex items-center gap-2">
              <Database className="w-4 h-4 text-[#6e9fe0]" />
              ACTIVE SUBGRAPH TRIPLES ({activeEdges.length})
            </h2>
            {subgraph && (
              <button
                onClick={handleResetGraph}
                className="text-[10px] font-mono text-[#7ee787] hover:underline cursor-pointer"
              >
                Show All ({fullGraph.edges.length})
              </button>
            )}
          </div>

          <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
            {activeEdges.length > 0 ? (
              activeEdges.map((edge: any, idx: number) => {
                const srcNode = activeNodes.find((n: any) => n.id === edge.source) || fullGraph.nodes.find((n: any) => n.id === edge.source);
                const tgtNode = activeNodes.find((n: any) => n.id === edge.target) || fullGraph.nodes.find((n: any) => n.id === edge.target);
                return (
                  <div
                    key={idx}
                    className="p-3 rounded bg-[#0a0b0d] border border-[#1f2124] flex items-center justify-between text-xs font-mono"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <button
                        onClick={() => srcNode && inspectEntity(srcNode.id, 1)}
                        className={`px-2 py-0.5 rounded border text-[10px] uppercase font-bold hover:opacity-80 transition-opacity cursor-pointer ${getLabelColor(
                          srcNode?.label || "Node"
                        )}`}
                      >
                        {srcNode?.name || edge.source}
                      </button>
                      <span className="text-[#7ee787] font-bold text-[11px]">
                        -[{edge.relation}]-&gt;
                      </span>
                      <button
                        onClick={() => tgtNode && inspectEntity(tgtNode.id, 1)}
                        className={`px-2 py-0.5 rounded border text-[10px] uppercase font-bold hover:opacity-80 transition-opacity cursor-pointer ${getLabelColor(
                          tgtNode?.label || "Node"
                        )}`}
                      >
                        {tgtNode?.name || edge.target}
                      </button>
                    </div>

                    <span className="text-[10px] text-[#8b8f94]">
                      {Math.round((edge.confidence || 0.9) * 100)}% Conf
                    </span>
                  </div>
                );
              })
            ) : (
              <div className="p-8 text-center text-xs text-[#8b8f94] font-mono">
                No relational graph edges in current view. Select an entity or run pipeline to extract triples.
              </div>
            )}
          </div>
        </div>

        {/* Neo4j Cypher Export Viewer */}
        <div className="p-5 rounded bg-[#131417] border border-[#1f2124] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1f2124] pb-3">
            <h2 className="text-xs font-bold font-mono text-[#e6e6e6] flex items-center gap-2">
              <Code className="w-4 h-4 text-[#e8b339]" />
              NEO4J CYPHER EXPORT SCRIPT
            </h2>

            <button
              onClick={handleCopyCypher}
              className="px-3 py-1 rounded bg-[#0a0b0d] hover:bg-[#1f2124] border border-[#1f2124] text-[#7ee787] font-mono text-xs flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied Cypher!" : "Copy Cypher"}</span>
            </button>
          </div>

          <pre className="p-4 rounded bg-[#0a0b0d] border border-[#1f2124] font-mono text-[11px] text-[#7ee787] max-h-[490px] overflow-auto whitespace-pre-wrap leading-relaxed">
            {cypherScript || "// Loading Cypher export script..."}
          </pre>
        </div>
      </div>
    </div>
  );
};
