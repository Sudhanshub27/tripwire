import React, { useState } from "react";
import {
  Search,
  Sparkles,
  RefreshCw,
  Filter,
  Database,
  Code2,
  Share2,
  ArrowRight,
  GitBranch,
  Layers,
} from "lucide-react";

interface RetrievalCandidate {
  id: string;
  candidate_type: string;
  title: string;
  text?: string;
  vector_score?: number | null;
  graph_score?: number | null;
  graph_distance?: number | null;
  path_evidence: string[];
  sources: string[];
  source: "VECTOR" | "GRAPH" | "BOTH";
  final_score: number;
  ranking_explanation: string;
  payload: Record<string, any>;
}

interface HybridSearchStats {
  execution_ms: number;
  vector_candidates_count: number;
  graph_candidates_count: number;
  merged_candidates_count: number;
  returned_count: number;
  ranking_strategy: string;
  is_truncated: boolean;
}

export const VectorSearchPage: React.FC = () => {
  const [query, setQuery] = useState("autonomous AI data pipeline");
  const [filterType, setFilterType] = useState("all");
  const [searchMode, setSearchMode] = useState<"hybrid" | "vector">("hybrid");
  const [graphDepth, setGraphDepth] = useState<number>(2);
  const [vectorWeight, setVectorWeight] = useState<number>(0.6);
  const [rankingStrategy, setRankingStrategy] = useState<"weighted" | "rrf">("weighted");

  const [results, setResults] = useState<RetrievalCandidate[]>([]);
  const [searchStats, setSearchStats] = useState<HybridSearchStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [reindexStatus, setReindexStatus] = useState<string | null>(null);
  const [selectedResult, setSelectedResult] = useState<RetrievalCandidate | null>(null);
  const [showConfig, setShowConfig] = useState(false);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      if (searchMode === "hybrid") {
        const payload = {
          query: query.trim(),
          candidate_type: filterType,
          vector_limit: 10,
          graph_depth: graphDepth,
          max_graph_candidates: 20,
          vector_weight: vectorWeight,
          graph_weight: Math.round((1 - vectorWeight) * 10) / 10,
          ranking_strategy: rankingStrategy,
          limit: 12,
        };

        const res = await fetch("/api/retrieval/hybrid", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          const data = await res.json();
          setResults(data.results || []);
          setSearchStats({
            execution_ms: data.execution_ms,
            vector_candidates_count: data.vector_candidates_count,
            graph_candidates_count: data.graph_candidates_count,
            merged_candidates_count: data.merged_candidates_count,
            returned_count: data.returned_count,
            ranking_strategy: data.ranking_strategy,
            is_truncated: data.is_truncated,
          });
        }
      } else {
        // Fallback to Vector-Only baseline endpoint
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&type=${filterType}&limit=12`);
        if (res.ok) {
          const data = await res.json();
          const mapped: RetrievalCandidate[] = (data.results || []).map((r: any) => ({
            id: r.id,
            candidate_type: r.record_type,
            title: r.title,
            text: r.payload?.description || r.payload?.abstract,
            vector_score: r.similarity_score,
            graph_score: null,
            graph_distance: null,
            path_evidence: [],
            sources: ["VECTOR"],
            source: "VECTOR",
            final_score: r.similarity_score,
            ranking_explanation: `Vector similarity: ${(r.similarity_score * 100).toFixed(1)}%`,
            payload: r.payload || {},
          }));
          setResults(mapped);
          setSearchStats({
            execution_ms: 0,
            vector_candidates_count: mapped.length,
            graph_candidates_count: 0,
            merged_candidates_count: mapped.length,
            returned_count: mapped.length,
            ranking_strategy: "vector_only",
            is_truncated: false,
          });
        }
      }
    } catch (err) {
      console.error("Retrieval query failed", err);
    } finally {
      setLoading(false);
    }
  };

  const handleReindex = async () => {
    setReindexing(true);
    setReindexStatus(null);
    try {
      const res = await fetch("/api/search/reindex", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setReindexStatus(`Indexed ${data.indexedCount} entities into LanceDB (${data.vectorDimension}D)`);
        handleSearch();
      }
    } catch (err) {
      setReindexStatus("Reindexing failed");
    } finally {
      setReindexing(false);
    }
  };

  React.useEffect(() => {
    handleSearch();
  }, [filterType, searchMode, graphDepth, vectorWeight, rankingStrategy]);

  const getTypeBadgeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case "startup": return "bg-[#7ee787]/15 text-[#7ee787] border-[#7ee787]/30";
      case "product": return "bg-[#d2a8ff]/15 text-[#d2a8ff] border-[#d2a8ff]/30";
      case "research_paper": return "bg-[#79c0ff]/15 text-[#79c0ff] border-[#79c0ff]/30";
      case "job": return "bg-[#e3b341]/15 text-[#e3b341] border-[#e3b341]/30";
      case "news": return "bg-[#f0883e]/15 text-[#f0883e] border-[#f0883e]/30";
      default: return "bg-[#8b8f94]/15 text-[#8b8f94] border-[#8b8f94]/30";
    }
  };

  const getSourceBadge = (source: string) => {
    switch (source) {
      case "BOTH":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-gradient-to-r from-[#d2a8ff]/20 to-[#7ee787]/20 text-[#d2a8ff] border border-[#d2a8ff]/40 shadow-sm">
            <Sparkles className="w-3 h-3 text-[#7ee787]" /> VECTOR + GRAPH
          </span>
        );
      case "GRAPH":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#7ee787]/15 text-[#7ee787] border border-[#7ee787]/30">
            <Share2 className="w-3 h-3 text-[#7ee787]" /> GRAPH ONLY
          </span>
        );
      case "VECTOR":
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#79c0ff]/15 text-[#79c0ff] border border-[#79c0ff]/30">
            <Database className="w-3 h-3 text-[#79c0ff]" /> VECTOR ONLY
          </span>
        );
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1f2124] pb-5">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#7ee787]" />
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#e6e6e6]">
              GRAPH + VECTOR HYBRID RETRIEVAL
            </h1>
          </div>
          <p className="text-xs text-[#8b8f94] font-mono mt-1">
            Combines dense vector similarity with bounded Knowledge Graph multi-hop expansion & deterministic ranking
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Search Mode Toggle */}
          <div className="flex items-center bg-[#0f1115] p-1 rounded-lg border border-[#272a30]">
            <button
              onClick={() => setSearchMode("hybrid")}
              className={`px-3 py-1.5 rounded-md font-mono text-xs flex items-center gap-1.5 transition-colors ${
                searchMode === "hybrid"
                  ? "bg-[#7ee787] text-[#0a0b0d] font-bold"
                  : "text-[#8b8f94] hover:text-[#e6e6e6]"
              }`}
            >
              <GitBranch className="w-3.5 h-3.5" /> Hybrid
            </button>
            <button
              onClick={() => setSearchMode("vector")}
              className={`px-3 py-1.5 rounded-md font-mono text-xs flex items-center gap-1.5 transition-colors ${
                searchMode === "vector"
                  ? "bg-[#79c0ff] text-[#0a0b0d] font-bold"
                  : "text-[#8b8f94] hover:text-[#e6e6e6]"
              }`}
            >
              <Database className="w-3.5 h-3.5" /> Vector Only
            </button>
          </div>

          <button
            onClick={handleReindex}
            disabled={reindexing}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-[#14171d] border border-[#272a30] hover:border-[#7ee787]/50 text-xs font-mono text-[#e6e6e6] transition-all disabled:opacity-50"
            title="Rebuild LanceDB and Knowledge Graph Indices"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-[#7ee787] ${reindexing ? "animate-spin" : ""}`} />
            {reindexing ? "Reindexing..." : "Reindex"}
          </button>
        </div>
      </div>

      {reindexStatus && (
        <div className="px-4 py-2.5 rounded-lg bg-[#7ee787]/10 border border-[#7ee787]/30 text-xs font-mono text-[#7ee787]">
          ✓ {reindexStatus}
        </div>
      )}

      {/* Main Search Bar */}
      <form onSubmit={handleSearch} className="space-y-4">
        <div className="relative flex items-center">
          <Search className="absolute left-4 w-5 h-5 text-[#8b8f94]" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search across all ingested entities by context, domain, or technology..."
            className="w-full pl-12 pr-40 py-3.5 bg-[#0f1115] border border-[#272a30] rounded-xl text-sm font-mono text-[#e6e6e6] focus:outline-none focus:border-[#7ee787] placeholder-[#5b5f66] shadow-lg transition-colors"
          />
          <div className="absolute right-2 flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowConfig(!showConfig)}
              className={`px-3 py-2 border rounded-lg font-mono text-xs transition-colors ${
                showConfig ? "bg-[#1f242d] border-[#7ee787] text-[#7ee787]" : "border-[#272a30] text-[#8b8f94] hover:text-[#e6e6e6]"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-[#7ee787] hover:bg-[#68d472] text-[#0a0b0d] font-mono font-bold text-xs rounded-lg transition-colors"
            >
              {loading ? "Searching..." : searchMode === "hybrid" ? "Hybrid Search" : "Vector Search"}
            </button>
          </div>
        </div>

        {/* Configuration Drawer */}
        {showConfig && searchMode === "hybrid" && (
          <div className="p-4 rounded-xl bg-[#0f1115] border border-[#272a30] grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
            <div>
              <label className="text-[#8b8f94] block mb-1.5">Graph Traversal Depth:</label>
              <div className="flex gap-2">
                {[1, 2, 3].map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setGraphDepth(d)}
                    className={`px-3 py-1 rounded border ${
                      graphDepth === d
                        ? "bg-[#7ee787]/20 border-[#7ee787] text-[#7ee787] font-bold"
                        : "border-[#272a30] text-[#8b8f94] hover:text-[#e6e6e6]"
                    }`}
                  >
                    {d} Hop{d > 1 ? "s" : ""}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-[#8b8f94] block mb-1.5">
                Vector vs Graph Weight: {Math.round(vectorWeight * 100)}% / {Math.round((1 - vectorWeight) * 100)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={vectorWeight}
                onChange={(e) => setVectorWeight(parseFloat(e.target.value))}
                className="w-full accent-[#7ee787]"
              />
            </div>

            <div>
              <label className="text-[#8b8f94] block mb-1.5">Ranking Strategy:</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setRankingStrategy("weighted")}
                  className={`px-3 py-1 rounded border ${
                    rankingStrategy === "weighted"
                      ? "bg-[#7ee787]/20 border-[#7ee787] text-[#7ee787] font-bold"
                      : "border-[#272a30] text-[#8b8f94] hover:text-[#e6e6e6]"
                  }`}
                >
                  Weighted Linear
                </button>
                <button
                  type="button"
                  onClick={() => setRankingStrategy("rrf")}
                  className={`px-3 py-1 rounded border ${
                    rankingStrategy === "rrf"
                      ? "bg-[#7ee787]/20 border-[#7ee787] text-[#7ee787] font-bold"
                      : "border-[#272a30] text-[#8b8f94] hover:text-[#e6e6e6]"
                  }`}
                >
                  RRF (k=60)
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Entity Category Filter Tabs */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-mono text-[#8b8f94] flex items-center gap-1.5 mr-2">
            <Filter className="w-3.5 h-3.5 text-[#4a4d52]" /> Filter Type:
          </span>
          {[
            { id: "all", label: "All Types" },
            { id: "startup", label: "Startups" },
            { id: "product", label: "Products" },
            { id: "research_paper", label: "Papers" },
            { id: "job", label: "Jobs" },
            { id: "news", label: "News" },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setFilterType(tab.id)}
              className={`px-3 py-1.5 rounded-md font-mono text-xs transition-colors ${
                filterType === tab.id
                  ? "bg-[#1f242d] text-[#7ee787] border border-[#7ee787]/40 font-semibold"
                  : "bg-[#0f1115] text-[#8b8f94] border border-[#1f2124] hover:text-[#e6e6e6]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </form>

      {/* Telemetry / Results Count Summary */}
      <div className="flex flex-wrap items-center justify-between text-xs font-mono text-[#8b8f94] bg-[#0f1115] p-3 rounded-lg border border-[#1f2124] gap-2">
        <div className="flex items-center gap-3">
          <span>Found {results.length} ranked result{results.length !== 1 ? "s" : ""}</span>
          {searchStats && searchMode === "hybrid" && (
            <>
              <span className="text-[#4a4d52]">|</span>
              <span className="text-[#79c0ff]">Vector Hits: {searchStats.vector_candidates_count}</span>
              <span className="text-[#4a4d52]">|</span>
              <span className="text-[#7ee787]">Graph Hits: {searchStats.graph_candidates_count}</span>
              <span className="text-[#4a4d52]">|</span>
              <span className="text-[#d2a8ff]">Merged: {searchStats.merged_candidates_count}</span>
            </>
          )}
        </div>

        {searchStats && searchStats.execution_ms > 0 && (
          <div className="text-[#7ee787] flex items-center gap-1.5">
            <span>⚡ {searchStats.execution_ms.toFixed(2)} ms</span>
            <span className="text-[#4a4d52]">({searchStats.ranking_strategy})</span>
          </div>
        )}
      </div>

      {/* Results Grid */}
      {results.length === 0 ? (
        <div className="p-12 text-center border border-dashed border-[#272a30] rounded-xl bg-[#0f1115]">
          <Database className="w-8 h-8 text-[#4a4d52] mx-auto mb-3" />
          <p className="text-sm font-mono text-[#8b8f94]">No candidates found for "{query}".</p>
          <p className="text-xs font-mono text-[#4a4d52] mt-1">Try searching for "Anthropic", "Claude", "Cogna AI", or "attention".</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {results.map((res) => {
            const scoreDisplay = (res.final_score * 100).toFixed(1);
            return (
              <div
                key={res.id}
                className="group relative bg-[#0f1115] border border-[#272a30] hover:border-[#7ee787]/50 rounded-xl p-5 flex flex-col justify-between transition-all hover:shadow-xl"
              >
                <div>
                  {/* Card Header */}
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono border uppercase ${getTypeBadgeColor(res.candidate_type)}`}>
                      {res.candidate_type}
                    </span>
                    {getSourceBadge(res.source)}
                  </div>

                  {/* Title and Score */}
                  <div className="flex items-baseline justify-between gap-2 mt-2">
                    <h3 className="text-sm font-bold text-[#e6e6e6] font-mono leading-snug group-hover:text-[#7ee787] transition-colors">
                      {res.title}
                    </h3>
                    <span className="font-mono text-xs font-bold text-[#7ee787] bg-[#7ee787]/10 px-2 py-0.5 rounded border border-[#7ee787]/20 whitespace-nowrap">
                      {scoreDisplay}%
                    </span>
                  </div>

                  {/* Description / Text */}
                  <p className="text-xs text-[#8b8f94] line-clamp-2 mt-2 font-mono leading-relaxed">
                    {res.text || res.payload?.description || res.payload?.abstract || res.payload?.tagline || "No description provided."}
                  </p>

                  {/* Scores breakdown pills */}
                  <div className="mt-3 flex flex-wrap items-center gap-1.5 text-[10px] font-mono">
                    {res.vector_score != null && (
                      <span className="px-1.5 py-0.5 rounded bg-[#79c0ff]/10 text-[#79c0ff] border border-[#79c0ff]/20">
                        Vector: {(res.vector_score * 100).toFixed(1)}%
                      </span>
                    )}
                    {res.graph_score != null && (
                      <span className="px-1.5 py-0.5 rounded bg-[#7ee787]/10 text-[#7ee787] border border-[#7ee787]/20">
                        Graph: {(res.graph_score * 100).toFixed(1)}%
                      </span>
                    )}
                    {res.graph_distance != null && (
                      <span className="px-1.5 py-0.5 rounded bg-[#d2a8ff]/10 text-[#d2a8ff] border border-[#d2a8ff]/20">
                        Dist: {res.graph_distance} hop{res.graph_distance !== 1 ? "s" : ""}
                      </span>
                    )}
                  </div>

                  {/* Graph Path / Relationship Evidence */}
                  {res.path_evidence && res.path_evidence.length > 0 && (
                    <div className="mt-3 p-2 rounded bg-[#0a0b0d] border border-[#1f2124] text-[10px] font-mono text-[#7ee787] space-y-1">
                      <div className="text-[#8b8f94] flex items-center gap-1">
                        <Share2 className="w-2.5 h-2.5" /> Relationship Evidence:
                      </div>
                      {res.path_evidence.slice(0, 2).map((ev, i) => (
                        <div key={i} className="truncate text-[#e6e6e6]">
                          • {ev}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Ranking Explanation */}
                  <div className="mt-2 text-[10px] font-mono text-[#5b5f66] line-clamp-2">
                    ℹ️ {res.ranking_explanation}
                  </div>
                </div>

                {/* Footer */}
                <div className="mt-4 pt-3 border-t border-[#1f2124] flex items-center justify-between">
                  <span className="text-[10px] font-mono text-[#4a4d52] truncate max-w-[140px]">
                    {res.id}
                  </span>
                  <button
                    onClick={() => setSelectedResult(res)}
                    className="inline-flex items-center gap-1 text-xs font-mono text-[#7ee787] hover:underline"
                  >
                    <Code2 className="w-3.5 h-3.5" /> Payload
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* JSON Payload Modal */}
      {selectedResult && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
          onClick={() => setSelectedResult(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-2xl bg-[#0f1115] border border-[#272a30] rounded-xl p-6 shadow-2xl max-h-[80vh] flex flex-col"
          >
            <div className="flex items-center justify-between pb-3 border-b border-[#272a30]">
              <div className="flex items-center gap-2">
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono border uppercase ${getTypeBadgeColor(selectedResult.candidate_type)}`}>
                  {selectedResult.candidate_type}
                </span>
                {getSourceBadge(selectedResult.source)}
                <h2 className="text-sm font-bold font-mono text-[#e6e6e6] ml-1">{selectedResult.title}</h2>
              </div>
              <button
                onClick={() => setSelectedResult(null)}
                className="text-xs font-mono text-[#8b8f94] hover:text-[#e6e6e6]"
              >
                ✕ Close
              </button>
            </div>

            <div className="mt-3 p-3 rounded-lg bg-[#0a0b0d] border border-[#1f2124] text-xs font-mono space-y-1.5">
              <div className="text-[#8b8f94] font-bold">Ranking Explanation:</div>
              <div className="text-[#7ee787]">{selectedResult.ranking_explanation}</div>
              {selectedResult.path_evidence.length > 0 && (
                <div className="pt-2 border-t border-[#1f2124]">
                  <div className="text-[#8b8f94] font-bold mb-1">Path Provenance:</div>
                  {selectedResult.path_evidence.map((p, i) => (
                    <div key={i} className="text-[#d2a8ff]">• {p}</div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex-1 overflow-y-auto mt-4 p-4 rounded-lg bg-[#0a0b0d] border border-[#1f2124]">
              <pre className="text-xs font-mono text-[#7ee787] whitespace-pre-wrap leading-relaxed">
                {JSON.stringify(selectedResult.payload, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
