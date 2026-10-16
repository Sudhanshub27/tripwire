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
  BookOpen,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ExternalLink,
  FileText,
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

interface Citation {
  citation_id: number;
  source_id: string;
  chunk_id?: string | null;
  document_id?: string | null;
  claim_text?: string | null;
  text_span?: string | null;
  reference_valid: boolean;
  claim_supported?: boolean | null;
}

interface SourceChunkProvenance {
  index: number;
  source_id: string;
  document_id: string;
  chunk_id: string;
  title: string;
  source_url?: string | null;
  page_number?: number | null;
  section?: string | null;
  text: string;
  retrieval_method: string;
  retrieval_score: number;
  candidate_type?: string | null;
  metadata: Record<string, any>;
}

interface CitationValidationResult {
  status: "VALID" | "INVALID_REFERENCE" | "UNSUPPORTED_CLAIM" | "INSUFFICIENT_EVIDENCE" | "VALIDATION_ERROR";
  reference_valid: boolean;
  semantic_valid: boolean;
  total_citations: number;
  valid_citations: number;
  invalid_citations: number;
  unsupported_claims: number;
  warnings: string[];
  details: Record<string, any>;
}

interface RAGQueryResponse {
  query: string;
  answer: string;
  citations: Citation[];
  sources: SourceChunkProvenance[];
  abstained: boolean;
  validation_status: "VALID" | "INVALID_REFERENCE" | "UNSUPPORTED_CLAIM" | "INSUFFICIENT_EVIDENCE" | "VALIDATION_ERROR";
  validation_result?: CitationValidationResult;
  execution_ms: number;
  generation_ms: number;
  validation_ms: number;
}

export const VectorSearchPage: React.FC = () => {
  const [query, setQuery] = useState("What models and funding does Anthropic have?");
  const [filterType, setFilterType] = useState("all");
  const [searchMode, setSearchMode] = useState<"rag" | "hybrid" | "vector">("rag");
  const [graphDepth, setGraphDepth] = useState<number>(2);
  const [vectorWeight, setVectorWeight] = useState<number>(0.6);
  const [rankingStrategy, setRankingStrategy] = useState<"weighted" | "rrf">("weighted");

  // Hybrid & Vector state
  const [results, setResults] = useState<RetrievalCandidate[]>([]);
  const [searchStats, setSearchStats] = useState<HybridSearchStats | null>(null);
  const [selectedResult, setSelectedResult] = useState<RetrievalCandidate | null>(null);

  // RAG state
  const [ragResponse, setRagResponse] = useState<RAGQueryResponse | null>(null);
  const [highlightedSourceIndex, setHighlightedSourceIndex] = useState<number | null>(null);

  const [loading, setLoading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [reindexStatus, setReindexStatus] = useState<string | null>(null);
  const [showConfig, setShowConfig] = useState(false);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      if (searchMode === "rag") {
        const payload = {
          query: query.trim(),
          candidate_type: filterType,
          vector_limit: 10,
          graph_depth: graphDepth,
          max_graph_candidates: 20,
          vector_weight: vectorWeight,
          graph_weight: Math.round((1 - vectorWeight) * 10) / 10,
          ranking_strategy: rankingStrategy,
          limit: 5,
          semantic_validation: true,
          validator_type: "heuristic",
        };

        const res = await fetch("/api/rag/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          const data: RAGQueryResponse = await res.json();
          setRagResponse(data);
        }
      } else if (searchMode === "hybrid") {
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
      console.error("Query failed", err);
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

  const getTypeBadgeColor = (type?: string | null) => {
    switch (type?.toLowerCase()) {
      case "startup": return "bg-[#7ee787]/15 text-[#7ee787] border-[#7ee787]/30";
      case "product": return "bg-[#d2a8ff]/15 text-[#d2a8ff] border-[#d2a8ff]/30";
      case "research_paper": return "bg-[#79c0ff]/15 text-[#79c0ff] border-[#79c0ff]/30";
      case "job": return "bg-[#e3b341]/15 text-[#e3b341] border-[#e3b341]/30";
      case "news": return "bg-[#f0883e]/15 text-[#f0883e] border-[#f0883e]/30";
      default: return "bg-[#8b8f94]/15 text-[#8b8f94] border-[#8b8f94]/30";
    }
  };

  const getValidationBadge = (status: string) => {
    switch (status) {
      case "VALID":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-[#7ee787]/15 text-[#7ee787] border border-[#7ee787]/30 shadow-sm">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#7ee787]" /> VALID CITATIONS
          </span>
        );
      case "INSUFFICIENT_EVIDENCE":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-[#e3b341]/15 text-[#e3b341] border border-[#e3b341]/30 shadow-sm">
            <AlertTriangle className="w-3.5 h-3.5 text-[#e3b341]" /> INSUFFICIENT EVIDENCE
          </span>
        );
      case "UNSUPPORTED_CLAIM":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-[#f85149]/15 text-[#f85149] border border-[#f85149]/30 shadow-sm">
            <XCircle className="w-3.5 h-3.5 text-[#f85149]" /> UNSUPPORTED CLAIM
          </span>
        );
      case "INVALID_REFERENCE":
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-[#f85149]/15 text-[#f85149] border border-[#f85149]/30 shadow-sm">
            <XCircle className="w-3.5 h-3.5 text-[#f85149]" /> INVALID REFERENCE
          </span>
        );
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

  const renderAnswerWithCitations = (answerText: string) => {
    // Split on citation markers like [1], [2], [1, 2]
    const parts = answerText.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
    return parts.map((part, i) => {
      const match = part.match(/\[(\d+(?:\s*,\s*\d+)*)\]/);
      if (match) {
        const nums = match[1].split(",").map((n) => parseInt(n.trim(), 10));
        return (
          <span key={i} className="inline-flex items-center gap-0.5 mx-0.5">
            {nums.map((num) => (
              <button
                key={num}
                type="button"
                onClick={() => setHighlightedSourceIndex(num === highlightedSourceIndex ? null : num)}
                className={`px-1.5 py-0.2 rounded font-mono text-[11px] font-bold transition-all cursor-pointer ${
                  highlightedSourceIndex === num
                    ? "bg-[#7ee787] text-[#0a0b0d] ring-2 ring-[#7ee787]/50"
                    : "bg-[#7ee787]/20 text-[#7ee787] hover:bg-[#7ee787] hover:text-[#0a0b0d]"
                }`}
                title={`Jump to Source [${num}]`}
              >
                [{num}]
              </button>
            ))}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1f2124] pb-5">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#7ee787]" />
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#e6e6e6]">
              {searchMode === "rag" ? "CITATION-AWARE GROUNDED RAG" : "GRAPH + VECTOR HYBRID RETRIEVAL"}
            </h1>
          </div>
          <p className="text-xs text-[#8b8f94] font-mono mt-1">
            {searchMode === "rag"
              ? "Answers grounded in retrieved evidence with verifiable [1], [2] citations and semantic claim validation"
              : "Combines dense vector similarity with bounded Knowledge Graph multi-hop expansion & deterministic ranking"}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Search Mode Toggle */}
          <div className="flex items-center bg-[#0f1115] p-1 rounded-lg border border-[#272a30]">
            <button
              onClick={() => setSearchMode("rag")}
              className={`px-3 py-1.5 rounded-md font-mono text-xs flex items-center gap-1.5 transition-colors ${
                searchMode === "rag"
                  ? "bg-[#7ee787] text-[#0a0b0d] font-bold shadow-sm"
                  : "text-[#8b8f94] hover:text-[#e6e6e6]"
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" /> Citation RAG
            </button>
            <button
              onClick={() => setSearchMode("hybrid")}
              className={`px-3 py-1.5 rounded-md font-mono text-xs flex items-center gap-1.5 transition-colors ${
                searchMode === "hybrid"
                  ? "bg-[#7ee787] text-[#0a0b0d] font-bold shadow-sm"
                  : "text-[#8b8f94] hover:text-[#e6e6e6]"
              }`}
            >
              <GitBranch className="w-3.5 h-3.5" /> Hybrid Search
            </button>
            <button
              onClick={() => setSearchMode("vector")}
              className={`px-3 py-1.5 rounded-md font-mono text-xs flex items-center gap-1.5 transition-colors ${
                searchMode === "vector"
                  ? "bg-[#79c0ff] text-[#0a0b0d] font-bold shadow-sm"
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
            placeholder={
              searchMode === "rag"
                ? "Ask a question (e.g. 'What models and funding does Anthropic have?')..."
                : "Search across all ingested entities by context, domain, or technology..."
            }
            className="w-full pl-12 pr-44 py-3.5 bg-[#0f1115] border border-[#272a30] rounded-xl text-sm font-mono text-[#e6e6e6] focus:outline-none focus:border-[#7ee787] placeholder-[#5b5f66] shadow-lg transition-colors"
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
              className="px-4 py-2 bg-[#7ee787] hover:bg-[#68d472] text-[#0a0b0d] font-mono font-bold text-xs rounded-lg transition-colors flex items-center gap-1.5"
            >
              {loading ? "Processing..." : searchMode === "rag" ? "Ask RAG" : searchMode === "hybrid" ? "Hybrid Search" : "Vector Search"}
            </button>
          </div>
        </div>

        {/* Configuration Drawer */}
        {showConfig && (
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

      {/* RAG Mode Presentation */}
      {searchMode === "rag" ? (
        <div className="space-y-6">
          {ragResponse && (
            <>
              {/* Answer Card */}
              <div className="bg-[#0f1115] border border-[#272a30] rounded-xl p-6 shadow-xl relative overflow-hidden">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1f2124] pb-4 mb-4">
                  <div className="flex items-center gap-3">
                    <ShieldCheck className="w-5 h-5 text-[#7ee787]" />
                    <span className="text-xs font-mono uppercase text-[#8b8f94] tracking-wider font-bold">
                      Grounded Answer
                    </span>
                    {getValidationBadge(ragResponse.validation_status)}
                    {ragResponse.abstained && (
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#e3b341]/15 text-[#e3b341] border border-[#e3b341]/30">
                        ABSTAINED (INSUFFICIENT EVIDENCE)
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-3 text-xs font-mono text-[#8b8f94]">
                    <span>Total: {ragResponse.execution_ms.toFixed(1)} ms</span>
                    <span className="text-[#4a4d52]">|</span>
                    <span>Gen: {ragResponse.generation_ms.toFixed(1)} ms</span>
                    <span className="text-[#4a4d52]">|</span>
                    <span>Val: {ragResponse.validation_ms.toFixed(1)} ms</span>
                  </div>
                </div>

                {/* Grounded Answer Text with Clickable Citations */}
                <div className="text-sm font-mono text-[#e6e6e6] leading-relaxed py-2">
                  {renderAnswerWithCitations(ragResponse.answer)}
                </div>

                {/* Validation Warnings / Diagnostics */}
                {ragResponse.validation_result?.warnings && ragResponse.validation_result.warnings.length > 0 && (
                  <div className="mt-4 p-3 rounded-lg bg-[#f85149]/10 border border-[#f85149]/30 text-xs font-mono text-[#f85149] space-y-1">
                    <div className="font-bold flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" /> Validation Diagnostics:
                    </div>
                    {ragResponse.validation_result.warnings.map((w, idx) => (
                      <div key={idx} className="pl-5 text-[11px] text-[#f85149]/90">
                        • {w}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Provenance-Backed Retrieved Sources */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-mono font-bold text-[#e6e6e6] flex items-center gap-2">
                    <FileText className="w-4 h-4 text-[#7ee787]" /> Retrieved Supporting Sources ({ragResponse.sources.length})
                  </h3>
                  {highlightedSourceIndex && (
                    <button
                      onClick={() => setHighlightedSourceIndex(null)}
                      className="text-xs font-mono text-[#7ee787] hover:underline"
                    >
                      Clear Highlight [#{highlightedSourceIndex}]
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {ragResponse.sources.map((src) => {
                    const isHighlighted = highlightedSourceIndex === src.index;
                    return (
                      <div
                        key={src.index}
                        id={`source-${src.index}`}
                        className={`bg-[#0f1115] rounded-xl p-5 border transition-all ${
                          isHighlighted
                            ? "border-[#7ee787] ring-2 ring-[#7ee787]/30 shadow-lg"
                            : "border-[#272a30] hover:border-[#7ee787]/40"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            <span className="w-6 h-6 rounded-md bg-[#7ee787]/20 text-[#7ee787] font-mono font-bold text-xs flex items-center justify-center border border-[#7ee787]/30">
                              [{src.index}]
                            </span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase ${getTypeBadgeColor(src.candidate_type)}`}>
                              {src.candidate_type || "entity"}
                            </span>
                          </div>
                          {getSourceBadge(src.retrieval_method)}
                        </div>

                        <h4 className="text-sm font-bold font-mono text-[#e6e6e6] mt-2 mb-1">
                          {src.title}
                        </h4>

                        {/* Provenance IDs */}
                        <div className="space-y-0.5 my-2 text-[11px] font-mono text-[#8b8f94]">
                          <div className="flex items-center gap-2 truncate">
                            <span className="text-[#5b5f66]">Doc ID:</span>
                            <span className="text-[#79c0ff] truncate">{src.document_id}</span>
                          </div>
                          <div className="flex items-center gap-2 truncate">
                            <span className="text-[#5b5f66]">Chunk ID:</span>
                            <span className="text-[#d2a8ff] truncate">{src.chunk_id}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-[#5b5f66]">Page:</span>
                            <span>{src.page_number != null ? src.page_number : "N/A"}</span>
                            <span className="text-[#5b5f66] ml-2">Section:</span>
                            <span>{src.section || "N/A"}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-[#5b5f66]">Source URL:</span>
                            {src.source_url ? (
                              <a
                                href={src.source_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[#7ee787] hover:underline flex items-center gap-1 truncate"
                              >
                                <span className="truncate">{src.source_url}</span>
                                <ExternalLink className="w-3 h-3 flex-shrink-0" />
                              </a>
                            ) : (
                              <span className="text-[#5b5f66] italic">Not provided</span>
                            )}
                          </div>
                        </div>

                        {/* Excerpt */}
                        <div className="p-2.5 rounded bg-[#0a0b0d] border border-[#1f2124] text-xs font-mono text-[#c9d1d9] leading-relaxed line-clamp-4">
                          {src.text}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      ) : (
        /* Hybrid / Vector Mode Presentation */
        <>
          {/* Telemetry Summary */}
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
                        {res.graph_distance != null && res.graph_distance > 0 && (
                          <span className="px-1.5 py-0.5 rounded bg-[#d2a8ff]/10 text-[#d2a8ff] border border-[#d2a8ff]/20">
                            Dist: {res.graph_distance}
                          </span>
                        )}
                      </div>

                      {/* Graph Path Evidence */}
                      {res.path_evidence && res.path_evidence.length > 0 && (
                        <div className="mt-3 p-2 rounded bg-[#0a0b0d] border border-[#1f2124] text-[10px] font-mono text-[#7ee787] truncate">
                          <span className="text-[#8b8f94] mr-1">Path:</span>
                          {res.path_evidence[0]}
                        </div>
                      )}
                    </div>

                    {/* Footer */}
                    <div className="mt-4 pt-3 border-t border-[#1f2124] flex items-center justify-between text-xs font-mono text-[#8b8f94]">
                      <span className="truncate max-w-[200px] text-[11px] text-[#5b5f66]">{res.id}</span>
                      <button
                        onClick={() => setSelectedResult(res)}
                        className="text-[#7ee787] hover:text-[#68d472] font-mono text-xs flex items-center gap-1"
                      >
                        Inspect <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* Result Inspection Modal */}
      {selectedResult && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0f1115] border border-[#272a30] rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#1f2124] pb-4">
              <div className="flex items-center gap-2">
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase ${getTypeBadgeColor(selectedResult.candidate_type)}`}>
                  {selectedResult.candidate_type}
                </span>
                <h3 className="text-lg font-bold font-mono text-[#e6e6e6]">{selectedResult.title}</h3>
              </div>
              <button
                onClick={() => setSelectedResult(null)}
                className="text-[#8b8f94] hover:text-[#e6e6e6] font-mono text-xs px-2 py-1 rounded border border-[#272a30]"
              >
                Close
              </button>
            </div>

            {/* Score & Ranking Explanation */}
            <div className="p-3 rounded-lg bg-[#14171d] border border-[#272a30] space-y-1 font-mono text-xs">
              <div className="text-[#7ee787] font-bold">
                Final Score: {(selectedResult.final_score * 100).toFixed(2)}%
              </div>
              <div className="text-[#8b8f94] text-[11px] leading-relaxed">
                {selectedResult.ranking_explanation}
              </div>
            </div>

            {/* Path Evidence */}
            {selectedResult.path_evidence.length > 0 && (
              <div className="space-y-1.5">
                <h4 className="text-xs font-mono font-bold text-[#8b8f94]">Knowledge Graph Relationship Paths:</h4>
                <div className="space-y-1">
                  {selectedResult.path_evidence.map((p, idx) => (
                    <div key={idx} className="p-2 rounded bg-[#0a0b0d] border border-[#1f2124] text-xs font-mono text-[#7ee787]">
                      {p}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Raw JSON Payload */}
            <div className="space-y-1.5">
              <h4 className="text-xs font-mono font-bold text-[#8b8f94] flex items-center gap-1.5">
                <Code2 className="w-3.5 h-3.5" /> Entity Attributes:
              </h4>
              <pre className="p-3 rounded-lg bg-[#0a0b0d] border border-[#1f2124] text-[11px] font-mono text-[#79c0ff] overflow-x-auto max-h-60">
                {JSON.stringify(selectedResult.payload, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VectorSearchPage;
