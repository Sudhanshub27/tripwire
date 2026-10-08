import React, { useEffect, useState, useMemo } from "react";
import {
  Filter,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Check,
  X,
  ShieldAlert,
  ArrowRight,
  RefreshCw,
  Sparkles,
} from "lucide-react";

interface ReviewItem {
  id: string;
  candidate: string;
  canonical: string;
  matching_method: string;
  fuzzy_score: number;
  confidence: number;
  decision: string;
  timestamp: string;
  source: string;
  match_reason?: string;
}

interface LogEntry {
  id: string;
  entity_name: string;
  raw_name: string;
  candidate?: string;
  canonical_name?: string;
  canonical?: string;
  entity_type: string;
  method_used: string;
  confidence_score: number;
  fuzzy_score?: number;
  decision?: string;
  status: string;
  source?: string;
  match_reason?: string;
  timestamp: string;
}

interface EntityLogResponse {
  summary: {
    totalProcessed?: number;
    exactMatchCount?: number;
    exactMatchPct?: number;
    normalizedCount?: number;
    normalizedPct?: number;
    fuzzyCount?: number;
    fuzzyPct?: number;
    unresolvedCount?: number;
    unresolvedPct?: number;
    matchCount?: number;
    reviewCount?: number;
    newCount?: number;
  };
  entries: LogEntry[];
}

export const EntityResolutionPage: React.FC = () => {
  const [filter, setFilter] = useState<string>("all");
  const [logData, setLogData] = useState<EntityLogResponse>({ summary: {}, entries: [] });
  const [reviews, setReviews] = useState<ReviewItem[]>([]);
  const [loadingLog, setLoadingLog] = useState<boolean>(true);
  const [loadingReviews, setLoadingReviews] = useState<boolean>(true);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [actionFeedback, setActionFeedback] = useState<{ id: string; message: string; type: "success" | "reject" } | null>(null);

  const fetchEntityLog = () => {
    fetch("/api/entity-log")
      .then((res) => res.json())
      .then((d) => {
        setLogData(d);
        setLoadingLog(false);
      })
      .catch((err) => {
        console.error("Failed to load entity log", err);
        setLoadingLog(false);
      });
  };

  const fetchReviews = () => {
    fetch("/api/resolution/review")
      .then((res) => res.json())
      .then((d) => {
        setReviews(Array.isArray(d) ? d : []);
        setLoadingReviews(false);
      })
      .catch((err) => {
        console.error("Failed to load review queue", err);
        setLoadingReviews(false);
      });
  };

  useEffect(() => {
    fetchEntityLog();
    fetchReviews();
  }, []);

  const handleApprove = async (id: string, candidate: string, canonical: string) => {
    setProcessingId(id);
    try {
      const res = await fetch(`/api/resolution/review/${id}/approve`, { method: "POST" });
      if (res.ok) {
        setReviews((prev) => prev.filter((r) => r.id !== id));
        setActionFeedback({
          id,
          message: `Approved: '${candidate}' merged into canonical '${canonical}'`,
          type: "success",
        });
        fetchEntityLog();
      }
    } catch (e) {
      console.error("Failed to approve review item", e);
    } finally {
      setProcessingId(null);
    }
  };

  const handleReject = async (id: string, candidate: string) => {
    setProcessingId(id);
    try {
      const res = await fetch(`/api/resolution/review/${id}/reject`, { method: "POST" });
      if (res.ok) {
        setReviews((prev) => prev.filter((r) => r.id !== id));
        setActionFeedback({
          id,
          message: `Rejected: '${candidate}' kept as a separate NEW entity`,
          type: "reject",
        });
        fetchEntityLog();
      }
    } catch (e) {
      console.error("Failed to reject review item", e);
    } finally {
      setProcessingId(null);
    }
  };

  const entries = logData.entries || [];

  const filteredEntries = useMemo(() => {
    if (filter === "all") return entries;
    if (["MATCH", "REVIEW", "NEW"].includes(filter)) {
      return entries.filter((e) => (e.decision || "").toUpperCase() === filter);
    }
    return entries.filter((e) => e.method_used === filter);
  }, [entries, filter]);

  const getDecisionBadge = (decision?: string, status?: string) => {
    const d = (decision || (status === "merged" ? "MATCH" : status === "needs_review" ? "REVIEW" : "NEW")).toUpperCase();
    switch (d) {
      case "MATCH":
        return "text-[#7ee787] border-[#7ee787]/30 bg-[#7ee787]/10";
      case "REVIEW":
        return "text-[#e8b339] border-[#e8b339]/30 bg-[#e8b339]/10";
      case "NEW":
      default:
        return "text-[#6e9fe0] border-[#6e9fe0]/30 bg-[#6e9fe0]/10";
    }
  };

  const getMethodBadge = (method: string) => {
    switch (method) {
      case "exact":
      case "approved_alias":
        return "text-[#7ee787] border-[#7ee787]/30 bg-[#7ee787]/10";
      case "normalized":
        return "text-[#6e9fe0] border-[#6e9fe0]/30 bg-[#6e9fe0]/10";
      case "fuzzy":
        return "text-[#e8b339] border-[#e8b339]/30 bg-[#e8b339]/10";
      case "unresolved":
      default:
        return "text-[#e5534b] border-[#e5534b]/30 bg-[#e5534b]/10";
    }
  };

  return (
    <div className="space-y-6">
      {/* Action Toast Feedback */}
      {actionFeedback && (
        <div
          className={`flex items-center justify-between px-4 py-3 rounded border text-xs font-mono transition-all ${
            actionFeedback.type === "success"
              ? "bg-[#7ee787]/10 border-[#7ee787]/40 text-[#7ee787]"
              : "bg-[#e5534b]/10 border-[#e5534b]/40 text-[#e5534b]"
          }`}
        >
          <div className="flex items-center gap-2">
            {actionFeedback.type === "success" ? (
              <CheckCircle2 className="w-4 h-4 text-[#7ee787]" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-[#e5534b]" />
            )}
            <span>{actionFeedback.message}</span>
          </div>
          <button
            onClick={() => setActionFeedback(null)}
            className="text-xs hover:opacity-75 font-bold uppercase tracking-wider ml-4"
          >
            ✕
          </button>
        </div>
      )}

      {/* 1. Decision Model Stats Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="p-3 bg-[#131417] border border-[#1f2124] rounded">
          <span className="font-mono text-[10px] uppercase text-[#8b8f94]">Total Resolved</span>
          <div className="text-xl font-bold font-mono text-[#e6e6e6] mt-1">
            {logData.summary?.totalProcessed ?? entries.length}
          </div>
        </div>
        <div className="p-3 bg-[#131417] border border-[#1f2124] rounded">
          <span className="font-mono text-[10px] uppercase text-[#7ee787] flex items-center gap-1">
            <Check className="w-3 h-3" /> Auto MATCH (≥85%)
          </span>
          <div className="text-xl font-bold font-mono text-[#7ee787] mt-1">
            {logData.summary?.matchCount ?? entries.filter((e) => e.decision === "MATCH").length}
          </div>
        </div>
        <div className="p-3 bg-[#131417] border border-[#1f2124] rounded">
          <span className="font-mono text-[10px] uppercase text-[#e8b339] flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" /> Review Queue (65-85%)
          </span>
          <div className="text-xl font-bold font-mono text-[#e8b339] mt-1">
            {reviews.length}
          </div>
        </div>
        <div className="p-3 bg-[#131417] border border-[#1f2124] rounded">
          <span className="font-mono text-[10px] uppercase text-[#6e9fe0] flex items-center gap-1">
            <Sparkles className="w-3 h-3" /> Distinct NEW (&lt;65%)
          </span>
          <div className="text-xl font-bold font-mono text-[#6e9fe0] mt-1">
            {logData.summary?.newCount ?? entries.filter((e) => e.decision === "NEW").length}
          </div>
        </div>
      </div>

      {/* 2. PENDING RESOLUTION REVIEW QUEUE */}
      <div className="border border-[#1f2124] rounded bg-[#131417] overflow-hidden">
        <div className="flex items-center justify-between p-3.5 bg-[#17191d] border-b border-[#1f2124]">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-[#e8b339]" />
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-[#e6e6e6]">
              HUMAN-IN-THE-LOOP RESOLUTION REVIEW QUEUE
            </h2>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-[#e8b339]/20 text-[#e8b339] border border-[#e8b339]/30">
              {reviews.length} PENDING
            </span>
          </div>
          <button
            onClick={() => {
              setLoadingReviews(true);
              fetchReviews();
            }}
            className="flex items-center gap-1 text-[11px] font-mono text-[#8b8f94] hover:text-[#e6e6e6] transition-colors"
          >
            <RefreshCw className={`w-3 h-3 ${loadingReviews ? "animate-spin" : ""}`} /> Refresh
          </button>
        </div>

        {loadingReviews ? (
          <div className="py-10 text-center text-[#8b8f94] font-mono text-xs">
            Loading review queue...
          </div>
        ) : reviews.length === 0 ? (
          <div className="py-8 px-4 text-center">
            <CheckCircle2 className="w-8 h-8 text-[#7ee787] mx-auto mb-2 opacity-80" />
            <p className="text-xs font-mono text-[#e6e6e6] font-bold">Review Queue is Clear</p>
            <p className="text-[11px] font-mono text-[#8b8f94] mt-1">
              All entities have been cleanly categorized into MATCH or NEW based on empirical thresholds.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-[#1f2124]/60">
            {reviews.map((rev) => (
              <div
                key={rev.id}
                className="p-4 hover:bg-[#191b1f] transition-colors flex flex-col md:flex-row md:items-center justify-between gap-4"
              >
                <div className="space-y-1.5 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-mono text-[#8b8f94]">{rev.id}</span>
                    <span className="text-sm font-semibold text-[#e6e6e6] bg-[#0a0b0d] px-2 py-0.5 rounded border border-[#1f2124]">
                      {rev.candidate}
                    </span>
                    <ArrowRight className="w-3.5 h-3.5 text-[#8b8f94]" />
                    <span className="text-sm font-semibold text-[#7ee787] bg-[#7ee787]/10 px-2 py-0.5 rounded border border-[#7ee787]/30">
                      {rev.canonical}
                    </span>
                    <span className="px-2 py-0.5 rounded border text-[10px] font-mono uppercase font-bold text-[#e8b339] border-[#e8b339]/30 bg-[#e8b339]/10">
                      REVIEW REQUIRED
                    </span>
                  </div>

                  <div className="flex items-center gap-3 text-xs font-mono text-[#8b8f94] flex-wrap">
                    <span>Method: <strong className="text-[#e6e6e6]">{rev.matching_method}</strong></span>
                    <span>•</span>
                    <span>
                      Confidence:{" "}
                      <strong className="text-[#e8b339]">
                        {(rev.confidence * 100).toFixed(1)}%
                      </strong>
                    </span>
                    <span>•</span>
                    <span>
                      Fuzzy Score:{" "}
                      <strong className="text-[#e8b339]">
                        {(rev.fuzzy_score * 100).toFixed(1)}%
                      </strong>
                    </span>
                    <span>•</span>
                    <span>Source: <strong className="text-[#e6e6e6]">{rev.source}</strong></span>
                  </div>

                  {rev.match_reason && (
                    <p className="text-[11px] font-mono text-[#8b8f94] italic">
                      Reason: {rev.match_reason}
                    </p>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 self-end md:self-center">
                  <button
                    disabled={processingId === rev.id}
                    onClick={() => handleApprove(rev.id, rev.candidate, rev.canonical)}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-[#7ee787]/15 hover:bg-[#7ee787]/25 text-[#7ee787] border border-[#7ee787]/40 rounded text-xs font-mono font-bold transition-all disabled:opacity-50 cursor-pointer"
                  >
                    <Check className="w-3.5 h-3.5" />
                    Approve (Merge)
                  </button>
                  <button
                    disabled={processingId === rev.id}
                    onClick={() => handleReject(rev.id, rev.candidate)}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-[#e5534b]/15 hover:bg-[#e5534b]/25 text-[#e5534b] border border-[#e5534b]/40 rounded text-xs font-mono font-bold transition-all disabled:opacity-50 cursor-pointer"
                  >
                    <X className="w-3.5 h-3.5" />
                    Reject (Keep New)
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 3. DEDUPLICATION & RESOLUTION AUDIT LOG */}
      <div className="space-y-3">
        <div className="flex items-center justify-between p-3 bg-[#131417] border border-[#1f2124] rounded">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-[#7ee787]" />
            <h2 className="text-xs font-mono font-bold uppercase text-[#e6e6e6]">
              DEDUPLICATION & RESOLUTION AUDIT LOG
            </h2>
          </div>

          {/* Filter Dropdown */}
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-[#8b8f94]">Filter:</span>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="h-8 px-2.5 bg-[#0a0b0d] border border-[#1f2124] focus:border-[#7ee787] text-[#e6e6e6] rounded font-mono text-xs outline-none cursor-pointer"
            >
              <option value="all">ALL ENTRIES ({entries.length})</option>
              <option value="MATCH">DECISION: MATCH</option>
              <option value="REVIEW">DECISION: REVIEW</option>
              <option value="NEW">DECISION: NEW</option>
              <option value="exact">METHOD: EXACT</option>
              <option value="normalized">METHOD: NORMALIZED</option>
              <option value="fuzzy">METHOD: FUZZY</option>
              <option value="unresolved">METHOD: UNRESOLVED</option>
            </select>
          </div>
        </div>

        {/* Audit Log Table */}
        <div className="w-full overflow-x-auto border border-[#1f2124] rounded bg-[#131417]">
          <table className="w-full text-left border-collapse">
            <thead className="sticky top-0 z-10 bg-[#0a0b0d] border-b border-[#1f2124] select-none">
              <tr>
                <th className="px-4 py-2.5 font-mono text-[11px] uppercase text-[#8b8f94]">ID</th>
                <th className="px-4 py-2.5 font-mono text-[11px] uppercase text-[#8b8f94]">Candidate (Raw)</th>
                <th className="px-4 py-2.5 font-mono text-[11px] uppercase text-[#8b8f94]">Matched Canonical</th>
                <th className="px-4 py-2.5 font-mono text-[11px] uppercase text-[#8b8f94]">Decision</th>
                <th className="px-4 py-2.5 font-mono text-[11px] uppercase text-[#8b8f94]">Method</th>
                <th className="px-4 py-2.5 font-mono text-[11px] uppercase text-[#8b8f94] text-right">Confidence</th>
                <th className="px-4 py-2.5 font-mono text-[11px] uppercase text-[#8b8f94]">Source</th>
                <th className="px-4 py-2.5 font-mono text-[11px] uppercase text-[#8b8f94]">Timestamp</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-[#1f2124]/60 text-xs">
              {loadingLog ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-[#8b8f94] font-mono">
                    Loading resolution log...
                  </td>
                </tr>
              ) : filteredEntries.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-[#8b8f94] font-mono">
                    No resolution log entries match filter.
                  </td>
                </tr>
              ) : (
                filteredEntries.map((row) => {
                  const candidateName = row.candidate || row.raw_name;
                  const canonicalName = row.canonical || row.canonical_name || row.entity_name;
                  const dec = row.decision || (row.status === "merged" ? "MATCH" : row.status === "needs_review" ? "REVIEW" : "NEW");
                  const isReview = dec === "REVIEW";

                  return (
                    <tr
                      key={row.id}
                      className={`hover:bg-[#191b1f] transition-colors ${
                        isReview ? "border-l-2 border-[#e8b339] bg-[#e8b339]/5" : ""
                      }`}
                    >
                      <td className="px-4 py-3 font-mono text-[#8b8f94]">{row.id}</td>
                      <td className="px-4 py-3 text-[#e6e6e6] font-medium">{candidateName}</td>
                      <td className="px-4 py-3 text-[#8b8f94] font-mono">
                        {dec === "MATCH" && canonicalName ? (
                          <span className="text-[#7ee787] font-medium">{canonicalName}</span>
                        ) : dec === "REVIEW" && canonicalName ? (
                          <span className="text-[#e8b339] font-medium">{canonicalName} (Suggested)</span>
                        ) : (
                          <span className="text-[#8b8f94] italic">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`px-2 py-0.5 rounded border text-[10px] font-mono uppercase font-bold ${getDecisionBadge(
                            dec,
                            row.status
                          )}`}
                        >
                          {dec}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`px-2 py-0.5 rounded border text-[10px] font-mono uppercase font-bold ${getMethodBadge(
                            row.method_used
                          )}`}
                        >
                          {row.method_used}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono tabular-nums text-right text-[#e6e6e6] font-bold">
                        {(row.confidence_score * 100).toFixed(0)}%
                      </td>
                      <td className="px-4 py-3 font-mono text-[#8b8f94] text-[11px]">
                        {row.source || "pipeline"}
                      </td>
                      <td className="px-4 py-3 font-mono text-[#8b8f94] whitespace-nowrap">
                        {row.timestamp ? row.timestamp.slice(11, 19) : "—"}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
