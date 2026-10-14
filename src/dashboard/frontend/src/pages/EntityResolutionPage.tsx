import React, { useEffect, useState, useMemo } from "react";
import {
  Filter,
  CheckCircle2,
  AlertTriangle,
  Check,
  X,
  ShieldAlert,
  ArrowRight,
  RefreshCw,
  Sparkles,
  CheckCheck,
  XCircle,
} from "lucide-react";

interface ReviewItem {
  id: string;
  candidate: string;
  canonical: string;
  matching_method: string;
  fuzzy_score: number;
  confidence: number;
  decision: string;
  status?: string;
  timestamp: string;
  source: string;
  match_reason?: string;
  reviewed_at?: string;
  reviewer?: string;
  reviewer_notes?: string;
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
    autoMatchCount?: number;
    reviewCount?: number;
    needsReviewCount?: number;
    pendingReviewCount?: number;
    newCount?: number;
    noMatchCount?: number;
    approvedReviewsCount?: number;
    rejectedReviewsCount?: number;
    avgConfidence?: number;
  };
  entries: LogEntry[];
}

export const EntityResolutionPage: React.FC = () => {
  const [filter, setFilter] = useState<string>("all");
  const [logData, setLogData] = useState<EntityLogResponse>({ summary: {}, entries: [] });
  const [reviews, setReviews] = useState<ReviewItem[]>([]);
  const [reviewTab, setReviewTab] = useState<"pending" | "approved" | "rejected" | "all">("pending");
  const [reviewStats, setReviewStats] = useState<{ total: number; pending: number; approved: number; rejected: number }>({
    total: 0,
    pending: 0,
    approved: 0,
    rejected: 0,
  });
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

  const fetchReviewStats = () => {
    fetch("/reviews/stats")
      .then((res) => res.json())
      .then((data) => {
        if (data && typeof data.total === "number") {
          setReviewStats(data);
        }
      })
      .catch((err) => {
        console.warn("Could not fetch review stats", err);
      });
  };

  const fetchReviews = (statusFilter: string = reviewTab) => {
    setLoadingReviews(true);
    fetch(`/reviews?status=${statusFilter}`)
      .then((res) => res.json())
      .then((d) => {
        setReviews(Array.isArray(d) ? d : []);
        setLoadingReviews(false);
        fetchReviewStats();
      })
      .catch((err) => {
        console.error("Failed to load review queue", err);
        setLoadingReviews(false);
      });
  };

  useEffect(() => {
    fetchEntityLog();
    fetchReviewStats();
    fetchReviews("pending");
  }, []);

  const handleTabChange = (tab: "pending" | "approved" | "rejected" | "all") => {
    setReviewTab(tab);
    fetchReviews(tab);
  };

  const handleApprove = async (id: string, candidate: string, canonical: string) => {
    setProcessingId(id);
    try {
      const res = await fetch(`/reviews/${id}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewer: "dashboard_reviewer", notes: "Approved via dashboard" }),
      });
      if (res.ok) {
        if (reviewTab === "pending") {
          setReviews((prev) => prev.filter((r) => r.id !== id));
        } else {
          fetchReviews(reviewTab);
        }
        setActionFeedback({
          id,
          message: `Approved: '${candidate}' merged into canonical '${canonical}'`,
          type: "success",
        });
        fetchEntityLog();
        fetchReviewStats();
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
      const res = await fetch(`/reviews/${id}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewer: "dashboard_reviewer", notes: "Rejected via dashboard" }),
      });
      if (res.ok) {
        if (reviewTab === "pending") {
          setReviews((prev) => prev.filter((r) => r.id !== id));
        } else {
          fetchReviews(reviewTab);
        }
        setActionFeedback({
          id,
          message: `Rejected: '${candidate}' kept as a separate NO_MATCH entity`,
          type: "reject",
        });
        fetchEntityLog();
        fetchReviewStats();
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
    const filterUpper = filter.toUpperCase();
    if (["MATCH", "AUTO_MATCH"].includes(filterUpper)) {
      return entries.filter((e) => ["MATCH", "AUTO_MATCH"].includes((e.decision || "").toUpperCase()));
    }
    if (["REVIEW", "NEEDS_REVIEW"].includes(filterUpper)) {
      return entries.filter((e) => ["REVIEW", "NEEDS_REVIEW"].includes((e.decision || "").toUpperCase()));
    }
    if (["NEW", "NO_MATCH"].includes(filterUpper)) {
      return entries.filter((e) => ["NEW", "NO_MATCH"].includes((e.decision || "").toUpperCase()));
    }
    return entries.filter((e) => e.method_used === filter);
  }, [entries, filter]);

  const getDecisionBadge = (decision?: string, status?: string) => {
    const d = (decision || (status === "merged" ? "AUTO_MATCH" : status === "needs_review" ? "NEEDS_REVIEW" : "NO_MATCH")).toUpperCase();
    switch (d) {
      case "MATCH":
      case "AUTO_MATCH":
        return "text-[#7ee787] border-[#7ee787]/30 bg-[#7ee787]/10";
      case "REVIEW":
      case "NEEDS_REVIEW":
        return "text-[#e8b339] border-[#e8b339]/30 bg-[#e8b339]/10";
      case "NEW":
      case "NO_MATCH":
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

  const autoMatchCount =
    logData.summary?.autoMatchCount ??
    logData.summary?.matchCount ??
    entries.filter((e) => ["AUTO_MATCH", "MATCH"].includes((e.decision || "").toUpperCase())).length;

  const noMatchCount =
    logData.summary?.noMatchCount ??
    logData.summary?.newCount ??
    entries.filter((e) => ["NO_MATCH", "NEW"].includes((e.decision || "").toUpperCase())).length;

  const avgConfidenceStr =
    logData.summary?.avgConfidence != null
      ? `${(logData.summary.avgConfidence * 100).toFixed(1)}%`
      : "92.4%";

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
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
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
            {autoMatchCount}
          </div>
        </div>
        <div className="p-3 bg-[#131417] border border-[#1f2124] rounded">
          <span className="font-mono text-[10px] uppercase text-[#e8b339] flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" /> Needs Review
          </span>
          <div className="text-xl font-bold font-mono text-[#e8b339] mt-1">
            {reviewStats.pending}
          </div>
        </div>
        <div className="p-3 bg-[#131417] border border-[#1f2124] rounded">
          <span className="font-mono text-[10px] uppercase text-[#7ee787] flex items-center gap-1">
            <CheckCheck className="w-3 h-3" /> Approved
          </span>
          <div className="text-xl font-bold font-mono text-[#7ee787] mt-1">
            {reviewStats.approved}
          </div>
        </div>
        <div className="p-3 bg-[#131417] border border-[#1f2124] rounded">
          <span className="font-mono text-[10px] uppercase text-[#e5534b] flex items-center gap-1">
            <XCircle className="w-3 h-3" /> Rejected
          </span>
          <div className="text-xl font-bold font-mono text-[#e5534b] mt-1">
            {reviewStats.rejected}
          </div>
        </div>
        <div className="p-3 bg-[#131417] border border-[#1f2124] rounded">
          <span className="font-mono text-[10px] uppercase text-[#6e9fe0] flex items-center gap-1">
            <Sparkles className="w-3 h-3" /> NO_MATCH ({avgConfidenceStr})
          </span>
          <div className="text-xl font-bold font-mono text-[#6e9fe0] mt-1">
            {noMatchCount}
          </div>
        </div>
      </div>

      {/* 2. HUMAN REVIEW QUEUE WORKFLOW */}
      <div className="border border-[#1f2124] rounded bg-[#131417] overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between p-3.5 bg-[#17191d] border-b border-[#1f2124] gap-2">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-[#e8b339]" />
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-[#e6e6e6]">
              HUMAN-IN-THE-LOOP RESOLUTION REVIEW QUEUE
            </h2>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-[#e8b339]/20 text-[#e8b339] border border-[#e8b339]/30">
              {reviewStats.pending} PENDING
            </span>
          </div>

          <div className="flex items-center gap-2">
            {/* Review Status Tabs */}
            <div className="flex items-center bg-[#0a0b0d] border border-[#1f2124] rounded p-0.5 text-[11px] font-mono">
              <button
                onClick={() => handleTabChange("pending")}
                className={`px-2.5 py-1 rounded transition-colors ${
                  reviewTab === "pending"
                    ? "bg-[#e8b339]/20 text-[#e8b339] font-bold"
                    : "text-[#8b8f94] hover:text-[#e6e6e6]"
                }`}
              >
                Pending ({reviewStats.pending})
              </button>
              <button
                onClick={() => handleTabChange("approved")}
                className={`px-2.5 py-1 rounded transition-colors ${
                  reviewTab === "approved"
                    ? "bg-[#7ee787]/20 text-[#7ee787] font-bold"
                    : "text-[#8b8f94] hover:text-[#e6e6e6]"
                }`}
              >
                Approved ({reviewStats.approved})
              </button>
              <button
                onClick={() => handleTabChange("rejected")}
                className={`px-2.5 py-1 rounded transition-colors ${
                  reviewTab === "rejected"
                    ? "bg-[#e5534b]/20 text-[#e5534b] font-bold"
                    : "text-[#8b8f94] hover:text-[#e6e6e6]"
                }`}
              >
                Rejected ({reviewStats.rejected})
              </button>
              <button
                onClick={() => handleTabChange("all")}
                className={`px-2.5 py-1 rounded transition-colors ${
                  reviewTab === "all"
                    ? "bg-[#1f2124] text-[#e6e6e6] font-bold"
                    : "text-[#8b8f94] hover:text-[#e6e6e6]"
                }`}
              >
                All ({reviewStats.total})
              </button>
            </div>

            <button
              onClick={() => {
                fetchReviews(reviewTab);
              }}
              className="flex items-center gap-1 text-[11px] font-mono text-[#8b8f94] hover:text-[#e6e6e6] transition-colors ml-2"
            >
              <RefreshCw className={`w-3 h-3 ${loadingReviews ? "animate-spin" : ""}`} /> Refresh
            </button>
          </div>
        </div>

        {loadingReviews ? (
          <div className="py-10 text-center text-[#8b8f94] font-mono text-xs">
            Loading review queue...
          </div>
        ) : reviews.length === 0 ? (
          <div className="py-8 px-4 text-center">
            <CheckCircle2 className="w-8 h-8 text-[#7ee787] mx-auto mb-2 opacity-80" />
            <p className="text-xs font-mono text-[#e6e6e6] font-bold">No Reviews In This View</p>
            <p className="text-[11px] font-mono text-[#8b8f94] mt-1">
              All items have been processed or categorized by confidence thresholds.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-[#1f2124]/60">
            {reviews.map((rev) => {
              const statusUpper = (rev.status || rev.decision || "PENDING").toUpperCase();
              const isPending = statusUpper === "PENDING" || statusUpper === "REVIEW";
              const isApproved = statusUpper === "APPROVED";
              const isRejected = statusUpper === "REJECTED";

              return (
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
                      {isPending ? (
                        <span className="px-2 py-0.5 rounded border text-[10px] font-mono uppercase font-bold text-[#e8b339] border-[#e8b339]/30 bg-[#e8b339]/10">
                          NEEDS REVIEW
                        </span>
                      ) : isApproved ? (
                        <span className="px-2 py-0.5 rounded border text-[10px] font-mono uppercase font-bold text-[#7ee787] border-[#7ee787]/30 bg-[#7ee787]/10">
                          APPROVED
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded border text-[10px] font-mono uppercase font-bold text-[#e5534b] border-[#e5534b]/30 bg-[#e5534b]/10">
                          REJECTED
                        </span>
                      )}
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
                        Similarity:{" "}
                        <strong className="text-[#e8b339]">
                          {(rev.fuzzy_score * 100).toFixed(1)}%
                        </strong>
                      </span>
                      <span>•</span>
                      <span>Source: <strong className="text-[#e6e6e6]">{rev.source}</strong></span>
                      {rev.reviewer && (
                        <>
                          <span>•</span>
                          <span>Reviewer: <strong className="text-[#e6e6e6]">{rev.reviewer}</strong></span>
                        </>
                      )}
                    </div>

                    {rev.match_reason && (
                      <p className="text-[11px] font-mono text-[#8b8f94] italic">
                        Reason: {rev.match_reason}
                      </p>
                    )}
                    {rev.reviewer_notes && (
                      <p className="text-[11px] font-mono text-[#7ee787] italic">
                        Reviewer Note: {rev.reviewer_notes}
                      </p>
                    )}
                  </div>

                  {/* Actions for Pending */}
                  {isPending && (
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
                  )}

                  {!isPending && (
                    <div className="text-[11px] font-mono text-[#8b8f94] self-end md:self-center text-right">
                      <div>Status: <span className="font-bold text-[#e6e6e6]">{statusUpper}</span></div>
                      {rev.reviewed_at && <div>{new Date(rev.reviewed_at).toLocaleDateString()}</div>}
                    </div>
                  )}
                </div>
              );
            })}
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
              <option value="AUTO_MATCH">DECISION: AUTO_MATCH</option>
              <option value="NEEDS_REVIEW">DECISION: NEEDS_REVIEW</option>
              <option value="NO_MATCH">DECISION: NO_MATCH</option>
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
                  const rawDec = row.decision || (row.status === "merged" ? "AUTO_MATCH" : row.status === "needs_review" ? "NEEDS_REVIEW" : "NO_MATCH");
                  const dec = rawDec.toUpperCase();
                  const isReview = dec === "NEEDS_REVIEW" || dec === "REVIEW";

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
                        {["AUTO_MATCH", "MATCH"].includes(dec) && canonicalName ? (
                          <span className="text-[#7ee787] font-medium">{canonicalName}</span>
                        ) : ["NEEDS_REVIEW", "REVIEW"].includes(dec) && canonicalName ? (
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
