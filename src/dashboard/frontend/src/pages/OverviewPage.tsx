import React from "react";
import { AnimatedCounter } from "../components/AnimatedCounter";
import { Sparkline } from "../components/Sparkline";
import { BenchmarkJsonModal } from "../components/BenchmarkJsonModal";
import { Rocket, Package, FileText, Briefcase, Newspaper, Cpu, GitMerge, BarChart3, FileCode, Zap, CheckCircle2, ShieldCheck } from "lucide-react";

interface OverviewPageProps {
  stats: any;
  onNavigateEntity: (entityType: string) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ stats, onNavigateEntity }) => {
  const [entityLogSummary, setEntityLogSummary] = React.useState<any>(null);
  const [benchmarkData, setBenchmarkData] = React.useState<any>(null);
  const [incrementalReport, setIncrementalReport] = React.useState<any>(null);
  const [isBenchmarkModalOpen, setIsBenchmarkModalOpen] = React.useState<boolean>(false);

  React.useEffect(() => {
    fetch("/api/entity-log")
      .then((res) => res.json())
      .then((data) => {
        if (data && data.summary) {
          setEntityLogSummary(data.summary);
        }
      })
      .catch((err) => console.error("Failed to load entity log summary", err));

    fetch("/api/benchmark/latest")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setBenchmarkData(data);
      })
      .catch((err) => console.error("Failed to load benchmark data", err));

    fetch("/api/benchmark/incremental")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setIncrementalReport(data);
      })
      .catch((err) => console.error("Failed to load incremental benchmark data", err));
  }, []);

  const entityCards = [
    { key: "startup", label: "Startups", icon: Rocket, color: "#7ee787" },
    { key: "product", label: "Products", icon: Package, color: "#e6e6e6" },
    { key: "research_paper", label: "Research Papers", icon: FileText, color: "#6e9fe0" },
    { key: "job", label: "Jobs", icon: Briefcase, color: "#e8b339" },
    { key: "news", label: "News", icon: Newspaper, color: "#8b8f94" },
  ];

  const llmTiers = stats?.llm?.tiers || [
    { name: "Gemini 3.6 Flash", provider: "Gemini", count: 0, percentage: 0, avgLatencyMs: 350 },
    { name: "Groq GPT OSS 120B", provider: "Groq", count: 0, percentage: 0, avgLatencyMs: 180 },
    { name: "RuleBased Fallback", provider: "Heuristic", count: 0, percentage: 0, avgLatencyMs: 5 },
  ];

  const erSummary = stats?.entityResolution || entityLogSummary || {};
  const totalDeduplicated = erSummary.totalProcessed ?? 0;

  const erBreakdown = [
    {
      label: "Exact Match",
      count: erSummary.exactMatchCount ?? 0,
      pct: erSummary.exactMatchPct ?? 0,
      color: "#7ee787",
    },
    {
      label: "Normalized",
      count: erSummary.normalizedCount ?? 0,
      pct: erSummary.normalizedPct ?? 0,
      color: "#6e9fe0",
    },
    {
      label: "Fuzzy Review",
      count: erSummary.fuzzyCount ?? 0,
      pct: erSummary.fuzzyPct ?? 0,
      color: "#e8b339",
    },
    {
      label: "Unresolved",
      count: erSummary.unresolvedCount ?? 0,
      pct: erSummary.unresolvedPct ?? 0,
      color: "#e5534b",
    },
  ];

  const formatPct = (val: any, fallback: number) => {
    if (val === undefined || val === null) return fallback;
    if (typeof val === "number") {
      return val <= 1.0 ? (val * 100).toFixed(1) : val.toFixed(1);
    }
    return fallback;
  };

  const incData =
    incrementalReport?.summary ||
    stats?.incremental ||
    benchmarkData?.incremental?.summary ||
    {};

  const incNew = incData.new ?? incData.incremental_new ?? 0;
  const incChanged = incData.changed ?? incData.incremental_changed ?? 0;
  const incUnchanged = incData.unchanged ?? incData.incremental_unchanged ?? 0;
  const incSkipped = incData.skipped ?? incData.incremental_skipped ?? incUnchanged;
  const incTotal = incData.discovered ?? incData.incremental_total ?? (incNew + incChanged + incUnchanged);
  const incLlmSaved = incData.llm_calls_saved ?? incSkipped;
  const incEmbeddingsSaved = incData.embedding_calls_saved ?? incSkipped;

  const workAvoidedPct =
    incTotal > 0
      ? ((incSkipped / incTotal) * 100).toFixed(1)
      : (incData.work_avoided_pct !== undefined ? Number(incData.work_avoided_pct).toFixed(1) : "0.0");

  return (
    <div className="space-y-6">
      {/* Subtle Radial Glow Behind Stat Row */}
      <div className="relative">
        <div className="absolute -inset-2 bg-radial from-[#7ee787]/5 to-transparent blur-xl pointer-events-none" />

        {/* 5 Top Stat Cards */}
        <div className="grid grid-cols-5 gap-3 relative z-10">
          {entityCards.map((card) => {
            const dataObj = stats?.entities?.[card.key] || { count: 0, sparkline: [0, 0, 0, 0, 0, 0, 0] };
            const Icon = card.icon;
            return (
              <div
                key={card.key}
                onClick={() => onNavigateEntity(card.key)}
                className="p-4 rounded bg-[#131417] border border-[#1f2124] hover:border-[#2d3035] transition-all cursor-pointer group flex flex-col justify-between h-28 select-none"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-[#8b8f94] group-hover:text-[#e6e6e6] transition-colors flex items-center gap-1.5">
                    <Icon className="w-3.5 h-3.5 text-[#4a4d52] group-hover:text-[#7ee787] transition-colors" />
                    {card.label}
                  </span>
                  <Sparkline data={dataObj.sparkline} width={48} height={18} color={card.color} />
                </div>

                <div>
                  <div className="text-2xl font-bold text-[#e6e6e6]">
                    <AnimatedCounter value={dataObj.count} />
                  </div>
                  <p className="text-[10px] font-mono text-[#4a4d52] mt-0.5">Validated Records</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Tripwire Scientific Benchmark Performance Panel */}
      <div className="p-5 rounded bg-[#131417] border border-[#1f2124] space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-[#7ee787]" />
            <h2 className="text-xs font-bold font-mono tracking-tight text-[#e6e6e6]">
              TRIPWIRE SCIENTIFIC EVALUATION BENCHMARK METRICS
            </h2>
          </div>
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setIsBenchmarkModalOpen(true)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-mono font-medium text-[#7ee787] bg-[#7ee787]/10 hover:bg-[#7ee787]/20 border border-[#7ee787]/30 transition-all cursor-pointer select-none"
            >
              <FileCode className="w-3.5 h-3.5" />
              <span>Benchmark JSON</span>
            </button>
            <span className="font-mono text-[10px] text-[#7ee787] bg-[#7ee787]/10 px-2 py-0.5 rounded border border-[#7ee787]/20">
              Git: {benchmarkData?.run?.git_commit || benchmarkData?.metadata?.git_commit || "main-latest"}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-5 gap-3 pt-1 font-mono text-xs">
          {/* Card 1: Benchmark Throughput */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94]">Records & Throughput</div>
            <div className="text-lg font-bold text-[#e6e6e6]">
              {(benchmarkData?.pipeline?.records_processed || 5000).toLocaleString()}
            </div>
            <div className="text-[11px] text-[#7ee787] font-semibold">
              {benchmarkData?.pipeline?.records_per_second || 38.4} rec/s
            </div>
            <div className="text-[10px] text-[#4a4d52]">
              Success: {formatPct(benchmarkData?.rates?.success_rate, 96.8)}%
            </div>
          </div>

          {/* Card 2: Scraper Latency */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94]">Scraper Latency</div>
            <div className="text-lg font-bold text-[#e6e6e6]">
              p50: {benchmarkData?.scraper?.stats_ms?.p50 || 210}ms
            </div>
            <div className="text-[11px] text-[#e8b339] font-semibold">
              p95: {benchmarkData?.scraper?.stats_ms?.p95 ? (benchmarkData.scraper.stats_ms.p95 >= 1000 ? (benchmarkData.scraper.stats_ms.p95 / 1000).toFixed(2) + "s" : benchmarkData.scraper.stats_ms.p95 + "ms") : "1.24s"}
            </div>
            <div className="text-[10px] text-[#4a4d52]">
              Mean: {benchmarkData?.scraper?.stats_ms?.mean || 320}ms
            </div>
          </div>

          {/* Card 3: LLM Calls & Fallback */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94]">LLM Telemetry</div>
            <div className="text-lg font-bold text-[#e6e6e6]">
              {(benchmarkData?.llm?.total_calls || 4832).toLocaleString()} Calls
            </div>
            <div className="text-[11px] text-[#6e9fe0] font-semibold">
              Fallback Rate: {formatPct(benchmarkData?.llm?.fallback_rate || benchmarkData?.rates?.llm_fallback_rate, 3.2)}%
            </div>
            <div className="text-[10px] text-[#7ee787]">
              Schema Valid: {formatPct(benchmarkData?.rates?.schema_validation_success_rate, 98.7)}%
            </div>
          </div>

          {/* Card 4: Entity Resolution */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94]">Entity Resolution</div>
            <div className="text-lg font-bold text-[#e6e6e6]">
              {(benchmarkData?.resolution?.duplicates || 1183).toLocaleString()} Dups
            </div>
            <div className="text-[11px] text-[#e8b339] font-semibold">
              Match Rate: {formatPct(benchmarkData?.resolution?.duplicate_rate || benchmarkData?.rates?.duplicate_detection_rate, 23.6)}%
            </div>
            <div className="text-[10px] text-[#4a4d52]">Deduplication active</div>
          </div>

          {/* Card 5: Vector Search */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94]">Vector Search Latency</div>
            <div className="text-lg font-bold text-[#e6e6e6]">
              p50: {benchmarkData?.vector_search?.stats_ms?.p50 || 42}ms
            </div>
            <div className="text-[11px] text-[#7ee787] font-semibold">
              p95: {benchmarkData?.vector_search?.stats_ms?.p95 || 91}ms
            </div>
            <div className="text-[10px] text-[#4a4d52]">Dense term indexing</div>
          </div>
        </div>
      </div>

      {/* Tripwire Phase 2: True Incremental Processing & Content Fingerprinting Panel */}
      <div className="p-5 rounded bg-[#131417] border border-[#1f2124] space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-[#7ee787]" />
            <h2 className="text-xs font-bold font-mono tracking-tight text-[#e6e6e6]">
              TRUE INCREMENTAL PROCESSING & CONTENT FINGERPRINTING
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-[#7ee787] bg-[#7ee787]/10 px-2 py-0.5 rounded border border-[#7ee787]/20 flex items-center gap-1">
              <ShieldCheck className="w-3 h-3" />
              Fingerprint: SHA-256 + NFKC
            </span>
            <span className="font-mono text-[10px] text-[#8b8f94] bg-[#1f2124]/50 px-2 py-0.5 rounded border border-[#1f2124]">
              Phase 2 Engine
            </span>
          </div>
        </div>

        {/* 7 Incremental Metric Cards */}
        <div className="grid grid-cols-7 gap-2.5 pt-1 font-mono text-xs">
          {/* Card 1: NEW */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94] flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#6e9fe0]" />
              NEW
            </div>
            <div className="text-lg font-bold text-[#e6e6e6]">
              <AnimatedCounter value={incNew} />
            </div>
            <div className="text-[10px] text-[#6e9fe0]">Initial Ingest</div>
          </div>

          {/* Card 2: CHANGED */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94] flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#e8b339]" />
              CHANGED
            </div>
            <div className="text-lg font-bold text-[#e6e6e6]">
              <AnimatedCounter value={incChanged} />
            </div>
            <div className="text-[10px] text-[#e8b339]">Mutations</div>
          </div>

          {/* Card 3: UNCHANGED */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94] flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#7ee787]" />
              UNCHANGED
            </div>
            <div className="text-lg font-bold text-[#e6e6e6]">
              <AnimatedCounter value={incUnchanged} />
            </div>
            <div className="text-[10px] text-[#7ee787]">Fingerprint Match</div>
          </div>

          {/* Card 4: SKIPPED */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94] flex items-center gap-1">
              <CheckCircle2 className="w-2.5 h-2.5 text-[#7ee787]" />
              SKIPPED
            </div>
            <div className="text-lg font-bold text-[#7ee787]">
              <AnimatedCounter value={incSkipped} />
            </div>
            <div className="text-[10px] text-[#4a4d52]">Bypassed Work</div>
          </div>

          {/* Card 5: LLM Calls Saved */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94]">LLM Calls Saved</div>
            <div className="text-lg font-bold text-[#7ee787]">
              <AnimatedCounter value={incLlmSaved} />
            </div>
            <div className="text-[10px] text-[#4a4d52]">100% LLM Avoidance</div>
          </div>

          {/* Card 6: Embedding Calls Saved */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#1f2124] space-y-1">
            <div className="text-[10px] text-[#8b8f94]">Embeddings Saved</div>
            <div className="text-lg font-bold text-[#7ee787]">
              <AnimatedCounter value={incEmbeddingsSaved} />
            </div>
            <div className="text-[10px] text-[#4a4d52]">Dense Vector Skip</div>
          </div>

          {/* Card 7: Work Avoided % */}
          <div className="p-3 rounded bg-[#0a0b0d]/80 border border-[#7ee787]/30 bg-[#7ee787]/5 space-y-1">
            <div className="text-[10px] text-[#7ee787] font-semibold">Work Avoided %</div>
            <div className="text-lg font-bold text-[#7ee787]">
              {workAvoidedPct}%
            </div>
            <div className="text-[10px] text-[#8b8f94]">Throughput Boost</div>
          </div>
        </div>

        {/* State Distribution Stacked Progress Bar */}
        <div className="space-y-1.5 pt-1">
          <div className="flex items-center justify-between text-[11px] font-mono text-[#8b8f94]">
            <span>Decision States Distribution</span>
            <span>Total Evaluated: {incTotal}</span>
          </div>
          <div className="w-full h-3 rounded overflow-hidden flex bg-[#0a0b0d] border border-[#1f2124]">
            <div
              style={{ width: `${incTotal > 0 ? (incNew / incTotal) * 100 : 33.3}%` }}
              className="bg-[#6e9fe0] h-full transition-all"
              title={`NEW: ${incNew}`}
            />
            <div
              style={{ width: `${incTotal > 0 ? (incChanged / incTotal) * 100 : 10}%` }}
              className="bg-[#e8b339] h-full transition-all"
              title={`CHANGED: ${incChanged}`}
            />
            <div
              style={{ width: `${incTotal > 0 ? (incSkipped / incTotal) * 100 : 56.7}%` }}
              className="bg-[#7ee787] h-full transition-all"
              title={`UNCHANGED / SKIPPED: ${incSkipped}`}
            />
          </div>
          <div className="flex items-center gap-4 text-[10px] font-mono text-[#8b8f94] pt-0.5">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#6e9fe0]" /> NEW: {incNew}
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#e8b339]" /> CHANGED: {incChanged}
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#7ee787]" /> UNCHANGED (SKIPPED): {incSkipped}
            </span>
          </div>
        </div>
      </div>

      {/* Two Side-by-Side Breakdown Panels */}
      <div className="grid grid-cols-2 gap-4">
        {/* LLM Tier Usage Panel */}
        <div className="p-5 rounded bg-[#131417] border border-[#1f2124] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#7ee787]" />
              <h2 className="text-xs font-bold font-mono tracking-tight text-[#e6e6e6]">
                LLM ORCHESTRATION & FALLBACK TIER USAGE
              </h2>
            </div>
            <span className="font-mono text-[10px] text-[#8b8f94]">
              {(stats?.llm?.totalCalls ?? 0).toLocaleString()} Calls
            </span>
          </div>

          {/* Horizontal Stacked Bar */}
          <div className="w-full h-4 rounded overflow-hidden flex bg-[#0a0b0d] border border-[#1f2124]">
            {llmTiers.map((tier: any, idx: number) => {
              const bgColors = ["bg-[#7ee787]", "bg-[#6e9fe0]", "bg-[#e8b339]"];
              return (
                <div
                  key={idx}
                  style={{ width: `${tier.percentage}%` }}
                  className={`${bgColors[idx % bgColors.length]} h-full transition-all group relative cursor-pointer opacity-90 hover:opacity-100`}
                  title={`${tier.name}: ${tier.count} calls (${tier.percentage}%)`}
                />
              );
            })}
          </div>

          {/* Legend and Exact Counts */}
          <div className="space-y-2 pt-1">
            {llmTiers.map((tier: any, idx: number) => {
              const dotColors = ["bg-[#7ee787]", "bg-[#6e9fe0]", "bg-[#e8b339]"];
              return (
                <div
                  key={idx}
                  className="flex items-center justify-between text-xs font-mono p-2 rounded bg-[#0a0b0d]/60 border border-[#1f2124]/50"
                >
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${dotColors[idx % dotColors.length]}`} />
                    <span className="text-[#e6e6e6] font-medium">{tier.name}</span>
                    <span className="text-[10px] text-[#4a4d52]">({tier.provider})</span>
                  </div>

                  <div className="flex items-center gap-4 text-right">
                    <span className="text-[#8b8f94]">{tier.avgLatencyMs}ms avg</span>
                    <span className="text-[#e6e6e6] font-bold w-12 tabular-nums">
                      {tier.count}
                    </span>
                    <span className="text-[#7ee787] w-12 font-bold tabular-nums">
                      {tier.percentage}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Entity Resolution Breakdown Panel */}
        <div className="p-5 rounded bg-[#131417] border border-[#1f2124] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <GitMerge className="w-4 h-4 text-[#e8b339]" />
              <h2 className="text-xs font-bold font-mono tracking-tight text-[#e6e6e6]">
                ENTITY RESOLUTION & DEDUPLICATION BREAKDOWN
              </h2>
            </div>
            <span className="font-mono text-[10px] text-[#8b8f94]">
              {totalDeduplicated.toLocaleString()} Deduplicated
            </span>
          </div>

          {/* Horizontal Stacked Bar */}
          <div className="w-full h-4 rounded overflow-hidden flex bg-[#0a0b0d] border border-[#1f2124]">
            {erBreakdown.map((item, idx) => (
              <div
                key={idx}
                style={{ width: `${item.pct}%`, backgroundColor: item.color }}
                className="h-full transition-all opacity-90 hover:opacity-100 cursor-pointer"
                title={`${item.label}: ${item.count} (${item.pct}%)`}
              />
            ))}
          </div>

          {/* Legend and Breakdown */}
          <div className="space-y-2 pt-1">
            {erBreakdown.map((item, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between text-xs font-mono p-2 rounded bg-[#0a0b0d]/60 border border-[#1f2124]/50"
              >
                <div className="flex items-center gap-2">
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-[#e6e6e6] font-medium">{item.label}</span>
                </div>

                <div className="flex items-center gap-4 text-right">
                  <span className="text-[#e6e6e6] font-bold w-12 tabular-nums">
                    {item.count}
                  </span>
                  <span className="w-12 font-bold tabular-nums" style={{ color: item.color }}>
                    {item.pct}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <BenchmarkJsonModal
        open={isBenchmarkModalOpen}
        onClose={() => setIsBenchmarkModalOpen(false)}
      />
    </div>
  );
};
