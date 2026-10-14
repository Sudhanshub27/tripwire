import React, { useEffect, useState } from "react";
import { LayoutDashboard, Database, GitMerge, Terminal, Search, Share2, Sparkles } from "lucide-react";

interface SidebarProps {
  currentRoute: string;
  onNavigate: (route: string) => void;
  onOpenCmdK: () => void;
  mockMode?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentRoute,
  onNavigate,
  onOpenCmdK,
  mockMode: mockModeProp,
}) => {
  const [mockMode, setMockMode] = useState<boolean>(mockModeProp ?? false);

  useEffect(() => {
    if (mockModeProp !== undefined) {
      setMockMode(mockModeProp);
    } else {
      fetch("/api/config")
        .then((res) => res.json())
        .then((data) => {
          if (data) {
            const val = data.mockMode ?? data.mock_mode;
            if (typeof val === "boolean") setMockMode(val);
          }
        })
        .catch(() => {});
    }
  }, [mockModeProp]);

  const navItems = [
    { id: "overview", label: "Overview", icon: LayoutDashboard },
    { id: "browser", label: "Data Browser", icon: Database },
    { id: "resolution", label: "Entity Resolution", icon: GitMerge },
    { id: "graph", label: "Knowledge Graph", icon: Share2 },
    { id: "vector", label: "Hybrid Search", icon: Sparkles },
    { id: "logs", label: "Pipeline Logs", icon: Terminal },
  ];

  return (
    <aside className="fixed top-0 left-0 bottom-0 w-[220px] bg-[#131417] border-r border-[#1f2124] flex flex-col z-30 select-none">
      {/* Brand Header */}
      <div className="h-14 px-4 flex items-center border-b border-[#1f2124]">
        <div className="flex items-center gap-2.5">
          <img
            src="/logo.png"
            alt="Tripwire Logo"
            className="w-6 h-6 object-contain rounded"
            onError={(e) => {
              e.currentTarget.style.display = 'none';
              const fallback = e.currentTarget.nextElementSibling as HTMLElement;
              if (fallback) fallback.style.display = 'flex';
            }}
          />
          <div className="w-6 h-6 rounded bg-[#7ee787] hidden items-center justify-center font-mono text-xs font-bold text-[#0a0b0d]">
            T
          </div>
          <div>
            <h1 className="text-xs font-bold font-mono tracking-tight text-[#e6e6e6]">
              TRIPWIRE
            </h1>
            <p className="text-[10px] font-mono text-[#8b8f94]">INGESTION PIPELINE</p>
          </div>
        </div>
      </div>

      {/* Quick Search trigger button */}
      <div className="p-3">
        <button
          onClick={onOpenCmdK}
          className="w-full h-8 px-2.5 rounded bg-[#0a0b0d] border border-[#1f2124] hover:border-[#2d3035] text-left flex items-center justify-between text-xs text-[#8b8f94] transition-colors"
        >
          <span className="flex items-center gap-2 font-mono text-[11px]">
            <Search className="w-3.5 h-3.5 text-[#4a4d52]" />
            Search...
          </span>
          <kbd className="font-mono text-[9px] px-1 py-0.5 rounded bg-[#131417] border border-[#1f2124] text-[#4a4d52]">
            ⌘K
          </kbd>
        </button>
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 px-2 py-2 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentRoute === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full h-9 px-3 rounded flex items-center gap-2.5 text-xs font-medium transition-all ${
                isActive
                  ? "bg-[#1f2124]/50 text-[#e6e6e6] border-l-2 border-[#7ee787]"
                  : "text-[#8b8f94] hover:text-[#e6e6e6] hover:bg-[#1f2124]/30"
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? "text-[#7ee787]" : "text-[#4a4d52]"}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-3 border-t border-[#1f2124] bg-[#0a0b0d]/50">
        <div className="flex items-center justify-between font-mono text-[10px] text-[#4a4d52]">
          <span>MOCK_MODE</span>
          <span className={mockMode ? "text-[#7ee787] font-semibold" : "text-[#8b8f94] font-semibold"}>
            {mockMode ? "ACTIVE" : "OFF (LIVE)"}
          </span>
        </div>
      </div>
    </aside>
  );
};
