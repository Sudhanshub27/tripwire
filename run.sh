#!/usr/bin/env bash

# ==============================================================================
# TRIPWIRE PIPELINE: MASTER EXECUTION SCRIPT
# ==============================================================================
# Executes virtual environment setup, automated unit/integration tests, multi-target data
# ingestion, LLM extraction, entity resolution, vector store indexing, and dashboard Web UI.
#
# Usage:
#   ./run.sh                  # Run full end-to-end pipeline (defaults to --dry-run if no args)
#   ./run.sh --dry-run        # Fast dry-run on sample records
#   ./run.sh --limit 50       # Run with custom record limits
#   ./run.sh --skip-tests     # Skip test suite and run ingestion + dashboard
#   ./run.sh --dashboard      # Skip ingestion and launch dashboard Web UI directly
#   ./run.sh --eval           # Run full pipeline scientific benchmarks
#   ./run.sh --incremental    # Run incremental dedup benchmark (baseline vs incremental)
# ==============================================================================

set -e

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "🚀 TRIPWIRE DATA PIPELINE: INITIALIZING MASTER RUN"
echo "======================================================================"

# 1. Check & Activate Virtual Environment
if [ -d "venv" ]; then
    echo "▶ Activating Python virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "▶ Activating Python virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "▶ Creating new Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "▶ Installing dependencies from requirements.txt..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
fi

# Check for --incremental flag
if [[ "$*" == *"--incremental"* ]] || [[ "$*" == *"--bench-incremental"* ]]; then
    echo ""
    echo "======================================================================"
    echo "⚡ EXECUTING INCREMENTAL PROCESSING BENCHMARK"
    echo "======================================================================"
    python3 -m evaluation.incremental_benchmark --records 20 --output evaluation/reports/incremental_benchmark.json
    echo "✅ Incremental benchmark complete. Report saved to evaluation/reports/incremental_benchmark.json"
    exit 0
fi

# Check for --eval flag
if [[ "$*" == *"--eval"* ]]; then
    echo ""
    echo "======================================================================"
    echo "📈 EXECUTING SCIENTIFIC PIPELINE EVALUATION BENCHMARK"
    echo "======================================================================"
    python3 -m evaluation.benchmark --mode mock --records 20 --iterations 3 --warmup-records 2 --output evaluation/reports/benchmark_report.json
    python3 -m evaluation.incremental_benchmark --records 20 --output evaluation/reports/incremental_benchmark.json
    python3 -m evaluation.llm.evaluator --provider chain --output evaluation/reports/llm_chain_eval.json
    python3 -m evaluation.resolution.evaluator --threshold 85.0 --output evaluation/reports/resolution_eval.json
    echo "✅ Scientific benchmark execution complete. Reports saved to evaluation/reports/"
    exit 0
fi

# Check for --dashboard or --serve flag
LAUNCH_DASHBOARD_DIRECT=0
if [[ "$*" == *"--dashboard"* ]] || [[ "$*" == *"--serve"* ]]; then
    LAUNCH_DASHBOARD_DIRECT=1
fi

# 2. Run Test Suite (unless skipped or dashboard-only)
if [ "$LAUNCH_DASHBOARD_DIRECT" -eq 0 ]; then
    if [[ "$*" == *"--skip-tests"* ]] || [[ "$*" == *"--no-tests"* ]]; then
        echo "▶ Skipping automated test suite (--skip-tests flag passed)..."
    else
        echo ""
        echo "======================================================================"
        echo "🧪 STAGE 1: RUNNING AUTOMATED UNIT & INTEGRATION TEST SUITE"
        echo "======================================================================"
        python3 -m pytest tests/ -v
    fi

    # Filter out script-only flags for src.main
    CLEAN_ARGS=()
    for arg in "$@"; do
        if [[ "$arg" != "--skip-tests" && "$arg" != "--no-tests" && "$arg" != "--dashboard" && "$arg" != "--serve" ]]; then
            CLEAN_ARGS+=("$arg")
        fi
    done

    # Default to --dry-run if no CLI flags are passed for fast launch
    if [ ${#CLEAN_ARGS[@]} -eq 0 ]; then
        CLEAN_ARGS=("--dry-run")
    fi

    # 3. Run Main Pipeline Ingestion Orchestrator
    echo ""
    echo "======================================================================"
    echo "⚡ STAGE 2: EXECUTING MULTI-TARGET INGESTION ORCHESTRATOR"
    echo "======================================================================"
    python3 -m src.main run all "${CLEAN_ARGS[@]}"

    # 4. Run Sheets / CSV Export (safe fallback)
    echo ""
    echo "======================================================================"
    echo "📊 STAGE 3: EXECUTING GOOGLE SHEETS / CSV EXPORT SYNC"
    echo "======================================================================"
    EXPORT_DRY_RUN=""
    for a in "${CLEAN_ARGS[@]}"; do
        if [[ "$a" == "--dry-run" ]]; then
            EXPORT_DRY_RUN="--dry-run"
        fi
    done
    python3 -m src.main export sheets $EXPORT_DRY_RUN
fi

# 5. Build React Dashboard Frontend Production Bundle
if [ -d "src/dashboard/frontend" ]; then
    echo ""
    echo "======================================================================"
    echo "📦 STAGE 4: BUILDING REACT FRONTEND PRODUCTION BUNDLE"
    echo "======================================================================"
    python3 -c "from src.dashboard.main import sync_static_assets; sync_static_assets()" || true
    (cd src/dashboard/frontend && npm run build)
fi

# 6. Launch Web Dashboard UI
echo ""
echo "======================================================================"
echo "🌐 STAGE 5: LAUNCHING REACT + FASTAPI DASHBOARD WEB UI"
echo "======================================================================"

# Automatically clear port 8000 if previously occupied
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8000/tcp 2>/dev/null || true
    sleep 0.5
elif command -v lsof >/dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 0.5
fi

echo "▶ Dashboard server running at: http://localhost:8000"
echo "▶ Press Ctrl+C to stop the dashboard server."

# Attempt to open browser automatically
(sleep 1.5 && python3 -m webbrowser "http://localhost:8000" 2>/dev/null) &

exec python3 -m uvicorn src.dashboard.main:app --host 0.0.0.0 --port 8000
