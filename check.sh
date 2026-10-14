#!/usr/bin/env bash
set -e

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🔍 PRE-PUSH VERIFICATION: RUNNING FULL CI & QUALITY GATES${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Locate Python environment binaries
PYTHON_BIN="python3"
if [ -f "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
elif [ -f "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
fi

RUFF_BIN="ruff"
if [ -f "./venv/bin/ruff" ]; then
    RUFF_BIN="./venv/bin/ruff"
elif [ -f "./.venv/bin/ruff" ]; then
    RUFF_BIN="./.venv/bin/ruff"
fi

PYTEST_BIN="pytest"
if [ -f "./venv/bin/pytest" ]; then
    PYTEST_BIN="./venv/bin/pytest"
elif [ -f "./.venv/bin/pytest" ]; then
    PYTEST_BIN="./.venv/bin/pytest"
fi

# Step 1: Python syntax check
echo -e "\n${BLUE}▶ 1/5 Checking Python syntax (compileall)...${NC}"
$PYTHON_BIN -m compileall -q src config evaluation tests
echo -e "${GREEN}✓ Python syntax valid.${NC}"

# Step 2: Ruff lint check
echo -e "\n${BLUE}▶ 2/5 Checking code style and imports (Ruff)...${NC}"
$RUFF_BIN check .
echo -e "${GREEN}✓ Ruff lint passed.${NC}"

# Step 3: Run pytest with coverage under simulated clean CI environment
echo -e "\n${BLUE}▶ 3/5 Running Pytest suite under clean CI conditions (isolated dist, .env & data/processed)...${NC}"
DIST_DIR="src/dashboard/frontend/dist"
BAK_DIR="src/dashboard/frontend/dist_ci_test_bak"
RESTORE_DIST=0

if [ -d "$DIST_DIR" ]; then
    mv "$DIST_DIR" "$BAK_DIR"
    RESTORE_DIST=1
fi

ENV_FILE=".env"
BAK_ENV=".env_ci_test_bak"
RESTORE_ENV=0

if [ -f "$ENV_FILE" ]; then
    mv "$ENV_FILE" "$BAK_ENV"
    RESTORE_ENV=1
fi

PROCESSED_DIR="data/processed"
BAK_PROCESSED="data/processed_ci_test_bak"
RESTORE_PROCESSED=0

if [ -d "$PROCESSED_DIR" ]; then
    mv "$PROCESSED_DIR" "$BAK_PROCESSED"
    mkdir -p "$PROCESSED_DIR"
    touch "$PROCESSED_DIR/.gitkeep"
    RESTORE_PROCESSED=1
fi

cleanup() {
    if [ "$RESTORE_DIST" -eq 1 ] && [ -d "$BAK_DIR" ]; then
        mv "$BAK_DIR" "$DIST_DIR"
    fi
    if [ "$RESTORE_ENV" -eq 1 ] && [ -f "$BAK_ENV" ]; then
        mv "$BAK_ENV" "$ENV_FILE"
    fi
    if [ "$RESTORE_PROCESSED" -eq 1 ] && [ -d "$BAK_PROCESSED" ]; then
        rm -rf "$PROCESSED_DIR"
        mv "$BAK_PROCESSED" "$PROCESSED_DIR"
    fi
}
trap cleanup EXIT INT TERM

$PYTEST_BIN tests/ --cov=src --cov-report=term-missing
echo -e "${GREEN}✓ Pytest passed under clean CI conditions.${NC}"

# Restore dist, .env, and data/processed
cleanup
RESTORE_DIST=0
RESTORE_ENV=0
RESTORE_PROCESSED=0

# Step 4: Frontend Lint & TypeScript Check
if [ -d "src/dashboard/frontend" ]; then
    echo -e "\n${BLUE}▶ 4/5 Checking Frontend (TypeScript & Lint)...${NC}"
    (cd src/dashboard/frontend && npm run lint)
    echo -e "${GREEN}✓ Frontend lint passed.${NC}"

    # Step 5: Frontend Production Build
    echo -e "\n${BLUE}▶ 5/5 Checking Frontend Production Build...${NC}"
    (cd src/dashboard/frontend && npm run build)
    echo -e "${GREEN}✓ Frontend build succeeded.${NC}"
fi

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}✅ ALL CHECKS PASSED! YOUR CODE IS 100% READY TO PUSH.${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo -e "You can safely push now:"
echo -e "  git add ."
echo -e "  git commit -m \"<your commit message>\""
echo -e "  git push"
echo -e "${GREEN}======================================================================${NC}"
