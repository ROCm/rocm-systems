#!/bin/bash
# docs/lint.sh — Scan all .md files for banned API strings
# Part of Phase 9 docs audit tooling

set +e

# Parse flags
STRICT=0
for arg in "$@"; do
  case "$arg" in
    --strict) STRICT=1 ;;
  esac
done

# Banned strings (9 total) — from design spec Section 4 Dimension 1
BANNED=(
  "interactive\.py"
  "LLMConversation"
  "llm_analyzer\.analyze_with_llm"
  "\-\-interactive"
  "\-\-resume-session"
  "AnalysisContext"
  "ROCINSIGHT_LLM_"
  "ROCPD_LLM_"
  "\.resume\(\)"
)

# Search paths: experimental/python/perfxpert + docs/ (exclude phase specs/plans)
SEARCH_DIRS=(
  "experimental/python/perfxpert"
  "docs"
)

VIOLATION_COUNT=0

# Scan all .md files (excluding phase specs/plans and git directories)
for dir in "${SEARCH_DIRS[@]}"; do
  if [ ! -d "$dir" ]; then
    continue
  fi

  while IFS= read -r file; do
    for banned in "${BANNED[@]}"; do
      if grep -q "$banned" "$file" 2>/dev/null; then
        echo "FAIL: $file contains banned string '$banned'"
        VIOLATION_COUNT=$((VIOLATION_COUNT + 1))
      fi
    done
  done < <(find "$dir" -name "*.md" \
    -not -path "*/docs/superpowers/specs/*" \
    -not -path "*/docs/superpowers/plans/*" \
    -not -path "*/.git/*" \
    -not -path "*/.pytest_cache/*" \
    -not -path "*/perfxpert/ai_analysis/*")
done

if [ $VIOLATION_COUNT -eq 0 ]; then
  if [ $STRICT -eq 0 ]; then
    echo "✓ No banned strings detected"
  fi
  exit 0
else
  if [ $STRICT -eq 0 ]; then
    echo "✗ Found $VIOLATION_COUNT banned string violations"
  fi
  exit 1
fi
