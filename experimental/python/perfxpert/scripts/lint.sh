#!/bin/bash
# scripts/lint.sh — Scan all .md files for banned API strings
# Part of the docs-audit tooling

set +e

# Parse flags
STRICT=0
for arg in "$@"; do
  case "$arg" in
    --strict) STRICT=1 ;;
  esac
done

# Banned strings — from design spec Section 4 Dimension 1
# + total-legacy-scrub additions (post-refactor cleanup).
#
# Historical-anchor exception: any hit on a line that contains the literal
# phrase "removed in Phase 7.1" is ignored — this lets us keep a searchable
# record of removed flags/classes without re-introducing live guidance.
# The literal string is a load-bearing linter anchor; do not edit it.
BANNED=(
  # Dimension 1 (original 9)
  "interactive\.py"
  "LLMConversation"
  "llm_analyzer\.analyze_with_llm"
  "\-\-interactive"
  "\-\-resume-session"
  "AnalysisContext"
  "ROCINSIGHT_LLM_"
  "ROCPD_LLM_"
  "\.resume\(\)"
  # Total-legacy-scrub additions
  "PERFXPERT_USE_AGENTS"
  "ROCINSIGHT_"
  "_route_to_legacy"
  "_route_to_agents"
  "_is_legacy_mode"
  "_LegacyTraceAnalysis"
  "LLMAnalyzer"
  "from perfxpert\.ai_analysis"
  # analyze_database( — only as bare top-level public API call.
  # Match the open-paren form; prose/docstring mentions of
  # "analyze_database" without a call paren still pass.
  "analyze_database("
)

# Search paths: experimental/python/perfxpert only. The repo-root docs/
# directory contains general ROCm monorepo docs (continuous-integration.md,
# gardening.md, etc.) — no perfxpert content. All perfxpert .md files live
# under experimental/python/perfxpert/.
SEARCH_DIRS=(
  "experimental/python/perfxpert"
)

VIOLATION_COUNT=0

# Scan all .md files (excluding audit directories and git directories).
# Exclusion list is documented in experimental/python/perfxpert/docs/
# known-issues.md under "Scanner scope limitations → scripts/lint.sh".
# Rationale for each path:
#   - docs/confluence: Confluence-amendment audit artifacts that
#     describe what was removed / renamed (must reference the old
#     symbol names to explain what was scrubbed). Not shipped docs.
#   - opencode/: upstream opencode submodule (MIT). Its tree + bun
#     node_modules contain unrelated third-party `.resume()` / READMEs
#     that would false-positive our brand-scrub list.
#   - node_modules: JS dep tree under any nested path; not ours.
#   - .git: internals
#   - .pytest_cache: test runner artifacts
#   - perfxpert/ai_analysis: the pre-agentic module tree; the
#     refactor deletes the entire directory, so scrubbing
#     those files now would produce a merge conflict the rebase
#     immediately resolves by removing them.
for dir in "${SEARCH_DIRS[@]}"; do
  if [ ! -d "$dir" ]; then
    continue
  fi

  while IFS= read -r file; do
    for banned in "${BANNED[@]}"; do
      # Use fgrep semantics for the historical-anchor filter so the
      # literal phrase is matched regardless of the banned pattern's
      # regex syntax. A line with "removed in Phase 7.1" on it is
      # treated as a historical marker and ignored.
      if grep "$banned" "$file" 2>/dev/null | grep -vqF 'removed in Phase 7.1'; then
        echo "FAIL: $file contains banned string '$banned'"
        VIOLATION_COUNT=$((VIOLATION_COUNT + 1))
      fi
    done
  done < <(find "$dir" -name "*.md" \
    -not -path "*docs/confluence/*" \
    -not -path "*/opencode/*" \
    -not -path "*/node_modules/*" \
    -not -path "*.git/*" \
    -not -path "*.pytest_cache/*" \
    -not -path "*perfxpert/ai_analysis/*")
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
