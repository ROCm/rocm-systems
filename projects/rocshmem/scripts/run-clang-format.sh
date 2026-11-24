#!/usr/bin/env bash
# run-clang-format.sh - apply or check clang-format on rocSHMEM sources.
#
# Defaults to src/ and include/. Files listed in .clang-format-ignore are
# skipped (clang-format honours that file itself).
#
# Usage (run from projects/rocshmem):
#   scripts/run-clang-format.sh [path ...]          # format in place
#   scripts/run-clang-format.sh --list [path ...]   # list files that would change
#   scripts/run-clang-format.sh --diff [path ...]   # show diff, change nothing
#
# Nothing is modified in --list/--diff mode. Set CLANG_FORMAT to use a
# non-default binary.

set -euo pipefail

MIN_VERSION=22

usage() { echo "Usage: $0 [--list | --diff] [path ...]" >&2; }

MODE=""
PATHS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list | --diff)
      [[ -n "$MODE" ]] && { usage; exit 1; }
      MODE="$1" ;;
    -h | --help) usage; exit 0 ;;
    --*) usage; exit 1 ;;
    *) PATHS+=("$1") ;;
  esac
  shift
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ ${#PATHS[@]} -eq 0 ]] && PATHS=(examples include src tests)

# Prefer a versioned binary (clang-format-22) over a possibly older default.
if [[ -z "${CLANG_FORMAT:-}" ]]; then
  if command -v "clang-format-$MIN_VERSION" &>/dev/null; then
    CLANG_FORMAT="clang-format-$MIN_VERSION"
  else
    CLANG_FORMAT=clang-format
  fi
fi
if ! command -v "$CLANG_FORMAT" &>/dev/null; then
  echo "ERROR: clang-format not found. Install clang-format >= $MIN_VERSION." >&2
  exit 2
fi
VERSION=$("$CLANG_FORMAT" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 | cut -d. -f1)
if [[ -z "$VERSION" || "$VERSION" -lt "$MIN_VERSION" ]]; then
  echo "ERROR: clang-format >= $MIN_VERSION required, found: $("$CLANG_FORMAT" --version)" >&2
  exit 2
fi

FILES=()
for p in "${PATHS[@]}"; do
  if [[ -f "$p" ]]; then
    FILES+=("$p")
  elif [[ -d "$p" ]]; then
    while IFS= read -r f; do FILES+=("$f"); done < <(
      find "$p" -type f \( -name '*.cc' -o -name '*.cpp' -o -name '*.hpp' \
        -o -name '*.h' -o -name '*.hip' \) | sort)
  else
    echo "ERROR: path not found: $p" >&2; exit 1
  fi
done
[[ ${#FILES[@]} -eq 0 ]] && { echo "No source files found." >&2; exit 0; }

NEED=()
IGNORED=()
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
for f in "${FILES[@]}"; do
  "$CLANG_FORMAT" --style=file "$f" >"$TMP"
  # No output means the file is excluded by .clang-format-ignore (not emptied).
  if [[ ! -s "$TMP" ]]; then IGNORED+=("$f"); continue; fi
  diff -q "$f" "$TMP" >/dev/null && continue
  NEED+=("$f")
  case "$MODE" in
    --diff) echo "--- $f"; diff -u "$f" "$TMP" || true ;;
    "") "$CLANG_FORMAT" -i --style=file "$f" ;;
  esac
done

echo "Would change/changed ${#NEED[@]} of ${#FILES[@]} files; ignored ${#IGNORED[@]}."
if [[ "$MODE" == "--list" ]]; then
  for f in ${NEED[@]+"${NEED[@]}"}; do echo "  $f"; done
fi
