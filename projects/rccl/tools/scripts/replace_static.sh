#!/bin/bash
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Usage:
#   ./replace_static.sh <source_file> [--replace-vars] [--verbose] [--exclude-list=func1,func2,var1]
#
# - Replaces all 'static' function definitions with non-static.
# - Replaces all 'static inline' with 'inline'.
# - If --replace-vars is given, also replaces 'static' at variable definitions.
# - If --exclude-list is given, skips listed functions/variables.
# - If --verbose is given, shows a diff of the changes.

set -e

SOURCE_FILE="$1"
shift

REPLACE_VARS=0
VERBOSE=0
EXCLUDE_LIST=""

for arg in "$@"; do
  case "$arg" in
    --replace-vars) REPLACE_VARS=1 ;;
    --verbose) VERBOSE=1 ;;
    --exclude-list=*) EXCLUDE_LIST="${arg#*=}" ;;
  esac
done

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "Source file '$SOURCE_FILE' not found!"
  exit 1
fi

TMP_FILE="${SOURCE_FILE}.tmp.$$"
cp "$SOURCE_FILE" "$TMP_FILE"

# Prepare exclude regex if needed
if [[ -n "$EXCLUDE_LIST" ]]; then
  # Convert comma-separated list to alternation regex
  EXCLUDE_REGEX="($(echo "$EXCLUDE_LIST" | sed 's/,/|/g'))"
fi

# Mark lines with excluded function or variable names
if [[ -n "$EXCLUDE_LIST" ]]; then
  IFS=',' read -ra EXCLUDES <<< "$EXCLUDE_LIST"
  for name in "${EXCLUDES[@]}"; do
    # Mark function definitions/declarations to skip (robust to qualifiers/types)
    sed -E -i "/static[[:space:]]+([a-zA-Z_][a-zA-Z0-9_:[:space:]\*\&]*)[[:space:]]+${name}[[:space:]]*(\(|;)/s/^/__STATIC_SKIP__/" "$TMP_FILE"
    # Mark variable definitions/declarations to skip (no '(' on the line)
    sed -E -i '/\(/!s/static[[:space:]]+.*\b'"${name}"'\b[[:space:]]*(=|;)/__STATIC_SKIP__&/' "$TMP_FILE"
  done
fi

# s/\bstatic[[:space:]]+inline\b/inline/g
  #   - Matches 'static' followed by one or more spaces and then 'inline' as a whole word.
  #   - Replaces it with just 'inline'.
  #   - Example: 'static inline int foo()' -> 'inline int foo()'
# s/^([[:space:]]*(inline[[:space:]]+|__device__[[:space:]]+|__forceinline__[[:space:]]+|__host__[[:space:]]+|__global__[[:space:]]+|)*)static[[:space:]]+/\1/g
  #   - Matches lines that start with optional whitespace, then any qualifiers (inline, __device__, etc.), then 'static' and spaces.
  #   - Removes 'static' but preserves the qualifiers and indentation.
  #   - Example: '  inline static int foo()' -> '  inline int foo()'
sed -E -i '/^__STATIC_SKIP__/!{
  s/\bstatic[[:space:]]+inline\b/inline/g
  s/^([[:space:]]*(inline[[:space:]]+|__device__[[:space:]]+|__forceinline__[[:space:]]+|__host__[[:space:]]+|__global__[[:space:]]+|)*)static[[:space:]]+/\1/g
}' "$TMP_FILE"

# # Always remove 'static' from function definitions/declarations, except excluded
# sed -E -i '/^__STATIC_SKIP__/!s/^([[:space:]]*(inline[[:space:]]+|__device__[[:space:]]+|__forceinline__[[:space:]]+|__host__[[:space:]]+|__global__[[:space:]]+|)*)static[[:space:]]+/\1/g' "$TMP_FILE"

# # Replace 'static inline' with 'inline' everywhere except marked lines
# sed -E -i '/^__STATIC_SKIP__/!s/\bstatic[[:space:]]+inline\b/inline/g' "$TMP_FILE"

# Remove 'static' at variable definitions, excluding variables in EXCLUDE_LIST
if [[ "$REPLACE_VARS" == "1" ]]; then
  if [[ -n "$EXCLUDE_LIST" ]]; then
    # Regex explanation:
    # '/^__STATIC_SKIP__/{b}; /\(/b; s/^([[:space:]]*)static([[:space:]]+)/\1/g'
    # - /^__STATIC_SKIP__/{b};   If the line starts with __STATIC_SKIP__, branch (skip substitution).
    # - /\(/b;                  If the line contains '(', branch (skip substitution; likely a function).
    # - s/^([[:space:]]*)static([[:space:]]+)/\1/g
    #     - Matches 'static' at the start of a line (possibly after indentation).
    #     - Removes 'static', preserving indentation.
    #     - Only applies to lines not skipped above (i.e., not excluded and not functions).
    sed -E -i '/^__STATIC_SKIP__/{b}; /\(/b; s/^([[:space:]]*)static([[:space:]]+)/\1/g' "$TMP_FILE"
  else
    # Regex explanation:
    # '/\(/!s/^([[:space:]]*)static([[:space:]]+)/\1/g'
    # - /\(/!    Only apply to lines that do NOT contain '(' (i.e., not functions).
    # - s/^([[:space:]]*)static([[:space:]]+)/\1/g
    #     - Matches 'static' at the start of a line (possibly after indentation).
    #     - Removes 'static', preserving indentation.
    sed -E -i '/\(/!s/^([[:space:]]*)static([[:space:]]+)/\1/g' "$TMP_FILE"
  fi
fi

# Remove the marker after all substitutions, preserving original line formatting
sed -E -i 's/([[:space:]]*)__STATIC_SKIP__/\1/' "$TMP_FILE"

if [[ "$VERBOSE" == "1" ]]; then
  echo "[INFO] Showing diff for changes:"
  diff -u "$SOURCE_FILE" "$TMP_FILE" || true
fi

mv "$TMP_FILE" "$SOURCE_FILE"
echo "Static function replacement complete for $SOURCE_FILE"
if [[ "$REPLACE_VARS" == "1" ]]; then
  echo "Static variable replacement also performed."
fi

