#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Find \NPI tasks (the markers left for "new product introduction" work) and
# print them one per line as:
#
#   $FILE:$LINE: $TODO
#
# A marker may also span
# multiple lines by ending each continued line with a trailing backslash; the
# comment leader (//, ///, #, *) on continuation lines is stripped and the text
# is joined with single spaces. For example:
#
#   /// \NPI I am a big complicated todo \
#   /// that takes multiple lines to describe.
#
# is reported as:
#
#   path/to/file:LINE: I am a big complicated todo that takes multiple lines to describe.
#
# Usage:
#   ./scripts/find-npi-tasks.sh [pathspec ...]
#
# With no arguments it scans tracked files under emulation/. Any arguments are
# forwarded to `git ls-files` as pathspecs (relative to the repository root).

set -euo pipefail

# Resolve this script's own path. It documents the \NPI convention (and so
# contains the marker), but it is not itself a task, so exclude it below.
self="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# docs/npi.md also documents the \NPI convention, carrying example markers
# rather than real tasks, so exclude it as well.
docs="$(cd "$(dirname "$0")/../docs" && pwd -P)/npi.md"

# Run from the repository root so reported paths are root-relative (e.g.
# emulation/rocjitsu/...), matching the paths git reports for tracked files.
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# Default to the emulation/ tree; allow callers to narrow or widen the scope.
pathspec=("$@")
if [[ ${#pathspec[@]} -eq 0 ]]; then
    pathspec=("emulation/")
fi

# Colorize output when FORCE_COLOR=1, or when stdout is a color-capable terminal
# and NO_COLOR is unset/empty (https://force-color.org/). File paths are shown in
# magenta and line numbers in green, matching grep's defaults.
c_file='' c_line='' c_reset=''
if [[ "${FORCE_COLOR:-}" == "1" ]] \
    || { [[ -z "${NO_COLOR:-}" && -t 1 && "${TERM:-}" != "dumb" ]]; }; then
    c_file=$'\033[35m' c_line=$'\033[32m' c_reset=$'\033[0m'
fi

# Scan tracked files, excluding this script and the convention's documentation.
git ls-files -z -- "${pathspec[@]}" \
    ":(exclude)${self#"$repo_root"/}" \
    ":(exclude)${docs#"$repo_root"/}" \
    | xargs -0 awk -v cfile="$c_file" -v cline="$c_line" -v creset="$c_reset" '
# A new file began while a multi-line task was still open: close it out first.
FNR == 1 && open { flush() }

# Strip a trailing backslash (the continuation marker) and remember whether one
# was present, so we know if the task carries on to the next line.
{ cont = sub(/\\[ \t]*$/, "") }

# Continuation line: drop the comment leader and append to the current task.
open {
    frag = $0
    sub(/^[ \t]*/, "", frag)
    sub(/^[\/#*]+[ \t]*/, "", frag)
    text = text " " frag
    if (!cont) flush()
    next
}

# Start of a task: capture everything after the \NPI marker on this line.
# Require whitespace or end-of-line after the marker so inline mentions such as
# `\NPI` in prose or the '\NPI' grep example are not mistaken for tasks.
/\\NPI([ \t]|$)/ {
    text = $0
    sub(/^.*\\NPI/, "", text)
    file = FILENAME
    line = FNR
    open = 1
    if (!cont) flush()
    next
}

END { if (open) flush() }

# Emit the accumulated task, collapsing internal whitespace to single spaces and
# dropping a trailing comment closer (--> or */).
function flush() {
    gsub(/[ \t]+/, " ", text)
    sub(/ *(-->|\*\/) *$/, "", text)
    sub(/^ +/, "", text)
    sub(/ +$/, "", text)
    printf "%s%s%s:%s%d%s: %s\n", cfile, file, creset, cline, line, creset, text
    open = 0
    text = ""
}
'
