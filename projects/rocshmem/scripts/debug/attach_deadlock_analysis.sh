#!/usr/bin/env bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################
#
# attach_deadlock_analysis.sh — Attach rocgdb to every running instance of an
#   executable, run the rocSHMEM deadlock analysis script on each, save full
#   output to per-process files, and print a compact summary to stdout.
#
# Usage:
#   ./attach_deadlock_analysis.sh <executable_name> [options]
#
# Options:
#   --directory <dir>      Output directory (default: rocshmem_deadlock_analysis_<timestamp>)
#   --cull                 Cull groups stuck in GPU barriers / gridsync (built-in patterns)
#   --cull=p1,p2,...       Cull groups whose backtrace contains any of the given substrings
#   --check-lanes          Enable per-lane register divergence check inside each group
#   --stats                Print per-group wavefront statistics
#   --color                Force ANSI color output (default when stdout is a tty)
#   --no-color             Disable ANSI color output
#   --stdout               After analysis, emit all output files to stdout wrapped in
#                          BEGIN/END FILE markers (used by mpiexec --tag-output collection)
#
# Options (via environment variables):
#   ROCGDB                 Path to rocgdb binary (default: rocgdb)
#   ROCSHMEM_GDB_TIMEOUT   Per-process timeout in seconds (default: 120)
#
# Examples:
#   ./attach_deadlock_analysis.sh rocshmem_functional_tests
#   ./attach_deadlock_analysis.sh rocshmem_functional_tests --directory /tmp/my_analysis
#   ./attach_deadlock_analysis.sh rocshmem_functional_tests --cull
#   ./attach_deadlock_analysis.sh rocshmem_functional_tests --check-lanes --stats
#   ./attach_deadlock_analysis.sh rocshmem_functional_tests --directory /tmp/my_analysis --cull=__syncthreads,cooperative_groups
#   ROCSHMEM_GDB_TIMEOUT=60 ./attach_deadlock_analysis.sh my_app
###############################################################################

set -o pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GDB_SCRIPT="${SCRIPT_DIR}/rocgdb_deadlock_analysis.py"

EXECUTABLE="${1:-}"
if [[ -z "${EXECUTABLE}" ]]; then
    echo "Usage: $0 <executable_name> [--directory <dir>] [--cull[=p1,p2,...]] [--check-lanes] [--stats] [--color|--no-color]" >&2
    echo "  Attaches rocgdb to all running <executable_name> processes on this host," >&2
    echo "  runs the rocSHMEM deadlock analysis, and saves output to the output dir." >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR=""
CULL_ENV=""
CHECK_LANES_ENV="0"
STATS_ENV="0"
COLOR_ENV=""
STDOUT_MODE="0"

# Parse options from remaining arguments
_args=("${@:2}")
_i=0
while [[ ${_i} -lt ${#_args[@]} ]]; do
    _arg="${_args[${_i}]}"
    if [[ "${_arg}" == "--directory" ]]; then
        (( _i++ ))
        OUTPUT_DIR="${_args[${_i}]:-}"
        if [[ -z "${OUTPUT_DIR}" ]]; then
            echo "ERROR: --directory requires an argument." >&2
            exit 1
        fi
    elif [[ "${_arg}" == --directory=* ]]; then
        OUTPUT_DIR="${_arg#--directory=}"
    elif [[ "${_arg}" == "--cull" ]]; then
        CULL_ENV="1"
    elif [[ "${_arg}" == --cull=* ]]; then
        CULL_ENV="${_arg#--cull=}"
    elif [[ "${_arg}" == "--check-lanes" ]]; then
        CHECK_LANES_ENV="1"
    elif [[ "${_arg}" == "--stats" ]]; then
        STATS_ENV="1"
    elif [[ "${_arg}" == "--color" ]]; then
        COLOR_ENV="always"
    elif [[ "${_arg}" == "--no-color" ]]; then
        COLOR_ENV="never"
    elif [[ "${_arg}" == "--stdout" ]]; then
        STDOUT_MODE="1"
    else
        echo "ERROR: Unknown option '${_arg}'." >&2
        exit 1
    fi
    (( _i++ ))
done

OUTPUT_DIR="${OUTPUT_DIR:-rocshmem_deadlock_analysis_${TIMESTAMP}}"

ROCGDB="${ROCGDB:-rocgdb}"
GDB_TIMEOUT="${ROCSHMEM_GDB_TIMEOUT:-120}"
if [[ ! "${GDB_TIMEOUT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ROCSHMEM_GDB_TIMEOUT must be a non-negative integer, got: '${GDB_TIMEOUT}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Color support for the script's own output
# ---------------------------------------------------------------------------

# Resolve color mode: explicit flag > env var > tty auto-detect
_use_color=0
if [[ "${COLOR_ENV}" == "always" ]]; then
    _use_color=1
elif [[ "${COLOR_ENV}" == "never" ]]; then
    _use_color=0
elif [[ -t 1 ]]; then
    _use_color=1
fi

if [[ ${_use_color} -eq 1 ]]; then
    _c_reset=$'\033[0m'
    _c_ok=$'\033[1;32m'       # bold green  — OK
    _c_warn=$'\033[1;33m'     # bold yellow — WARN
    _c_bad=$'\033[1;31m'      # bold red    — TIMEOUT / SKIP / bad counts
    _c_header=$'\033[1;34m'   # bold blue   — section headers
    _c_hint=$'\033[1;36m'     # bold cyan   — HINT text (matches rocgdb)
    _c_fn=$'\033[1;32m'       # bold green  — rocSHMEM function name
else
    _c_reset=''
    _c_ok=''
    _c_warn=''
    _c_bad=''
    _c_header=''
    _c_hint=''
    _c_fn=''
fi

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

if [[ ! -f "${GDB_SCRIPT}" ]]; then
    echo "ERROR: GDB analysis script not found: ${GDB_SCRIPT}" >&2
    exit 1
fi

if ! command -v "${ROCGDB}" >/dev/null 2>&1; then
    echo "ERROR: '${ROCGDB}' not found in PATH." >&2
    echo "       Set the ROCGDB environment variable to the full path of rocgdb." >&2
    exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
    echo "ERROR: 'timeout' command not found (required for per-process timeout)." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" || {
    echo "ERROR: Cannot create output directory: ${OUTPUT_DIR}" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Discover PIDs
# ---------------------------------------------------------------------------

PIDS=()

# Use ps 'args' (full command line) rather than 'comm' (truncated to 15 chars)
# so that long executable names like rocshmem_functional_tests are matched.
# Strip any leading path from argv[0] before comparing.
while IFS= read -r pid; do
    [[ -n "${pid}" ]] && PIDS+=("${pid}")
done < <(ps -eo pid,args 2>/dev/null \
         | awk -v exe="${EXECUTABLE}" \
               'NR>1 { cmd=$2; sub(".*/","",cmd); if (cmd==exe) print $1 }')

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "No processes found matching executable '${EXECUTABLE}'." >&2
    echo "Ensure the application is running on this host." >&2
    exit 1
fi

echo "Found ${#PIDS[@]} process(es) matching '${EXECUTABLE}': ${PIDS[*]}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Per-process timeout: ${GDB_TIMEOUT}s"
echo ""

# ---------------------------------------------------------------------------
# Attach and analyze each process
# ---------------------------------------------------------------------------

ANALYZED=0
FAILED=0

for PID in "${PIDS[@]}"; do
    # Read the MPI PE rank from the process environment (null-separated entries).
    # Falls back to "unknown" if the process has already exited or is not an MPI rank.
    PE_RANK="$(tr '\0' '\n' < "/proc/${PID}/environ" 2>/dev/null \
               | sed -n 's/^OMPI_COMM_WORLD_RANK=//p' \
               | tr -cd '0-9' | head -c 10)"
    PE_RANK="${PE_RANK:-unknown}"

    OUT_FILE="${OUTPUT_DIR}/pe${PE_RANK}_pid_${PID}.txt"

    printf "Analyzing PID %s (PE %s) ... " "${PID}" "${PE_RANK}"

    # Check the process still exists before attempting to attach
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "SKIP (process no longer exists)"
        echo "SKIP: PID ${PID} exited before attach" >> "${OUT_FILE}"
        (( FAILED++ )) || true
        continue
    fi

    # Run rocgdb in batch mode.
    # ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1 causes the Python script to run the
    # analysis immediately upon attach (the process is stopped on attach).
    _gdb_env=(
        ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1
        ROCSHMEM_DEADLOCK_CULL="${CULL_ENV}"
        ROCSHMEM_DEADLOCK_CHECK_LANES="${CHECK_LANES_ENV}"
        ROCSHMEM_DEADLOCK_STATS="${STATS_ENV}"
    )
    [[ -n "${COLOR_ENV}" ]] && _gdb_env+=(ROCSHMEM_DEADLOCK_COLOR="${COLOR_ENV}")

    timeout "${GDB_TIMEOUT}" \
        env "${_gdb_env[@]}" \
        "${ROCGDB}" -batch \
            -p "${PID}" \
            -x "${GDB_SCRIPT}" \
            >"${OUT_FILE}" 2>&1
    GDB_EXIT=$?

    # Check if we got useful output regardless of exit code
    # (rocgdb often returns non-zero even on success in batch mode)
    if grep -q 'Unique backtrace groups:' "${OUT_FILE}" 2>/dev/null; then
        printf "${_c_ok}OK${_c_reset}\n"
        (( ANALYZED++ )) || true
    elif [[ ${GDB_EXIT} -eq 124 ]]; then
        printf "${_c_bad}TIMEOUT${_c_reset} (${GDB_TIMEOUT}s)\n"
        echo "TIMEOUT: rocgdb did not finish within ${GDB_TIMEOUT}s" >> "${OUT_FILE}"
        (( FAILED++ )) || true
    elif grep -q 'No GPU wavefront threads found' "${OUT_FILE}" 2>/dev/null; then
        printf "${_c_ok}OK${_c_reset} (no GPU threads)\n"
        (( ANALYZED++ )) || true
    else
        printf "${_c_warn}WARN${_c_reset} (rocgdb exit=${GDB_EXIT}; see ${OUT_FILE})\n"
        (( FAILED++ )) || true
    fi
done

# ---------------------------------------------------------------------------
# Compact stdout summary
# ---------------------------------------------------------------------------

echo ""
printf "${_c_header}=== Attach and Analyze Summary ===${_c_reset}\n"
printf "Processes analyzed: %d / %d\n" "${ANALYZED}" "${#PIDS[@]}"
echo ""

for PID in "${PIDS[@]}"; do
    PE_RANK="$(tr '\0' '\n' < "/proc/${PID}/environ" 2>/dev/null \
               | sed -n 's/^OMPI_COMM_WORLD_RANK=//p' \
               | tr -cd '0-9' | head -c 10)"
    PE_RANK="${PE_RANK:-unknown}"
    OUT_FILE="${OUTPUT_DIR}/pe${PE_RANK}_pid_${PID}.txt"
    # If the process has exited since the analysis loop, fall back to any file
    # matching this PID (the rank may have been resolved then).
    if [[ ! -f "${OUT_FILE}" ]]; then
        OUT_FILE="$(ls "${OUTPUT_DIR}"/pe*_pid_${PID}.txt 2>/dev/null | head -1)"
    fi

    if [[ -z "${OUT_FILE}" || ! -f "${OUT_FILE}" ]]; then
        printf "  PE %-4s  PID %-8s  ${_c_bad}SKIPPED${_c_reset} (no output file)\n" "${PE_RANK}" "${PID}"
        continue
    fi

    # Extract group count from the analysis output
    # (avoid using $GROUPS which is a bash special variable for the user's GIDs)
    NUM_GROUPS="$(grep 'Unique backtrace groups:' "${OUT_FILE}" 2>/dev/null \
                  | awk '{print $NF}' | head -1)"
    NUM_GROUPS="${NUM_GROUPS:-?}"

    # Strip ANSI when extracting text for sorting/matching; re-apply the
    # script's own colors for display so the summary is consistently colored
    # regardless of whether rocgdb wrote colors into the output file.
    _strip_ansi='s/\x1b\[[0-9;]*m//g'

    # Always extract wavefront count inside rocSHMEM from the Summary section.
    ROCSHMEM_WFS="$(grep 'wavefront(s) inside rocSHMEM' "${OUT_FILE}" 2>/dev/null \
                    | sed "${_strip_ansi}" | awk '{print $1}' | head -1)"

    # Extract the dominant rocSHMEM API function (most frequently occurring).
    STUCK_FN="$(grep '\[rocSHMEM\] Stuck in:' "${OUT_FILE}" 2>/dev/null \
                | sed "${_strip_ansi}" \
                | sort | uniq -c | sort -rn | head -1 \
                | sed 's/^[[:space:]]*[0-9]*[[:space:]]*//' \
                | sed 's/.*\[rocSHMEM\] Stuck in:[[:space:]]*//')"

    # Determine the dominant stuck pattern from [HINT] lines
    # (most frequently occurring hint).
    HINT="$(grep '\[HINT\]' "${OUT_FILE}" 2>/dev/null \
            | sed "${_strip_ansi}" \
            | sort | uniq -c | sort -rn | head -1 \
            | sed 's/^[[:space:]]*[0-9]*[[:space:]]*//' \
            | sed 's/^\[HINT\][[:space:]]*//')"

    # Apply script-level colors to the extracted plain-text fields.
    [[ -n "${STUCK_FN}" ]] && STUCK_FN="${_c_fn}${STUCK_FN}${_c_reset}"
    [[ -n "${HINT}" ]]     && HINT="${_c_hint}${HINT}${_c_reset}"

    if [[ -n "${ROCSHMEM_WFS}" ]]; then
        DOMINANT="${_c_bad}${ROCSHMEM_WFS}${_c_reset} wf(s) in ${STUCK_FN:-rocSHMEM}"
        if [[ -n "${HINT}" ]]; then
            DOMINANT="${DOMINANT}; ${HINT}"
        else
            DOMINANT="${DOMINANT}; No deadlock hint available"
        fi
    elif grep -q 'No GPU wavefront threads found' "${OUT_FILE}" 2>/dev/null; then
        DOMINANT="(no GPU threads found)"
    elif grep -q 'SKIP\|TIMEOUT' "${OUT_FILE}" 2>/dev/null; then
        DOMINANT="${_c_bad}$(grep -m1 'SKIP\|TIMEOUT' "${OUT_FILE}" \
                    | sed "${_strip_ansi}" | sed 's/^[[:space:]]*//')${_c_reset}"
    else
        DOMINANT="${HINT:-0 wf(s) in rocSHMEM}"
    fi

    printf "  PE %-4s  PID %-8s  groups=%-3s  %s\n" "${PE_RANK}" "${PID}" "${NUM_GROUPS}" "${DOMINANT}"
done

if [[ "${STDOUT_MODE}" -eq 0 ]]; then
    echo ""
    echo "Full output in: ${OUTPUT_DIR}/"
fi

# In --stdout mode, emit every output file to stdout wrapped in markers so that
# the caller (mpiexec --tag-output) can reconstruct the files on the head node.
if [[ "${STDOUT_MODE}" -eq 1 ]]; then
    for _f in "${OUTPUT_DIR}"/pe*_pid_*.txt; do
        [[ -f "${_f}" ]] || continue
        echo "=== BEGIN FILE: $(basename "${_f}") ==="
        cat "${_f}"
        echo "=== END FILE: $(basename "${_f}") ==="
    done
fi

# Exit with non-zero if any process failed analysis
[[ ${FAILED} -eq 0 ]]
