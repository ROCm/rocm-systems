#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Turn HRR capture on for a workload, then say whether it actually recorded
# anything. Sole agent entry point for this skill.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSPECT="$SCRIPT_DIR/inspect_archive.py"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# The environment variable libamdhip64 reads to enable capture. Its presence as
# a literal in a shared object distinguishes a capture-capable runtime from one
# built without HRR.
CAPTURE_VAR="HIP_HRR_CAPTURE_OUTPUT"

usage() {
  cat <<'EOF' >&2
usage:
  hrr_capture.sh preflight [--output PATH] [--playback PATH]
  hrr_capture.sh run --output PATH [--playback PATH] [--] <command> [args...]
  hrr_capture.sh verify --output PATH [--playback PATH] [--json] [--no-playback]

  preflight  Check this environment can capture: which HIP runtime the workload
             will load, whether it has capture built in, and whether the output
             path is big enough and outlives the container.
  run        Preflight, then run the command with capture enabled, then verify.
  verify     Report what an existing archive contains.

Options:
  --output PATH        Archive directory to write (run/verify: required)
  --playback PATH      hrr-playback to cross-check the archive with. It has to be
                       the one matching the runtime that captured, or it will
                       refuse the archive over its format version
  --json               verify: emit JSON instead of text
  --no-playback        verify: report from the manifests only, no cross-check
  --skip-preflight     run: do not preflight at all
  --force              run: proceed even when preflight fails, for any reason:
                       no capture-capable runtime, too little space, or an
                       output path that will not outlive the container
  --min-free-gb N      Minimum free space to insist on (default 20)
  -h, --help           Show this help

Requires Linux, bash, GNU coreutils and python3. This script never selects a
GPU: whatever device mask the workload already uses is left alone, because
capture has to record the run being reproduced.
EOF
}

log()  { printf '[capture] %s\n' "$*" >&2; }
fail() { printf '[capture] error: %s\n' "$*" >&2; exit 1; }

# --- runtime discovery ------------------------------------------------------

# Does this shared object carry HRR capture?
lib_has_capture() {
  grep -a -q -- "$CAPTURE_VAR" "$1" 2>/dev/null
}

# Candidate libamdhip64 paths, most-likely-to-win first.
#
# Precedence follows the dynamic loader: LD_PRELOAD beats everything, then
# LD_LIBRARY_PATH, then a runtime bundled with the application itself, then the
# ldconfig cache, then the ROCm install. Bundled copies are worth listing
# because an application that ships its own runtime uses that one, whatever is
# installed on the system.
candidate_libs() {
  local entry dir pkg_dir
  local IFS_SAVE="$IFS"

  IFS=': '
  for entry in ${LD_PRELOAD:-}; do
    case "$entry" in *libamdhip64*) [[ -f "$entry" ]] && echo "LD_PRELOAD|$entry" ;; esac
  done
  for dir in ${LD_LIBRARY_PATH:-}; do
    for entry in "$dir"/libamdhip64.so*; do
      [[ -f "$entry" ]] && echo "LD_LIBRARY_PATH|$entry"
    done
  done
  IFS="$IFS_SAVE"

  # Runtimes bundled inside installed packages. Listed generically rather than
  # by package name: any package may ship one, and the site directories are
  # asked of the interpreter rather than guessed.
  while read -r pkg_dir; do
    [[ -n "$pkg_dir" && -d "$pkg_dir" ]] || continue
    while read -r entry; do
      [[ -n "$entry" ]] && echo "bundled package|$entry"
    done < <(find "$pkg_dir" -maxdepth 3 -name 'libamdhip64.so*' -type f 2>/dev/null)
  done < <(python3 - <<'PY' 2>/dev/null || true
import site
seen = []
for path in list(site.getsitepackages()) + [site.getusersitepackages()]:
    if path and path not in seen:
        seen.append(path)
        print(path)
PY
)

  if command -v ldconfig >/dev/null 2>&1; then
    while read -r entry; do
      [[ -f "$entry" ]] && echo "ldconfig|$entry"
    done < <(ldconfig -p 2>/dev/null | awk '/libamdhip64\.so/ {print $NF}')
  fi

  for entry in "$ROCM_PATH"/lib/libamdhip64.so*; do
    [[ -f "$entry" ]] && echo "$ROCM_PATH/lib|$entry"
  done
}

# Identity of a file for dedup purposes. Device and inode rather than the
# resolved path, because the same library is routinely installed as several
# hardlinked names (libamdhip64.so, .so.7, .so.7.x.y) that readlink cannot
# collapse: a hardlink has no target to follow.
file_identity() {
  stat -c '%d:%i' "$1" 2>/dev/null || readlink -f "$1" 2>/dev/null || echo "$1"
}

# Print the candidate table; return 0 when the runtime that will win has capture.
check_runtime() {
  local found_any=0 effective="" effective_src="" seen="" src path resolved status ident
  log "HIP runtimes visible from here, in load order:"
  while IFS='|' read -r src path; do
    [[ -n "$path" ]] || continue
    resolved="$(readlink -f "$path" 2>/dev/null || echo "$path")"
    ident="$(file_identity "$resolved")"
    case ":$seen:" in *":$ident:"*) continue ;; esac
    seen="$seen:$ident"
    found_any=1
    if lib_has_capture "$resolved"; then
      status="capture: yes"
    else
      status="capture: NO"
    fi
    printf '[capture]   %-16s %s (%s)\n' "$src" "$resolved" "$status" >&2
    if [[ -z "$effective" ]]; then
      effective="$resolved"
      effective_src="$src"
    fi
  done < <(candidate_libs)

  if (( ! found_any )); then
    log "no libamdhip64 found at all; this environment cannot run a HIP workload"
    return 1
  fi

  if lib_has_capture "$effective"; then
    log "the runtime that will load is $effective (via $effective_src) and it has capture"
    return 0
  fi

  log "the runtime that will load is $effective (via $effective_src) and it has NO capture"
  log "capture is compiled into libamdhip64, so a build without it ignores $CAPTURE_VAR"
  log "and leaves an empty archive behind with no error."
  if candidate_libs | while IFS='|' read -r _ path; do
       lib_has_capture "$path" && echo yes
     done | grep -q yes; then
    log "another libamdhip64 here does have capture: put it first with LD_PRELOAD."
  else
    log "use a ROCm build whose libamdhip64 has HRR compiled in, or preload one."
  fi
  return 1
}

# --- output path ------------------------------------------------------------

check_output_path() {
  local out="$1" min_free_gb="$2" parent fstype avail_kb avail_gb
  case "$out" in
    /*) : ;;
    *) log "output path is relative; the archive will land wherever the workload's"
       log "working directory happens to be. An absolute path is safer." ;;
  esac

  parent="$(dirname "$out")"
  mkdir -p "$parent" 2>/dev/null || fail "cannot create $parent"
  [[ -w "$parent" ]] || fail "not writable: $parent"

  fstype="$(stat -f -c %T "$parent" 2>/dev/null || echo unknown)"
  avail_kb="$(df -Pk "$parent" 2>/dev/null | awk 'NR==2 {print $4}')"
  avail_gb=$(( ${avail_kb:-0} / 1024 / 1024 ))
  log "output: $out"
  log "  filesystem: $fstype, free: ${avail_gb} GiB"

  case "$fstype" in
    overlayfs|overlay)
      log "  this path is on the container's own writable layer, so the archive dies"
      log "  with the container. Write to a bind-mounted host directory instead." ;;
    tmpfs)
      log "  this path is in RAM, so the archive competes with the workload for memory"
      log "  and is lost on reboot." ;;
  esac

  if (( avail_gb < min_free_gb )); then
    log "  only ${avail_gb} GiB free, and an archive runs to tens of GiB for a workload of"
    log "  any size. Capture writes until the disk fills, so point --output somewhere bigger."
    return 1
  fi
  return 0
}

# --- verbs ------------------------------------------------------------------

VERB="${1:-}"
[[ -n "$VERB" ]] || { usage; exit 1; }
shift || true
case "$VERB" in -h|--help|help) usage; exit 0 ;; esac

OUTPUT=""
PLAYBACK=""
SKIP_PREFLIGHT=0
FORCE=0
MIN_FREE_GB=20
PASSTHROUGH=()
CMD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --playback) PLAYBACK="${2:-}"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --force) FORCE=1; shift ;;
    --json) PASSTHROUGH+=(--json); shift ;;
    --no-playback) PASSTHROUGH+=(--no-playback); shift ;;
    --min-free-gb) MIN_FREE_GB="${2:-20}"; shift 2 ;;
    --) shift; CMD=("$@"); break ;;
    -*) fail "unknown option: $1" ;;
    *) CMD=("$@"); break ;;
  esac
done

case "$VERB" in
  preflight)
    rc=0
    check_runtime || rc=1
    if [[ -n "$OUTPUT" ]]; then
      check_output_path "$OUTPUT" "$MIN_FREE_GB" || rc=1
    fi
    if [[ -n "$PLAYBACK" ]]; then
      if [[ -x "$PLAYBACK" ]]; then
        log "hrr-playback given: $PLAYBACK"
      else
        log "the --playback path is not executable: $PLAYBACK"
        rc=1
      fi
    elif command -v hrr-playback >/dev/null 2>&1 || [[ -x "$ROCM_PATH/bin/hrr-playback" ]]; then
      log "hrr-playback is available, so the archive can be checked and replayed here"
    else
      log "no hrr-playback here; the archive can still be captured and checked from its"
      log "manifests, and replayed elsewhere on a matching build"
    fi
    exit "$rc"
    ;;

  run)
    [[ -n "$OUTPUT" ]] || fail "--output is required"
    [[ ${#CMD[@]} -gt 0 ]] || fail "no command given"
    if (( ! SKIP_PREFLIGHT )); then
      if ! check_runtime; then
        (( FORCE )) || fail "preflight failed: no capture-capable runtime, so nothing would be recorded. Pass --force to run anyway."
      fi
      if ! check_output_path "$OUTPUT" "$MIN_FREE_GB"; then
        (( FORCE )) || fail "preflight failed on the output path. Pass --force to run anyway."
      fi
    fi

    log "capturing to $OUTPUT"
    log "running: ${CMD[*]}"
    set +e
    HIP_HRR_CAPTURE_OUTPUT="$OUTPUT" "${CMD[@]}"
    workload_rc=$?
    set -e
    log "workload exited $workload_rc"
    if (( workload_rc != 0 )); then
      log "a non-zero exit is fine here: a workload that crashed is the one worth capturing"
    fi

    python3 "$INSPECT" --archive "$OUTPUT" ${PLAYBACK:+--playback "$PLAYBACK"} \
      "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" || true
    exit "$workload_rc"
    ;;

  verify)
    [[ -n "$OUTPUT" ]] || fail "--output is required"
    exec python3 "$INSPECT" --archive "$OUTPUT" ${PLAYBACK:+--playback "$PLAYBACK"} \
      "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
    ;;

  *)
    usage
    exit 1
    ;;
esac
