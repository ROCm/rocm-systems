#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Turn HRR capture on for a workload, then say whether it actually recorded
# anything. Sole agent entry point for this skill.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSPECT="$SCRIPT_DIR/inspect_archive.py"
# The sibling skill, which replays an archive and says what failed. Also where
# the only script that finds or builds a matching hrr-playback lives.
TRIAGE="$SCRIPT_DIR/../../decode-and-triage/scripts/triage_archive.sh"
ENSURE_PLAYBACK="$SCRIPT_DIR/../../decode-and-triage/scripts/ensure_playback.sh"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# The environment variable libamdhip64 reads to enable capture. Its presence as
# a literal in a shared object distinguishes a capture-capable runtime from one
# built without HRR.
CAPTURE_VAR="HIP_HRR_CAPTURE_OUTPUT"

# Set by check_runtime when it could not establish which runtime will bind, so
# `run` can repeat the caveat at the moment it matters.
RUNTIME_UNRESOLVED=0

usage() {
  cat <<'EOF' >&2
usage:
  hrr_capture.sh preflight [--output PATH] [--playback PATH] [-- <command> [args...]]
  hrr_capture.sh run --output PATH [--playback PATH] [--] <command> [args...]
  hrr_capture.sh verify --output PATH [--playback PATH] [--json] [--no-playback]

  preflight  Check this environment can capture: which HIP runtime the workload
             will load, whether it has capture built in, and whether the output
             path is big enough and outlives the container. Give it the command
             after --: a binary's own RPATH beats every path a search can see,
             so without the command this answers a weaker question.
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
  --min-free-gb N      Minimum free space to insist on (default 50)
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

# Only the names a linked binary can ask for. A build or backup left beside the
# real thing, `libamdhip64.so.hrr-orig` and the like, is not a runtime anything
# loads, and treating it as one has already produced a wrong verdict.
is_runtime_name() {
  # Anchored: `libamdhip64.so.7.hrr-orig` and `libamdhip64.so.7.bak` match a
  # `.so.[0-9]*` glob and are not runtimes anything loads.
  [[ "${1##*/}" =~ ^libamdhip64\.so(\.[0-9]+)*$ ]]
}

# Runtimes in one directory, soname first. `libamdhip64.so` is a development
# alias: nothing links it, everything links the soname `libamdhip64.so.N`, so
# the alias must never be the one a verdict is pronounced on. They are normally
# hardlinks to one file, and the order only matters when something has broken
# that, which is exactly when the verdict was wrong before.
emit_libs() {
  local src="$1" dir="$2" entry
  for entry in "$dir"/libamdhip64.so.[0-9]*; do
    [[ -f "$entry" ]] && is_runtime_name "$entry" && echo "$src|$entry"
  done
  [[ -f "$dir/libamdhip64.so" ]] && echo "$src|$dir/libamdhip64.so"
  return 0
}

# The interpreter whose packages matter. Named directly for `python train.py`,
# but the workloads this skill is aimed at are console scripts: `vllm`,
# `torchrun`, `accelerate`, `pytest`. Those are text files whose shebang names
# the interpreter of their own virtual environment, and asking the PATH's
# python3 instead reports the wrong site-packages, or none at all.
workload_interpreter() {
  local bin="${WORKLOAD_BIN:-}" shebang
  case "${bin##*/}" in
    python|python[0-9]*) echo "$bin"; return 0 ;;
  esac
  if [[ -f "$bin" ]] && read -r shebang < "$bin" 2>/dev/null; then
    case "$shebang" in
      '#!'*python*)
        shebang="${shebang#\#!}"
        set -- $shebang
        # `#!/usr/bin/env python3` names the interpreter in the second word.
        [[ "${1##*/}" == "env" ]] && shift
        [[ -n "${1:-}" ]] && { echo "$1"; return 0; }
        ;;
    esac
  fi
  command -v python3 2>/dev/null || echo python3
}

# Launchers that wrap the real workload. The binary that matters is the one
# after them, otherwise preflight resolves `srun` or `env` and learns nothing.
strip_launchers() {
  while (( $# )); do
    case "${1##*/}" in
      env)
        shift
        # Skip VAR=value assignments that env takes before the command.
        while (( $# )) && [[ "$1" == *=* ]]; do shift; done ;;
      timeout|stdbuf|nohup|nice|ionice)
        shift
        while (( $# )) && [[ "$1" == -* ]]; do shift; done ;;
      mpirun|mpiexec|srun|torchrun|accelerate|deepspeed)
        # Their own options and the command are not separable without knowing
        # each launcher's grammar, so stop here and say so rather than guess.
        printf '%s\n' "$1"; return 0 ;;
      *) printf '%s\n' "$1"; return 0 ;;
    esac
  done
  return 0
}

# Directories on the binary's own DT_RPATH or DT_RUNPATH, with $ORIGIN expanded,
# for whichever of the two tags is asked for. The order matters and the two are
# not interchangeable: DT_RPATH is searched before LD_LIBRARY_PATH, DT_RUNPATH
# after it, and nothing else listed here can see either.
binary_rpath_dirs() {
  local bin="$1" tag="${2:-RPATH|RUNPATH}" origin list dir
  [[ -n "$bin" && -f "$bin" ]] || return 0
  command -v readelf >/dev/null 2>&1 || return 0
  origin="$(dirname "$(readlink -f "$bin")")"
  while read -r list; do
    while IFS= read -r dir; do
      [[ -n "$dir" ]] || continue
      dir="${dir//\$\{ORIGIN\}/$origin}"
      echo "${dir//\$ORIGIN/$origin}"
    done < <(tr ':' '\n' <<<"$list")
  done < <(readelf -d "$bin" 2>/dev/null |
             awk -v tag="$tag" '$0 ~ "\\((" tag ")\\)" { match($0, /\[.*\]/); print substr($0, RSTART + 1, RLENGTH - 2) }')
}

# The runtime the loader will actually bind for this binary. This is the only
# answer that accounts for DT_RPATH, and it is why the list below is a listing
# rather than a verdict. Empty when the binary links no HIP runtime of its own,
# which is the normal case for an interpreter: Python loads the runtime later,
# out of whatever package asks for it, where no static inspection can see it.
resolved_runtime() {
  local bin="$1" path
  [[ -n "$bin" && -f "$bin" ]] || return 1
  command -v ldd >/dev/null 2>&1 || return 1
  path="$(ldd "$bin" 2>/dev/null | awk '/libamdhip64\.so/ { print $3; exit }')"
  [[ -n "$path" && -f "$path" ]] || return 1
  readlink -f "$path"
}

# Candidate libamdhip64 paths, most-likely-to-win first.
#
# Precedence follows the dynamic loader: LD_PRELOAD beats everything, then the
# binary's own DT_RPATH, then LD_LIBRARY_PATH, then its DT_RUNPATH, then a
# runtime bundled with the application itself, then the ldconfig cache, then
# the ROCm install. Bundled
# copies are worth listing because an application that ships its own runtime
# uses that one, whatever is installed on the system.
candidate_libs() {
  local entry dir pkg_dir
  local IFS_SAVE="$IFS"

  IFS=': '
  for entry in ${LD_PRELOAD:-}; do
    case "$entry" in *libamdhip64*) [[ -f "$entry" ]] && echo "LD_PRELOAD|$entry" ;; esac
  done
  IFS="$IFS_SAVE"

  while read -r dir; do
    [[ -d "$dir" ]] && emit_libs "workload RPATH" "$dir"
  done < <(binary_rpath_dirs "${WORKLOAD_BIN:-}" RPATH)

  IFS=': '
  for dir in ${LD_LIBRARY_PATH:-}; do
    [[ -d "$dir" ]] && emit_libs LD_LIBRARY_PATH "$dir"
  done
  IFS="$IFS_SAVE"

  # After LD_LIBRARY_PATH, which is where the loader looks for it. DT_RUNPATH
  # is the modern linker's default, so listing it as if it were DT_RPATH put
  # the wrong candidate first.
  while read -r dir; do
    [[ -d "$dir" ]] && emit_libs "workload RUNPATH" "$dir"
  done < <(binary_rpath_dirs "${WORKLOAD_BIN:-}" RUNPATH)

  # Runtimes bundled inside installed packages. Listed generically rather than
  # by package name: any package may ship one, and the site directories are
  # asked of the interpreter rather than guessed.
  while read -r pkg_dir; do
    [[ -n "$pkg_dir" && -d "$pkg_dir" ]] || continue
    while read -r dir; do
      [[ -n "$dir" ]] && emit_libs "bundled package" "$dir"
    done < <(find "$pkg_dir" -maxdepth 3 -name 'libamdhip64.so*' -type f -printf '%h\n' 2>/dev/null | sort -u)
    # The interpreter that will run the workload, not the one on PATH. A job
    # launched as /opt/venv/bin/python has different site directories from the
    # shell's python3, and asking the wrong one hides the wheel that matters.
  done < <("$(workload_interpreter)" - <<'PY' 2>/dev/null || true
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
      [[ -f "$entry" ]] && is_runtime_name "$entry" && echo "ldconfig|$entry"
    done < <(ldconfig -p 2>/dev/null | awk '/libamdhip64\.so/ {print $NF}')
  fi

  emit_libs "$ROCM_PATH/lib" "$ROCM_PATH/lib"
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
  local found_any=0 resolved_ok=0 effective="" effective_src="" seen="" src path resolved status ident bound
  local no_capture_seen=""
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
      [[ -n "$no_capture_seen" ]] || no_capture_seen="$resolved"
    fi
    printf '[capture]   %-16s %s (%s)\n' "$src" "$resolved" "$status" >&2
    if [[ -z "$effective" ]]; then
      effective="$resolved"
      effective_src="$src"
    fi
  done < <(candidate_libs)

  # The listing above is what is visible from here. When a command was given and
  # it links a HIP runtime itself, the loader's own answer replaces the guess.
  if bound="$(resolved_runtime "${WORKLOAD_BIN:-}")"; then
    effective="$bound"
    effective_src="bound by ${WORKLOAD_BIN##*/}"
    resolved_ok=1
    found_any=1
    lib_has_capture "$effective" && status="capture: yes" || status="capture: NO"
    printf '[capture]   %-16s %s (%s)\n' "resolved" "$effective" "$status" >&2
  elif [[ -n "${WORKLOAD_BIN:-}" ]]; then
    log "note: ${WORKLOAD_BIN##*/} links no HIP runtime of its own, so the one it ends up"
    log "using is loaded later, from wherever the application asks for it. For a Python"
    log "workload that is the runtime inside the installed package, listed above."
  else
    log "note: no command given, so this is what is visible here rather than what a"
    log "workload will load. Pass the command after -- to have it resolved properly."
  fi

  if (( ! found_any )); then
    log "no libamdhip64 found at all; this environment cannot run a HIP workload"
    return 1
  fi

  if lib_has_capture "$effective"; then
    if (( resolved_ok )); then
      log "the runtime that will load is $effective (via $effective_src) and it has capture"
      return 0
    fi
    # Unresolved and the best guess looks fine. Saying "you are ready" here is
    # how a Python workload gets a clean preflight and an empty capture: the
    # wheel's own runtime is bound later and is not this one.
    RUNTIME_UNRESOLVED=1
    if [[ -n "$no_capture_seen" ]]; then
      # The PyTorch case exactly: binding cannot be established, and a runtime
      # that cannot capture is sitting in the load path. Passing here is how a
      # clean preflight turns into an empty archive an hour later.
      log "UNRESOLVED, and refusing: nothing here was resolved, and $no_capture_seen has no"
      log "capture. If the workload binds that one, which a wheel or a plugin commonly does,"
      log "nothing will be recorded. Install a capture-capable runtime where the workload"
      log "will find it, or pass --force to capture anyway and check the verdict afterwards."
      return 1
    fi
    log "UNRESOLVED: nothing here was resolved, and the first candidate ($effective) has"
    log "capture. That is a guess, not a verdict. Every runtime visible from here can"
    log "capture, so the guess is a safe one, but the workload may still bind something"
    log "none of these searches can see."
    return 0
  fi

  log "the runtime that will load is $effective (via $effective_src) and it has NO capture"
  log "capture is compiled into libamdhip64, so a build without it ignores $CAPTURE_VAR"
  log "and no archive is created at all: verify then reports 'no archive'."
  # Collected rather than piped into `grep -q`: under `pipefail` an early-closing
  # reader makes the whole pipeline fail, and the advice below would then flip to
  # the wrong branch precisely when a usable runtime does exist.
  local alternatives
  alternatives="$(candidate_libs | while IFS='|' read -r _ path; do
                    lib_has_capture "$path" && echo yes
                  done)"
  if [[ -n "$alternatives" ]]; then
    log "another libamdhip64 here does have capture. For a plain binary, LD_PRELOAD puts"
    log "it in front. For a framework that loads its own runtime through an RPATH, such"
    log "as a PyTorch or vLLM wheel, LD_PRELOAD does not work: install the capture-capable"
    log "runtime into the package's own lib directory instead, with its matching"
    log "libhsa-runtime64. See references/workload-shapes.md."
  else
    log "use a ROCm build whose libamdhip64 has HRR compiled in."
  fi
  return 1
}

# --- output path ------------------------------------------------------------

# Filesystem type of the mount a path sits on, from /proc/self/mountinfo, which
# knows about bind mounts where `stat -f` on a parent directory does not. Falls
# back to stat where mountinfo is unavailable.
path_fstype() {
  local target="$1" resolved
  resolved="$(readlink -f "$target" 2>/dev/null || echo "$target")"
  if [[ -r /proc/self/mountinfo ]]; then
    awk -v path="$resolved" '
      {
        mount_point = $5
        for (i = 6; i <= NF; i++) if ($i == "-") { fstype = $(i + 1); break }
        if (path == mount_point || index(path, mount_point == "/" ? "/" : mount_point "/") == 1) {
          if (length(mount_point) >= best_len) { best_len = length(mount_point); best = fstype }
        }
      }
      END { if (best != "") print best }
    ' /proc/self/mountinfo && return 0
  fi
  stat -f -c %T "$resolved" 2>/dev/null || echo unknown
}

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

  # The filesystem of the output path itself, resolved through its own mount
  # rather than its parent's. With `-v /host/captures:/data/captures` the
  # parent `/data` is on the container's overlay while the archive's directory
  # is the bind mount, and keying on the parent refused a perfectly good path.
  fstype="$(path_fstype "$parent")"
  avail_kb="$(df -Pk "$parent" 2>/dev/null | awk 'NR==2 {print $4}')"
  avail_gb=$(( ${avail_kb:-0} / 1024 / 1024 ))
  log "output: $out"
  log "  filesystem: $fstype, free: ${avail_gb} GiB"

  case "$fstype" in
    overlayfs|overlay)
      log "  this path is on the container's own writable layer, so the archive dies with"
      log "  the container and there is nothing left to send. Write to a bind-mounted host"
      log "  directory instead, or pass --force to capture there anyway."
      return 1 ;;
    tmpfs)
      log "  this path is in RAM, so the archive competes with the workload for memory"
      log "  and is lost on reboot." ;;
  esac

  if (( avail_gb < min_free_gb )); then
    log "  ${avail_gb} GiB free against the ${min_free_gb} GiB this asks for. An archive runs to"
    log "  tens of GiB for a workload of any size and capture writes until the disk fills, so"
    log "  point --output somewhere bigger, or lower the bar with --min-free-gb for a short run."
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
MIN_FREE_GB=50
PASSTHROUGH=()
CMD=()

# A flag whose value is missing used to fall off the end of `shift 2` and exit 1
# with nothing printed, and 1 is also verify's "nothing captured" code, so a
# scripted gate could not tell a typo from an empty archive.
needs_value() {
  (( $2 >= 2 )) || fail "$1 needs a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --output) needs_value --output $#; OUTPUT="$2"; shift 2 ;;
    --playback) needs_value --playback $#; PLAYBACK="$2"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --force) FORCE=1; shift ;;
    --json) PASSTHROUGH+=(--json); shift ;;
    --no-playback) PASSTHROUGH+=(--no-playback); shift ;;
    --min-free-gb)
      needs_value --min-free-gb $#
      [[ "$2" =~ ^[0-9]+$ ]] || fail "--min-free-gb takes a whole number of GiB, not '$2'"
      MIN_FREE_GB="$2"; shift 2 ;;
    --) shift; CMD=("$@"); break ;;
    *) fail "unexpected argument: $1. The workload command goes after --" ;;
  esac
done

# The executable the command will run, when there is one. Preflight answers a
# different and much weaker question without it.
WORKLOAD_BIN=""
if [[ ${#CMD[@]} -gt 0 ]]; then
  WORKLOAD_NAME="$(strip_launchers "${CMD[@]}")"
  WORKLOAD_BIN="$(command -v -- "${WORKLOAD_NAME:-${CMD[0]}}" 2>/dev/null || true)"
fi

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
      log "no hrr-playback here. The archive can still be captured and checked from its"
      log "manifests alone with 'verify --no-playback', and replayed elsewhere on a"
      log "matching build. To get one that matches this machine: $ENSURE_PLAYBACK"
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

    # Capture resumes into an archive that is already there and leaves the
    # other pid-* directories where they are, so a second run against the same
    # --output silently merges two attempts into what reads afterwards as one
    # multi-process capture.
    if compgen -G "$OUTPUT/pid-*" >/dev/null 2>&1; then
      if (( FORCE )); then
        log "$OUTPUT already holds a capture and --force was given: this run will be added"
        log "to it, and the result will read as one multi-process archive."
      else
        fail "$OUTPUT already holds a capture. Capture would add this run to it and the two would be indistinguishable afterwards. Move it aside, or give --output a new path."
      fi
    fi

    if (( RUNTIME_UNRESOLVED )); then
      log "note: preflight could not resolve which runtime this workload binds, so this run"
      log "may record nothing. The verdict afterwards is what tells you, not the exit status."
    fi
    log "capturing to $OUTPUT"
    log "running: ${CMD[*]}"
    # Backgrounded and waited on, with the signals forwarded, so that a stop
    # aimed at this script reaches the workload. Left in the foreground, a
    # SIGTERM from timeout, systemd, a pod stop or a cancelled CI job killed
    # the wrapper only: the workload carried on capturing and nothing verified
    # the archive.
    workload_pid=0
    forward_signal() {
      local sig="$1"
      (( workload_pid )) && kill -"$sig" "$workload_pid" 2>/dev/null || true
    }
    trap 'forward_signal TERM' TERM
    trap 'forward_signal INT' INT
    trap 'forward_signal HUP' HUP

    set +e
    HIP_HRR_CAPTURE_OUTPUT="$OUTPUT" "${CMD[@]}" &
    workload_pid=$!
    wait "$workload_pid"
    workload_rc=$?
    set -e
    trap - TERM INT HUP
    log "workload exited $workload_rc"
    if (( workload_rc != 0 )); then
      log "a non-zero exit is fine here: a workload that crashed is the one worth capturing"
    fi

    inspect_rc=0
    python3 "$INSPECT" --archive "$OUTPUT" ${PLAYBACK:+--playback "$PLAYBACK"} \
      "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" || inspect_rc=$?

    # Offer the replay here, while the machine that captured is still the
    # machine in front of you. Its runtime is by definition the one the reader
    # has to match, so the pairing problem that dogs every later attempt does
    # not exist at this moment.
    if (( inspect_rc == 0 )); then
      log "the archive holds events. To find out what went wrong, replay it here and now:"
      log "  $TRIAGE --archive $OUTPUT/pid-<pid>"
      log "this host captured it, so its runtime is the one the reader has to match."
    fi
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
