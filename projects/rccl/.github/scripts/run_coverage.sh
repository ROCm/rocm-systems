#!/usr/bin/env bash
# RCCL code-coverage launcher for a SLURM GPU cluster. Allocates GPU nodes, uses
# mnctl to launch a multi-node ROCm container on each, runs test_runner.py
# --coverage-report, and leaves the report in the test_runner-created
# rccl_test_artifacts_<RUN_ID>_<timestamp>/ dir (under the RCCL checkout).
# Standalone or from CI.
set -euo pipefail

# --- Resolve repo + mnctl locations from this script's path -----------------
SELF="$(readlink -f "$0")"
RCCL_DIR="${RCCL_DIR:-$(cd "$(dirname "$SELF")/../.." && pwd)}"          # projects/rccl
MNCTL_DIR="${MNCTL_DIR:-$(cd "$RCCL_DIR/../../rccl-utils/MultiNodeDocker" 2>/dev/null && pwd)}"
RCCL_TESTS_DIR="${RCCL_TESTS_DIR:-$(cd "$RCCL_DIR/../rccl-tests" 2>/dev/null && pwd)}"  # perf binaries

# --- Inputs (env with sensible defaults; overridable per cluster/target) ----
ROCM_IMAGE="${ROCM_IMAGE:-registry-sc-harbor.amd.com/framework/therock-release:47_gfx94X_7.13.0rc2_ubuntu24.04_py3.12_pytorch_release-2.11_96bfee1}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.Multinode.Ubuntu}"
GPU_ARCH="${GPU_ARCH:-gfx942}"
NIC_TYPE="${NIC_TYPE:-mellanox}"
NODES="${NODES:-2}"
TEST_CONFIG="${TEST_CONFIG:-mi300x_mellanox_ib.json}"
TEST_SUITE="${TEST_SUITE:-}"
TEST_NAME="${TEST_NAME:-}"

# SLURM knobs (defaults match the current cluster: partition/account 'rccl', 8 GPU/node).
PARTITION="${PARTITION:-rccl}"
ACCOUNT="${ACCOUNT:-rccl}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
RESERVATION="${RESERVATION:-}"
# ALLOC_MODE: auto=reuse a running alloc else salloc; new=always salloc; existing=require one; inherit=already inside.
ALLOC_MODE="${ALLOC_MODE:-auto}"

# Recognizable, filesystem-safe run id: arch + ROCm version + branch [+ PR]. Used as
# test_runner's --report-suffix, the per-run hostfile name, and the log prefix, so a
# run is recognizable at a glance and its dirs/logs correlate. NO timestamp here --
# test_runner's rccl_test_artifacts_<RUN_ID>_<timestamp> dir already carries one.
# Reused across the salloc re-exec; override RUN_ID to set it explicitly.
RCCL_BRANCH="${RCCL_BRANCH:-$(git -C "$RCCL_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)}"
PR_NUMBER="${PR_NUMBER:-}"
sanitize_id() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-' | sed -E 's/-+/-/g; s/^-|-$//g'; }
_rocm_ver="$(printf '%s' "$ROCM_IMAGE" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9]*' | head -n1 || true)"
RUN_ID="${RUN_ID:-$(sanitize_id "${GPU_ARCH}-${_rocm_ver:-rocm}-${RCCL_BRANCH:-nobranch}")${PR_NUMBER:+-pr$PR_NUMBER}}"

CONTAINER="${MNCTL_CONTAINER_NAME:-rccl-cov-$GPU_ARCH}"     # per-arch name -> isolated shared dir
HOSTFILE="${HOSTFILE:-$HOME/.mnctl/$RUN_ID.hostfile}"       # per-run so concurrent runs don't clash
# test_runner writes the exact workspace path it creates here (RCCL_ARTIFACTS_DIR_FILE);
# lives under $RCCL_DIR (bound to /work/rccl) so both container and login node see it.
ARTIFACTS_POINTER="$RCCL_DIR/.coverage_artifacts_${RUN_ID}"
# test_runner MPI hostfile: user override wins, else the dynamic mnctl hostfile below.
MPI_HOSTFILE="${RCCL_TEST_MPI_HOSTFILE:-$HOSTFILE}"

# mnctl reuses the cached image; FORCE_REBUILD=1 forces a clean image+deps rebuild.
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# Shared-FS caching for UCX/OpenMPI: shared/builds dirs on a real shared FS (NFS/GPFS/Lustre)
# so the leader builds once and all nodes + future runs reuse it. Nested per-container.
SHARED_FS_ROOT="${SHARED_FS_ROOT:-$HOME}"
SHARED_DIR="${SHARED_DIR:-$SHARED_FS_ROOT/.docker-shared/$CONTAINER}"
BUILDS_DIR="${BUILDS_DIR:-$SHARED_FS_ROOT/.docker-builds/$CONTAINER}"

export ROCM_IMAGE DOCKERFILE GPU_ARCH NIC_TYPE NODES TEST_CONFIG TEST_SUITE TEST_NAME \
       RUN_ID RCCL_BRANCH PR_NUMBER PARTITION ACCOUNT GPUS_PER_NODE TIME_LIMIT RESERVATION CONTAINER HOSTFILE \
       MPI_HOSTFILE RCCL_DIR MNCTL_DIR RCCL_TESTS_DIR FORCE_REBUILD SHARED_FS_ROOT SHARED_DIR BUILDS_DIR \
       ARTIFACTS_POINTER

[[ -n "$RCCL_TESTS_DIR" && -d "$RCCL_TESTS_DIR" ]] || {
  echo "[run_coverage] ERROR: rccl-tests source not found. Set RCCL_TESTS_DIR to a checkout" >&2
  echo "               (expected sibling projects/rccl-tests)." >&2
  exit 1
}

[[ -n "$MNCTL_DIR" && -d "$MNCTL_DIR/mnctl" ]] || {
  echo "[run_coverage] ERROR: mnctl not found. Set MNCTL_DIR to a rccl-utils/MultiNodeDocker" >&2
  echo "               checkout (expected sibling rccl-utils/MultiNodeDocker)." >&2
  exit 1
}

log() { echo "[run_coverage $RUN_ID] $*"; }

# --- If asked for a fresh allocation, re-exec self inside salloc (then 'inherit') ---
if [[ "$ALLOC_MODE" == "new" || ( "$ALLOC_MODE" == "auto" && -z "${SLURM_JOB_NODELIST:-}" && -z "$(squeue -u "$USER" -t R -h -o '%i' 2>/dev/null | head -n1)" ) ]]; then
  log "Allocating $NODES node(s) on partition=$PARTITION account=$ACCOUNT ..."
  exec salloc -N "$NODES" -p "$PARTITION" -A "$ACCOUNT" \
       --gres=gpu:"$GPUS_PER_NODE" --ntasks-per-node="$GPUS_PER_NODE" \
       ${RESERVATION:+--reservation="$RESERVATION"} -t "$TIME_LIMIT" \
       bash -c "ALLOC_MODE=inherit RUN_ID='$RUN_ID' '$SELF'"
fi

# --- Resolve the node list --------------------------------------------------
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  NODELIST="$SLURM_JOB_NODELIST"                                 # inside salloc / inherit
else
  # existing/auto: take the newest RUNNING job for this user (SyncBench pattern).
  read -r JOBID NODELIST < <(squeue -u "$USER" -t R -h -o "%i %N" | sort -nr | head -n1) || true
  [[ -n "${NODELIST:-}" ]] || { log "ERROR: no running SLURM allocation found"; exit 1; }
  log "Attaching to existing job $JOBID"
fi

mapfile -t HOSTS < <(scontrol show hostnames "$NODELIST")
HEAD="${HOSTS[0]}"
log "Nodes: ${HOSTS[*]}  (head=$HEAD)"

# --- Write the MPI hostfile (must be on shared FS so all nodes/containers see it) ---
# NB: the results dir is NOT created here -- test_runner creates it during the run.
mkdir -p "$SHARED_DIR" "$BUILDS_DIR" "$(dirname "$HOSTFILE")"
: > "$HOSTFILE"
for h in "${HOSTS[@]}"; do echo "$h slots=$GPUS_PER_NODE" >> "$HOSTFILE"; done
log "Run ID: $RUN_ID"
log "Hostfile: $HOSTFILE"; cat "$HOSTFILE"

# A user-supplied MPI hostfile must exist; mnctl already mounts the dynamic one.
[[ "$MPI_HOSTFILE" == "$HOSTFILE" || -f "$MPI_HOSTFILE" ]] || {
  log "ERROR: RCCL_TEST_MPI_HOSTFILE not found: $MPI_HOSTFILE"; exit 1
}

# Force a clean image + deps rebuild only when explicitly requested.
REBUILD_FLAG=""
[[ "$FORCE_REBUILD" == "1" ]] && REBUILD_FLAG="--rebuild"

# --- Optional gtest filters -------------------------------------------------
# --suite-name globs the DISPLAY name (e.g. "UBR Tests - Multi Node"), not the config
# key (e.g. "ubr_multi_node"). If TEST_SUITE is a config key, translate it to the
# matching display name(s) (':'-joined = OR); otherwise pass through as a glob.
CFG_PATH="$RCCL_DIR/tools/scripts/test_runner/configs/$TEST_CONFIG"
if [[ -n "$TEST_SUITE" && -f "$CFG_PATH" ]]; then
  mapped="$(python3 - "$CFG_PATH" "$TEST_SUITE" <<'PY'
import json, sys
cfg_path, key = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    data = json.load(f)
print(":".join(s["name"] for s in data.get("test_suites", [])
                if s.get("config") == key and "name" in s))
PY
)"
  if [[ -n "$mapped" ]]; then
    log "Suite filter '$TEST_SUITE' (config key) -> display name(s) '$mapped'"
    TEST_SUITE="$mapped"
  fi
fi

# --- Make results host-readable for the runner/artifact upload (idempotent) --
# test_runner creates /work/rccl/rccl_test_artifacts_<RUN_ID>_<timestamp>/ (report/ +
# logs/ + results/) as root during the run -- it may not exist yet if we failed early.
# It lives under $RCCL_DIR (a shared-FS bind mount the runner already sees), so no
# copy-back is needed: just chown it (in-container, as root) so the non-root runner
# can read/upload it. Runs before teardown on both the success and failure paths.
COLLECTED=0
collect_results() {
  [[ "$COLLECTED" == "1" ]] && return 0
  COLLECTED=1
  # Resolve THIS run's artifacts dir. test_runner wrote the exact path it created to
  # $ARTIFACTS_POINTER (RCCL_ARTIFACTS_DIR_FILE); read its basename for a deterministic,
  # glob-free match. Fall back to a RUN_ID-scoped find if the pointer is missing (e.g.
  # the run failed before setup_directories, or an older test_runner without the export).
  local art=""
  if [[ -s "$ARTIFACTS_POINTER" ]]; then
    art="$(basename "$(head -n1 "$ARTIFACTS_POINTER")")"
  fi
  if [[ -z "$art" ]]; then
    # trailing _<timestamp> makes lexical order == chronological, so sort|tail is newest.
    art="$(find "$RCCL_DIR" -maxdepth 1 -type d -name "rccl_test_artifacts_${RUN_ID}_*" \
             -printf '%f\n' 2>/dev/null | sort | tail -n1)"
  fi
  if [[ -z "$art" ]]; then
    log "WARNING: no artifacts dir found (run may have failed before creating it)."
    return 0
  fi
  local uid gid; uid="$(id -u)"; gid="$(id -g)"
  log "Fixing artifacts ownership to $uid:$gid in the container ..."
  ssh -o StrictHostKeyChecking=no "$HEAD" \
    "docker exec -e U=$uid -e G=$gid -e ART='$art' '$CONTAINER' bash -lc '
       chown -R \${U}:\${G} \"/work/rccl/\$ART\"'" \
    >/dev/null 2>&1 || true
  log "Coverage artifacts: $RCCL_DIR/$art"
  [[ -n "${GITHUB_ENV:-}" ]] && echo "COVERAGE_ARTIFACT_DIR=$art" >> "$GITHUB_ENV"
}

# --- Idempotent teardown: collect results first, then stop containers --------
# Guarded so it never runs twice; the per-node `docker rm -f` is a safety net.
CLEANED=0
cleanup() {
  [[ "$CLEANED" == "1" ]] && return 0
  CLEANED=1
  collect_results
  log "Tearing down containers ..."
  ssh -o StrictHostKeyChecking=no "$HEAD" \
    "cd '$MNCTL_DIR' && python3 -m mnctl --stop-all --hostfile '$HOSTFILE'" || true
  for h in "${HOSTS[@]}"; do
    ssh -o StrictHostKeyChecking=no "$h" "docker rm -f '$CONTAINER'" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

# --- Optional non-interactive registry login on every node ------------------
# Set REGISTRY_USER + REGISTRY_TOKEN (Harbor CLI secret / robot token) for a TTY-less
# login; token is piped via stdin so it never lands in argv/ps. Unset = already logged in.
REGISTRY_HOST="${REGISTRY_HOST:-${ROCM_IMAGE%%/*}}"
REGISTRY_USER="${REGISTRY_USER:-}"
REGISTRY_TOKEN="${REGISTRY_TOKEN:-}"
if [[ -n "$REGISTRY_USER" && -n "$REGISTRY_TOKEN" ]]; then
  log "Logging in to $REGISTRY_HOST as $REGISTRY_USER on all nodes ..."
  for h in "${HOSTS[@]}"; do
    if ! echo "$REGISTRY_TOKEN" | ssh -o StrictHostKeyChecking=no "$h" \
        "docker login '$REGISTRY_HOST' -u '$REGISTRY_USER' --password-stdin" >/dev/null 2>&1; then
      log "ERROR: docker login failed on $h"; exit 1
    fi
  done
fi

# --- Fail fast on registry auth: pull the base image on every node up front ---
# Cheaper than failing after a long build; also warms the cache for `docker build FROM`.
log "Preflight: verifying registry access to $ROCM_IMAGE on all nodes ..."
auth_fail=0
for h in "${HOSTS[@]}"; do
  if ! out=$(ssh -o StrictHostKeyChecking=no "$h" "docker pull '$ROCM_IMAGE'" 2>&1); then
    log "ERROR: image pull failed on $h:"; echo "$out" | tail -n 5
    auth_fail=1
  fi
done
if [[ "$auth_fail" == "1" ]]; then
  log "Registry access failed. Run 'docker login ${ROCM_IMAGE%%/*}' on the affected node(s) and retry."
  exit 1
fi
log "Preflight OK: base image reachable on all nodes."

# --- Ensure mnctl is visible on the compute nodes (self-contained) ----------
# The runner workspace holding MNCTL_DIR may be node-local; if the head can't see it,
# stage the source onto shared FS and run mnctl from there. No-op when already visible.
if ! ssh -o StrictHostKeyChecking=no "$HEAD" "test -d '$MNCTL_DIR/mnctl'"; then
  staged="$SHARED_FS_ROOT/.mnctl-src/MultiNodeDocker"
  log "mnctl not visible on $HEAD; staging $MNCTL_DIR -> $staged (shared FS) ..."
  mkdir -p "$(dirname "$staged")"
  rsync -a --delete "$MNCTL_DIR/" "$staged/"
  MNCTL_DIR="$staged"
fi

# --- Launch containers on all nodes (mnctl runs from the head compute node) --
# The head node has docker+GPUs and SSHes to peers; shared-dir/builds-dir on shared FS
# build UCX/OpenMPI once and reuse. mnctl auto-mounts $HOSTFILE; add a user one if set.
MPI_HOSTFILE_MOUNT=""
[[ "$MPI_HOSTFILE" != "$HOSTFILE" ]] && MPI_HOSTFILE_MOUNT="--volume '$MPI_HOSTFILE:$MPI_HOSTFILE:ro'"

log "Launching containers via mnctl ..."
ssh -o StrictHostKeyChecking=no "$HEAD" "
  cd '$MNCTL_DIR' &&
  python3 -m mnctl --launch-all --ssh --hostfile '$HOSTFILE' \
    --rocm-image '$ROCM_IMAGE' --dockerfile '$DOCKERFILE' \
    --gpu-targets '$GPU_ARCH' --nic-type '$NIC_TYPE' --name '$CONTAINER' \
    --shared-dir '$SHARED_DIR' --builds-dir '$BUILDS_DIR' --shared-fs yes \
    $REBUILD_FLAG \
    --volume '$RCCL_DIR:/work/rccl' --volume '$RCCL_TESTS_DIR:/work/rccl-tests' \
    $MPI_HOSTFILE_MOUNT
"

# --- Clean ONLY a stale (path-mismatched) build tree (in-container, as root) --
# Wipe build/ only when its baked CMAKE_HOME_DIRECTORY != the container mount, so
# compatible re-runs stay incremental. Root artifacts can't be removed from the host.
log "Checking for stale CMake cache inside the container ..."
ssh -o StrictHostKeyChecking=no "$HEAD" "docker exec -i '$CONTAINER' bash -s" <<'REMOTE'
set -euo pipefail
for src in /work/rccl /work/rccl-tests; do
  [ -d "$src/build" ] || continue
  stale=""
  while IFS= read -r cache; do
    home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache" | head -n1)"
    if [ -n "$home" ] && [ "$home" != "$src" ]; then stale="$home"; break; fi
  done < <(find "$src/build" -name CMakeCache.txt 2>/dev/null)
  if [ -n "$stale" ]; then
    echo "[cleanup] WARNING: stale CMake cache in $src/build (configured for '$stale', expected '$src'); removing to avoid a path conflict."
    rm -rf "$src/build"
  else
    echo "[cleanup] $src/build is compatible; keeping it (incremental build)."
  fi
done
REMOTE

# --- Run coverage inside the head container (mpirun fans out to peers) -------
# Filters + report suffix pass via `docker exec -e` and the arg array is rebuilt
# in-container, so values with spaces survive the ssh/docker/bash quoting layers.
# --report-suffix=$RUN_ID names the rccl_test_artifacts_<RUN_ID>_<timestamp> dir;
# no --results-dir, so emitted results default into that same dir (self-contained).
log "Running test_runner.py --coverage-report ..."
rm -f "$ARTIFACTS_POINTER"   # drop any stale pointer from a prior same-RUN_ID run
ssh -o StrictHostKeyChecking=no "$HEAD" "
  docker exec \
    -e SUITE='$TEST_SUITE' -e TNAME='$TEST_NAME' \
    -e TCFG='$TEST_CONFIG' -e MHOST='$MPI_HOSTFILE' -e RSUFFIX='$RUN_ID' \
    -e RCCL_ARTIFACTS_DIR_FILE='/work/rccl/.coverage_artifacts_${RUN_ID}' \
    '$CONTAINER' bash -lc '
      cd /work/rccl
      filters=()
      [ -n \"\$SUITE\" ] && filters+=(--suite-name \"\$SUITE\")
      [ -n \"\$TNAME\" ] && filters+=(--test-name \"\$TNAME\")
      ROCM_PATH=/opt/rocm MPI_PATH=/opt/shared/ompi RCCL_TESTS_DIR=/work/rccl-tests \
      RCCL_TEST_MPI_HOSTFILE=\"\$MHOST\" \
      python3 tools/scripts/test_runner/test_runner.py \
        --config tools/scripts/test_runner/configs/\"\$TCFG\" \
        \"\${filters[@]}\" --report-suffix \"\$RSUFFIX\" \
        --coverage-report --verbose --emit-results
  '
"

collect_results

log "Coverage artifacts under $RCCL_DIR/rccl_test_artifacts_${RUN_ID}_*"
