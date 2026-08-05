#!/usr/bin/env bash
#
# run_corpus.sh — run the rocjitsu-corpus gfx1250 regression through mirage.
#
# The corpus (https://github.com/kuhar/rocjitsu-corpus) ships a runner,
# `scripts/run_gfx1250_regression.sh`, that sources an env file and then
# executes the IREE tools (`iree-check-module`, `iree-e2e-matmul-test`) over a
# packaged set of gfx1250 VMFBs.
#
# This bridge runs the *same* corpus but routes every IREE tool invocation
# through `mirage run --profile <profile> -- <tool> ...`, so mirage injects the
# emulator env contract (HotSwap by default). It supports two modes:
#
#   * host mode (default): the IREE tools come from the host PATH and run
#     under a non-containerised mirage profile.
#   * container mode (when MIRAGE_CORPUS_IMAGE is set): the IREE tools come
#     from inside the image (see tests/corpus/Dockerfile) and the corpus runs
#     under a *containerised* mirage profile built around that image. The
#     corpus checkout is bind-mounted into the container at its identical host
#     path so the packaged VMFB paths resolve unchanged.
#
# Either way the corpus checkout is never modified: the bridge generates tiny
# wrapper executables for the IREE tools and points the corpus runner's IREE_*
# env vars at them.
#
# Inputs (all via environment):
#   MIRAGE_BIN               mirage binary (default: mirage on PATH)
#   CORPUS_ROOT              rocjitsu-corpus checkout (required)
#   MIRAGE_CORPUS_EMULATOR   emulator to run under (default: hotswap)
#   MIRAGE_CORPUS_PROFILE    mirage profile name (default: corpus-<emulator>[-img])
#   MIRAGE_CORPUS_ENV_FILE   corpus env file (default: <script>/env/<emulator>.sh)
#   MIRAGE_CORPUS_ONLY       all | e2e | matmul (default: all)
#   MIRAGE_CORPUS_OUT_DIR    results/logs output dir (default: corpus runner default)
#   MIRAGE_CORPUS_IMAGE      container image with the IREE tools -> enables
#                            container mode (default: unset -> host mode)
#   MIRAGE_CORPUS_PROVIDER   container provider for container mode (auto-detect)
#   MIRAGE_CORPUS_AGENT      builtin GPU agent the profile pins
#                            (default: MI450X for hotswap, MI300X otherwise)
#
# Exit codes:
#   0    all selected corpus cases passed
#   1    one or more corpus cases failed
#   2    usage / configuration error
#   77   prerequisites missing -> ctest reports the test as SKIPPED
#
set -uo pipefail

SKIP=77

note() { echo "corpus: $*" >&2; }
skip() { echo "skipping: $*" >&2; exit "$SKIP"; }
die()  { echo "error: $*" >&2; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- inputs -----------------------------------------------------------------
MIRAGE_BIN=${MIRAGE_BIN:-mirage}
CORPUS_ROOT=${CORPUS_ROOT:-}
EMULATOR=${MIRAGE_CORPUS_EMULATOR:-hotswap}
ENV_FILE=${MIRAGE_CORPUS_ENV_FILE:-$script_dir/env/$EMULATOR.sh}
ONLY=${MIRAGE_CORPUS_ONLY:-all}
OUT_DIR=${MIRAGE_CORPUS_OUT_DIR:-}
IMAGE=${MIRAGE_CORPUS_IMAGE:-}

if [[ -n "$IMAGE" ]]; then
  MODE=container
  PROFILE=${MIRAGE_CORPUS_PROFILE:-corpus-$EMULATOR-img}
else
  MODE=host
  PROFILE=${MIRAGE_CORPUS_PROFILE:-corpus-$EMULATOR}
fi

case "$ONLY" in
  all | e2e | matmul) ;;
  *) die "MIRAGE_CORPUS_ONLY must be all, e2e, or matmul (got: $ONLY)" ;;
esac

# Builtin GPU agent the profile pins. HotSwap only supports MI450X.
if [[ -n "${MIRAGE_CORPUS_AGENT:-}" ]]; then
  AGENT=$MIRAGE_CORPUS_AGENT
elif [[ "$EMULATOR" == hotswap ]]; then
  AGENT=MI450X
else
  AGENT=MI300X
fi

# ---- shared prerequisites ---------------------------------------------------
command -v "$MIRAGE_BIN" >/dev/null 2>&1 \
  || skip "mirage binary not found (MIRAGE_BIN=$MIRAGE_BIN)"
[[ -n "$CORPUS_ROOT" && -d "$CORPUS_ROOT" ]] \
  || skip "corpus checkout not found (CORPUS_ROOT=$CORPUS_ROOT)"
CORPUS_ROOT=$(cd "$CORPUS_ROOT" && pwd)

runner="$CORPUS_ROOT/scripts/run_gfx1250_regression.sh"
[[ -x "$runner" ]] || skip "corpus runner missing or not executable: $runner"
[[ -r "$ENV_FILE" ]] || skip "env file not readable: $ENV_FILE"

# Wrappers: drop-in replacements for the IREE tools that forward argv through
# `mirage run`, so the corpus runner stays unmodified and unaware it is driving
# an emulator. `$1` is the wrapper name, `$2` the command the wrapper invokes
# inside `mirage run` (a host path in host mode, a bare tool name in container
# mode where it is resolved on the container PATH).
wrap_dir=$(mktemp -d)
trap 'rm -rf "$wrap_dir"' EXIT

make_wrapper() {
  local name=$1 target=$2
  cat >"$wrap_dir/$name" <<EOF
#!/usr/bin/env bash
exec "$MIRAGE_BIN" run --profile "$PROFILE" -- "$target" "\$@"
EOF
  chmod +x "$wrap_dir/$name"
}

ensure_profile() {
  # $1: profile JSON document
  if "$MIRAGE_BIN" profile show "$PROFILE" >/dev/null 2>&1; then
    return 0
  fi
  note "creating mirage profile '$PROFILE' (mode=$MODE emulator=$EMULATOR)"
  printf '%s\n' "$1" | "$MIRAGE_BIN" profile import - >/dev/null
}

if [[ "$MODE" == host ]]; then
  # ---- host mode ------------------------------------------------------------
  # The corpus executes the IREE tools on the host; without them there is
  # nothing to run, so skip (rather than fail).
  real_iree_check=$(command -v "${IREE_CHECK_MODULE:-iree-check-module}" 2>/dev/null || true)
  real_iree_matmul=$(command -v "${IREE_E2E_MATMUL_TEST:-iree-e2e-matmul-test}" 2>/dev/null || true)
  case "$ONLY" in
    all)
      [[ -n "$real_iree_check" && -n "$real_iree_matmul" ]] \
        || skip "iree tools not found in PATH (iree-check-module / iree-e2e-matmul-test)" ;;
    e2e)
      [[ -n "$real_iree_check" ]] || skip "iree-check-module not found in PATH" ;;
    matmul)
      [[ -n "$real_iree_matmul" ]] || skip "iree-e2e-matmul-test not found in PATH" ;;
  esac

  # Non-containerised profile for the chosen emulator.
  read -r -d '' profile_json <<JSON
{
  "name": "$PROFILE",
  "emulator": {
    "emulator": "$EMULATOR",
    "plugins": {},
    "exec_mode": "Functional",
    "options": {},
    "topology": { "num_nodes": 1, "gpus_per_node": 1, "agent": "$AGENT" }
  }
}
JSON
  ensure_profile "$profile_json" \
    || skip "could not create mirage profile '$PROFILE' (is emulator '$EMULATOR' installed?)"

  if [[ -n "$real_iree_check" ]]; then
    make_wrapper iree-check-module "$real_iree_check"
    export IREE_CHECK_MODULE="$wrap_dir/iree-check-module"
  fi
  if [[ -n "$real_iree_matmul" ]]; then
    make_wrapper iree-e2e-matmul-test "$real_iree_matmul"
    export IREE_E2E_MATMUL_TEST="$wrap_dir/iree-e2e-matmul-test"
  fi
else
  # ---- container mode -------------------------------------------------------
  # The IREE tools live inside $IMAGE, so the host does not need them. We do
  # need a container provider and the image present locally.
  provider=${MIRAGE_CORPUS_PROVIDER:-}
  if [[ -z "$provider" ]]; then
    if command -v podman >/dev/null 2>&1; then provider=podman
    elif command -v docker >/dev/null 2>&1; then provider=docker
    else skip "no container provider found (need podman or docker for image '$IMAGE')"; fi
  else
    command -v "$provider" >/dev/null 2>&1 || skip "container provider not found: $provider"
  fi
  "$provider" image inspect "$IMAGE" >/dev/null 2>&1 \
    || skip "image not present: $IMAGE (build it: cmake --build <build> --target corpus_image)"

  # Containerised profile around the IREE image. The corpus checkout is
  # bind-mounted at its identical host path so the packaged VMFB module paths
  # (absolute, under $CORPUS_ROOT) resolve unchanged inside the container.
  # mirage merges the emulator's own GPU device/mount injection on top.
  provider_field=""
  [[ -n "${MIRAGE_CORPUS_PROVIDER:-}" ]] && provider_field="\"provider\": \"$provider\","
  read -r -d '' profile_json <<JSON
{
  "name": "$PROFILE",
  "emulator": {
    "emulator": "$EMULATOR",
    "plugins": {},
    "exec_mode": "Functional",
    "options": {},
    "topology": { "num_nodes": 1, "gpus_per_node": 1, "agent": "$AGENT" }
  },
  "containerize": {
    $provider_field
    "image": "$IMAGE",
    "mounts": [
      { "host_path": "$CORPUS_ROOT", "container_path": "$CORPUS_ROOT", "read_only": true }
    ]
  }
}
JSON
  ensure_profile "$profile_json" \
    || skip "could not create containerised mirage profile '$PROFILE' (is emulator '$EMULATOR' installed?)"

  # Wrappers invoke the bare tool name, resolved on the container PATH.
  make_wrapper iree-check-module iree-check-module
  make_wrapper iree-e2e-matmul-test iree-e2e-matmul-test
  export IREE_CHECK_MODULE="$wrap_dir/iree-check-module"
  export IREE_E2E_MATMUL_TEST="$wrap_dir/iree-e2e-matmul-test"
fi

# ---- run the corpus regression through mirage -------------------------------
args=(--only "$ONLY")
[[ -n "$OUT_DIR" ]] && args+=(--out-dir "$OUT_DIR")

note "running corpus regression (mode=$MODE emulator=$EMULATOR profile=$PROFILE only=$ONLY)"
note "  corpus:   $CORPUS_ROOT"
note "  env file: $ENV_FILE"
[[ "$MODE" == container ]] && note "  image:    $IMAGE"
exec "$runner" "${args[@]}" "$ENV_FILE"
