#!/usr/bin/env bash
# build-bundled-opencode.sh — apply perfxpert patches + build opencode +
# install the patched binary into perfxpert/_bundled/opencode.
#
# `pip install perfxpert` ships with an unbuilt placeholder for the
# opencode binary; running this script populates
# _bundled/opencode with our fork. The launcher prefers this bundled
# binary over any upstream install on disk, so users get the
# tool-priority gate + AMD rebrand patches automatically.
#
# Usage:
#   bash scripts/build-bundled-opencode.sh                 # build + install
#   bash scripts/build-bundled-opencode.sh --skip-install  # just build, don't copy
#
# Environment:
#   PERFXPERT_OPENCODE_DIR  override submodule location
#   PERFXPERT_BUNDLED_DIR   override output directory
#
# Requires: bun (https://bun.sh/install). Exits 2 if bun is missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERFXPERT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUBMODULE="${PERFXPERT_OPENCODE_DIR:-${PERFXPERT_ROOT}/opencode}"
BUNDLED_DIR="${PERFXPERT_BUNDLED_DIR:-${PERFXPERT_ROOT}/perfxpert/_bundled}"
OUTPUT="${BUNDLED_DIR}/opencode"

SKIP_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    -h|--help)
      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'
      exit 0
      ;;
  esac
done

# --- 1. Submodule sanity -----------------------------------------------------
if [ ! -f "${SUBMODULE}/package.json" ]; then
  echo "build-bundled-opencode: opencode submodule missing at ${SUBMODULE}" >&2
  echo "  Run: git submodule update --init --recursive" >&2
  exit 2
fi

# --- 2. bun availability -----------------------------------------------------
if ! command -v bun >/dev/null 2>&1; then
  cat >&2 <<'EOF'
build-bundled-opencode: bun is required to compile opencode.

Install bun (one line, no root required):
    curl -fsSL https://bun.sh/install | bash

Then re-run:
    perfxpert-code install-patches

Or point perfxpert at an already-built opencode binary:
    export PERFXPERT_OPENCODE_PATH=/path/to/opencode
EOF
  exit 2
fi

echo "build-bundled-opencode: bun=$(bun --version), submodule=${SUBMODULE}"

# --- 3. Apply patches --------------------------------------------------------
# Idempotent: the apply script bails out on dirty tree, so we only apply
# when the submodule is at the pinned upstream commit with no local diff. If patches
# are already applied (tree is dirty), we skip silently. A non-git tree is
# rejected outright: the patch script relies on git apply/checkout semantics.
cd "${SUBMODULE}"
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "build-bundled-opencode: ${SUBMODULE} is not a git checkout; refusing to build without the repo-pinned submodule metadata" >&2
  exit 2
fi
if git diff --quiet HEAD -- 2>/dev/null; then
  echo "build-bundled-opencode: applying ${PERFXPERT_ROOT}/.patches/*.patch"
  bash "${SCRIPT_DIR}/apply-opencode-patches.sh"
else
  echo "build-bundled-opencode: submodule already patched (dirty tree) — skipping apply"
fi

# --- 4. Install deps + build -------------------------------------------------
cd "${SUBMODULE}"
if [ ! -f "bun.lock" ] && [ ! -f "bun.lockb" ]; then
  echo "build-bundled-opencode: missing bun.lock/bun.lockb in ${SUBMODULE}" >&2
  exit 2
fi
if [ ! -d "node_modules" ] || [ "${PERFXPERT_FORCE_INSTALL:-0}" = "1" ]; then
  echo "build-bundled-opencode: running 'bun install --frozen-lockfile --ignore-scripts' at ${SUBMODULE}"
  if ! bun install --frozen-lockfile --ignore-scripts; then
    echo "build-bundled-opencode: frozen lockfile install failed; retrying 'bun install --ignore-scripts'" >&2
    bun install --ignore-scripts
  fi
  echo "build-bundled-opencode: running explicit postinstall 'bun run --cwd packages/opencode fix-node-pty'"
  bun run --cwd packages/opencode fix-node-pty
else
  echo "build-bundled-opencode: node_modules/ present — skipping 'bun install'"
fi

cd "${SUBMODULE}/packages/opencode"
echo "build-bundled-opencode: compiling opencode (bun run build --single)"
bun run build --single

file_size_bytes() {
  local path="$1"
  if stat -c%s "$path" >/dev/null 2>&1; then
    stat -c%s "$path"
  elif stat -f%z "$path" >/dev/null 2>&1; then
    stat -f%z "$path"
  else
    wc -c < "$path" | tr -d '[:space:]'
  fi
}

file_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo "<sha256-unavailable>"
  fi
}

# --- 5. Locate built binary and copy to _bundled/ ----------------------------
CANDIDATES=(
  "${SUBMODULE}/packages/opencode/dist/opencode-linux-x64/bin/opencode"
  "${SUBMODULE}/packages/opencode/dist/opencode-linux-arm64/bin/opencode"
  "${SUBMODULE}/packages/opencode/dist/opencode-darwin-x64/bin/opencode"
  "${SUBMODULE}/packages/opencode/dist/opencode-darwin-arm64/bin/opencode"
  "${SUBMODULE}/packages/opencode/dist/opencode/bin/opencode"
)

BUILT=""
for c in "${CANDIDATES[@]}"; do
  if [ -x "$c" ]; then
    BUILT="$c"
    break
  fi
done

if [ -z "${BUILT}" ]; then
  echo "build-bundled-opencode: ERROR — no built binary found under packages/opencode/dist/" >&2
  find "${SUBMODULE}/packages/opencode/dist" -maxdepth 4 -name opencode -type f 2>/dev/null >&2 || true
  exit 3
fi

echo "build-bundled-opencode: built ${BUILT} ($(file_size_bytes "${BUILT}") bytes)"

if [ "${SKIP_INSTALL}" = "1" ]; then
  echo "build-bundled-opencode: --skip-install set; not copying to ${OUTPUT}"
  exit 0
fi

mkdir -p "${BUNDLED_DIR}"
cp -f "${BUILT}" "${OUTPUT}"
chmod +x "${OUTPUT}"
echo "build-bundled-opencode: bundled → ${OUTPUT}"
echo "build-bundled-opencode: sha256=$(file_sha256 "${OUTPUT}")"
