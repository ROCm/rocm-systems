#!/usr/bin/env bash
# build-patched-opencode.sh — apply perfxpert patches + build opencode +
# install the patched binary into a PerfXpert-managed artifact path.
#
# The launcher prefers this patched binary over any upstream install on disk,
# so users get the tool-priority gate + AMD rebrand patches automatically
# without shipping a generated binary inside the Python package.
#
# Usage:
#   bash scripts/build-patched-opencode.sh                 # build + install
#   bash scripts/build-patched-opencode.sh --skip-install  # just build, don't copy
#
# Environment:
#   PERFXPERT_OPENCODE_DIR  override submodule location
#   PERFXPERT_PATCHED_OPENCODE_PATH  override output binary path
#   XDG_CACHE_HOME          cache root for default output path
#
# Requires: bun (https://bun.sh/install). Exits 2 if bun is missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERFXPERT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUBMODULE="${PERFXPERT_OPENCODE_DIR:-${PERFXPERT_ROOT}/opencode}"
PATCH_MANIFEST="${PERFXPERT_PATCH_MANIFEST:-${PERFXPERT_ROOT}/.patches/SHA256SUMS}"
DEFAULT_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}"
OUTPUT="${PERFXPERT_PATCHED_OPENCODE_PATH:-${DEFAULT_CACHE_ROOT}/perfxpert/opencode/opencode}"

SKIP_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    -h|--help)
      awk '
        NR == 1 { next }
        /^#/ {
          sub(/^# ?/, "")
          print
          next
        }
        { exit }
      ' "${BASH_SOURCE[0]}"
      exit 0
      ;;
  esac
done

# --- 1. Submodule sanity -----------------------------------------------------
if [ ! -f "${SUBMODULE}/package.json" ]; then
  echo "build-patched-opencode: opencode submodule missing at ${SUBMODULE}" >&2
  echo "  Run: git submodule update --init --recursive" >&2
  exit 2
fi

# --- 2. bun availability -----------------------------------------------------
if ! command -v bun >/dev/null 2>&1; then
  cat >&2 <<'EOF'
build-patched-opencode: bun is required to compile opencode.

Install bun (one line, no root required):
    curl -fsSL https://bun.sh/install | bash

Then re-run:
    perfxpert-code install-patches
EOF
  exit 2
fi

echo "build-patched-opencode: bun=$(bun --version), submodule=${SUBMODULE}"

patch_series_already_applied() {
  local tmp_root tmp_tree current_diff expected_diff rc
  tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/perfxpert-opencode-patchcheck.XXXXXX")" || return 1
  tmp_tree="${tmp_root}/opencode-clean"
  current_diff="${tmp_root}/current.patch"
  expected_diff="${tmp_root}/expected.patch"

  git diff --binary HEAD -- > "${current_diff}" || {
    rm -rf "${tmp_root}"
    return 1
  }

  if ! git worktree add --detach "${tmp_tree}" HEAD >/dev/null 2>&1; then
    rm -rf "${tmp_root}"
    return 1
  fi

  if ! (
    cd "${tmp_tree}"
    PERFXPERT_OPENCODE_DIR="${tmp_tree}" bash "${SCRIPT_DIR}/apply-opencode-patches.sh" >/dev/null
    git diff --binary HEAD -- > "${expected_diff}"
  ); then
    git worktree remove --force "${tmp_tree}" >/dev/null 2>&1 || true
    rm -rf "${tmp_root}"
    return 1
  fi

  if cmp -s "${current_diff}" "${expected_diff}"; then
    rc=0
  else
    rc=1
  fi

  git worktree remove --force "${tmp_tree}" >/dev/null 2>&1 || true
  rm -rf "${tmp_root}"
  return "${rc}"
}

# --- 3. Apply patches --------------------------------------------------------
# The build artifact is trusted later as "patched", so never build from a dirty
# submodule whose patch state cannot be proven. Re-running after our own patch
# application is allowed only when the current diff exactly matches the managed
# patch series generated from HEAD.
cd "${SUBMODULE}"
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "build-patched-opencode: ${SUBMODULE} is not a git checkout; refusing to build without the repo-pinned submodule metadata" >&2
  exit 2
fi
if git diff --quiet HEAD -- 2>/dev/null; then
  echo "build-patched-opencode: applying ${PERFXPERT_ROOT}/.patches/*.patch"
  bash "${SCRIPT_DIR}/apply-opencode-patches.sh"
elif patch_series_already_applied; then
  echo "build-patched-opencode: managed patch series already applied in ${SUBMODULE}"
else
  echo "build-patched-opencode: ${SUBMODULE} has uncommitted changes; refusing to build a patched artifact from an unverifiable submodule state" >&2
  echo "  Run: (cd ${SUBMODULE} && git checkout HEAD -- .)" >&2
  echo "  Then re-run: bash ${BASH_SOURCE[0]}" >&2
  exit 2
fi

# --- 4. Install deps + build -------------------------------------------------
cd "${SUBMODULE}"
if [ ! -f "bun.lock" ] && [ ! -f "bun.lockb" ]; then
  echo "build-patched-opencode: missing bun.lock/bun.lockb in ${SUBMODULE}" >&2
  exit 2
fi
if [ ! -d "node_modules" ] || [ "${PERFXPERT_FORCE_INSTALL:-0}" = "1" ]; then
  echo "build-patched-opencode: running 'bun install --frozen-lockfile --ignore-scripts' at ${SUBMODULE}"
  if ! bun install --frozen-lockfile --ignore-scripts; then
    echo "build-patched-opencode: frozen lockfile install failed; retrying 'bun install --ignore-scripts'" >&2
    bun install --ignore-scripts
  fi
  echo "build-patched-opencode: running explicit postinstall 'bun run --cwd packages/opencode fix-node-pty'"
  bun run --cwd packages/opencode fix-node-pty
else
  echo "build-patched-opencode: node_modules/ present — skipping 'bun install'"
fi

cd "${SUBMODULE}/packages/opencode"
echo "build-patched-opencode: compiling opencode (bun run build --single --skip-install)"
bun run build --single --skip-install

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

# --- 5. Locate built binary and copy to artifact path ------------------------
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
  echo "build-patched-opencode: ERROR — no built binary found under packages/opencode/dist/" >&2
  find "${SUBMODULE}/packages/opencode/dist" -maxdepth 4 -name opencode -type f 2>/dev/null >&2 || true
  exit 3
fi

echo "build-patched-opencode: built ${BUILT} ($(file_size_bytes "${BUILT}") bytes)"

if [ "${SKIP_INSTALL}" = "1" ]; then
  echo "build-patched-opencode: --skip-install set; not copying to ${OUTPUT}"
  exit 0
fi

mkdir -p "$(dirname "${OUTPUT}")"
cp -f "${BUILT}" "${OUTPUT}"
chmod +x "${OUTPUT}"
file_sha256 "${PATCH_MANIFEST}" > "${OUTPUT}.perfxpert-patch-manifest-sha256"
echo "build-patched-opencode: installed → ${OUTPUT}"
echo "build-patched-opencode: sha256=$(file_sha256 "${OUTPUT}")"
