#!/usr/bin/env bash
#
# build-and-package.sh — build HotSwap's `libhsa-hotswap.so` and stage it
# into a distributable directory.
#
# mirage does NOT build or install HotSwap for you. This script makes the
# build + packaging steps easy and reproducible. By default it stages the
# result into the workspace's Rust target/lib directory, which mirage
# discovers automatically (it searches `../lib` relative to the mirage
# binary). To use it elsewhere, copy the produced `libhsa-hotswap.so` to any
# location mirage searches (see ../README.md → "Where mirage looks"), e.g.:
#
#     cp target/lib/libhsa-hotswap.so "${ROCM_PATH:-/opt/rocm}/lib/"
#   or
#     export HOTSWAP_LIB="$PWD/target/lib/libhsa-hotswap.so"
#
# HotSwap lives in a fork of llvm-project that carries the load-time ISA
# rewriter runtime.
set -euo pipefail

# ---- defaults ---------------------------------------------------------------
LIB_NAME="libhsa-hotswap.so"
DEFAULT_REPO="https://github.com/martin-luecke/llvm-project.git"
DEFAULT_BRANCH="hotswap"

# Resolve this script's directory so paths are independent of the caller's cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# The mirage workspace root is two levels up (hotswap/scripts/ -> mirage/).
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"

# Default output is the workspace's Rust target/lib directory. mirage's
# discovery looks in `../lib` relative to the `target/<profile>/mirage`
# binary (i.e. `target/lib`), so staging there makes the freshly-built
# library discoverable without any extra configuration. Honor
# $CARGO_TARGET_DIR and `cargo metadata` when available.
resolve_target_dir() {
  if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    printf '%s\n' "$CARGO_TARGET_DIR"
    return
  fi
  if command -v cargo >/dev/null 2>&1; then
    local td
    td="$(cargo metadata --no-deps --format-version 1 \
            --manifest-path "$WORKSPACE_ROOT/Cargo.toml" 2>/dev/null \
          | tr ',' '\n' | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p' | head -n1)"
    if [[ -n "$td" ]]; then
      printf '%s\n' "$td"
      return
    fi
  fi
  printf '%s\n' "$WORKSPACE_ROOT/target"
}

SRC=""                       # existing checkout; empty => clone DEFAULT_REPO
REPO="${DEFAULT_REPO}"
BRANCH="${DEFAULT_BRANCH}"
OUT="$(resolve_target_dir)/lib"
BUILD_DIR="$(resolve_target_dir)/build-hotswap"
JOBS="$(nproc 2>/dev/null || echo 4)"
BUILD_TYPE="Release"
CLEAN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build HotSwap's ${LIB_NAME} and stage it into an output directory.
This script builds and packages only — it does not install the library.

Options:
  --src DIR         Use an existing llvm-project checkout instead of cloning.
  --repo URL        Git URL to clone when --src is not given.
                    (default: ${DEFAULT_REPO})
  --branch NAME     Branch/tag/commit to build. (default: ${DEFAULT_BRANCH})
  --out DIR         Output directory for the packaged library. (default: ${OUT})
  --build-dir DIR   CMake build directory. (default: ${BUILD_DIR})
  --jobs N          Parallel build jobs. (default: ${JOBS})
  --build-type T    CMake build type. (default: ${BUILD_TYPE})
  --clean           Remove the build directory before configuring.
  -h, --help        Show this help.

After it completes, install the result yourself, e.g.:
  cp ${OUT}/${LIB_NAME} "\${ROCM_PATH:-/opt/rocm}/lib/"
EOF
}

# ---- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)        SRC="${2:?--src needs a value}"; shift 2 ;;
    --repo)       REPO="${2:?--repo needs a value}"; shift 2 ;;
    --branch)     BRANCH="${2:?--branch needs a value}"; shift 2 ;;
    --out)        OUT="${2:?--out needs a value}"; shift 2 ;;
    --build-dir)  BUILD_DIR="${2:?--build-dir needs a value}"; shift 2 ;;
    --jobs)       JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    --build-type) BUILD_TYPE="${2:?--build-type needs a value}"; shift 2 ;;
    --clean)      CLEAN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;34m[hotswap]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[hotswap] error:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- prerequisites ----------------------------------------------------------
for tool in cmake ninja git; do
  command -v "$tool" >/dev/null 2>&1 || die "required tool not found on PATH: $tool"
done

# ---- obtain source ----------------------------------------------------------
if [[ -z "$SRC" ]]; then
  SRC="$(pwd)/llvm-project-hotswap"
  if [[ -d "$SRC/.git" ]]; then
    log "Reusing existing clone at $SRC"
    git -C "$SRC" fetch --depth 1 origin "$BRANCH"
    git -C "$SRC" checkout -q "$BRANCH"
    git -C "$SRC" reset --hard -q "origin/$BRANCH" || true
  else
    log "Cloning $REPO ($BRANCH) into $SRC"
    git clone --depth 1 --branch "$BRANCH" "$REPO" "$SRC"
  fi
else
  [[ -d "$SRC" ]] || die "--src path does not exist: $SRC"
  log "Using existing checkout: $SRC"
fi

# llvm-project's CMake entry point lives in the llvm/ subdirectory.
LLVM_CMAKE_SRC="$SRC/llvm"
[[ -f "$LLVM_CMAKE_SRC/CMakeLists.txt" ]] || \
  die "could not find llvm/CMakeLists.txt under $SRC (is this an llvm-project checkout?)"

# ---- configure --------------------------------------------------------------
if [[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]]; then
  log "Cleaning build directory $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

log "Configuring (build type: $BUILD_TYPE)"
cmake -G Ninja -S "$LLVM_CMAKE_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU"

# ---- build ------------------------------------------------------------------
log "Building $LIB_NAME (jobs: $JOBS)"
# The load-time ISA rewriter target builds the HSA tools library. Build the
# whole tree if the dedicated target is unavailable in this checkout.
if ! cmake --build "$BUILD_DIR" --target "$LIB_NAME" -j "$JOBS" 2>/dev/null; then
  log "Dedicated target unavailable; building default targets"
  cmake --build "$BUILD_DIR" -j "$JOBS"
fi

# ---- package ----------------------------------------------------------------
FOUND="$(find "$BUILD_DIR" -name "$LIB_NAME" -type f -print -quit 2>/dev/null || true)"
[[ -n "$FOUND" ]] || die "build finished but $LIB_NAME was not produced under $BUILD_DIR"

mkdir -p "$OUT"
cp -f "$FOUND" "$OUT/$LIB_NAME"
ABS_OUT="$(cd -- "$OUT" >/dev/null 2>&1 && pwd)"
log "Packaged: $ABS_OUT/$LIB_NAME"

cat <<EOF

Done. ${LIB_NAME} is staged at: $ABS_OUT/$LIB_NAME

This is the workspace Rust target/lib directory, which mirage discovers
automatically (it searches \`../lib\` relative to the mirage binary).
No further action is needed to use it with mirage from this workspace.

To make it discoverable elsewhere, copy it to any other searched location:
  - sudo cp "$ABS_OUT/$LIB_NAME" "\${ROCM_PATH:-/opt/rocm}/lib/"
  - export HOTSWAP_LIB="$ABS_OUT/$LIB_NAME"

See ../README.md → "Where mirage looks" for the full list.
EOF
