#!/usr/bin/env bash
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build, stage, and verify the prebuilt waitcheck and ConSan bundle documented
# in docs/sanitizers.md. This is the whole build recipe behind
# .github/workflows/rocjitsu-sanitizer-artifacts.yml; the workflow only wires
# up the container, the checkout, and the artifact upload, so everything here
# can be run and debugged locally.
#
# The build needs no ROCm SDK: the HSA/KFD headers are vendored under
# lib/rocjitsu/external_headers.
#
# Usage:
#   ./scripts/build_sanitizer_artifacts.sh <command>...
#
# Commands:
#   deps         Install the distro packages the build needs (dnf/AlmaLinux 8).
#   zstd         Build and install the static zstd the binaries link in.
#   configure    Configure the CMake build.
#   build        Build librocjitsu_dbi_hooks.so and rj_waitcheck.
#   stage        Copy the binaries into the bundle and write its metadata.
#   smoke-test   Check the staged bundle against its runtime contract.
#   summary      Print a Markdown summary of the staged bundle.
#   all          zstd, configure, build, stage, smoke-test.
#
# Example, reproducing a CI bundle in the same image CI uses:
#   docker run --rm -v "$PWD:/src" -w /src/emulation/rocjitsu \
#     ghcr.io/rocm/therock_build_manylinux_x86_64 \
#     bash -c 'scripts/build_sanitizer_artifacts.sh deps all'
#
# Environment:
#   ROCJITSU_SOURCE_DIR      rocjitsu source tree. Default: this script's parent.
#   ROCJITSU_BUILD_DIR       CMake build tree. Default: <source>/build/sanitizers.
#   ROCJITSU_STAGE_DIR       Bundle staging root. Default: <source>/build/sanitizer-stage.
#   ROCJITSU_BUILD_TYPE      CMake build type. Default: Release.
#   ROCJITSU_BUILD_JOBS      Build parallelism. Default: half the CPUs.
#   ROCJITSU_BUILD_IMAGE     Container image recorded in the manifest. Default: unset.
#   ROCJITSU_RETENTION_DAYS  Retention shown in the summary. Default: 30.
#   ZSTD_PREFIX              Static zstd install prefix. Default: <source>/build/zstd-prefix.
#
# Under Actions the standard GITHUB_* variables supply the manifest's
# provenance and receive the step outputs and job summary; outside it the
# commit and branch come from git and the summary goes to stdout.

set -euo pipefail

# Upstream's own release, pinned by the checksum published with the release
# asset and cross-checked against an independent distribution manifest.
readonly ZSTD_VERSION="1.5.7"
readonly ZSTD_SHA256="eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3"

# Everything the binaries may still load from the host. Anything outside this
# set means a dependency escaped static linking, which smoke-test treats as a
# build failure.
readonly ALLOWED_NEEDED='^(libc|libm|libdl|libpthread|librt)\.so\.[0-9]+$|^ld-linux-x86-64\.so\.2$'

# Absorbing zlib and zstd only stays safe while their symbols remain local. The
# hook is loaded into someone else's process, so an exported inflate() would
# interpose that process's own zlib.
readonly FORBIDDEN_EXPORTS='^(ZSTD_|ZDICT_|inflate|deflate|compress|uncompress|gz|crc32|adler32)'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${ROCJITSU_SOURCE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BUILD_DIR="${ROCJITSU_BUILD_DIR:-$SOURCE_DIR/build/sanitizers}"
STAGE_DIR="${ROCJITSU_STAGE_DIR:-$SOURCE_DIR/build/sanitizer-stage}"
ZSTD_PREFIX="${ZSTD_PREFIX:-$SOURCE_DIR/build/zstd-prefix}"
WORK_DIR="${RUNNER_TEMP:-$SOURCE_DIR/build/sanitizer-work}"
BUILD_TYPE="${ROCJITSU_BUILD_TYPE:-Release}"
BUILD_IMAGE="${ROCJITSU_BUILD_IMAGE:-}"
RETENTION_DAYS="${ROCJITSU_RETENTION_DAYS:-30}"

readonly HOOK_NAME="librocjitsu_dbi_hooks.so"
readonly WAITCHECK_NAME="rj_waitcheck"
HOOK_BUILT="$BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/$HOOK_NAME"
WAITCHECK_BUILT="$BUILD_DIR/tools/$WAITCHECK_NAME"
HOOK_STAGED="$STAGE_DIR/lib/$HOOK_NAME"
WAITCHECK_STAGED="$STAGE_DIR/bin/$WAITCHECK_NAME"

usage() {
  cat <<'EOF'
Usage:
  build_sanitizer_artifacts.sh <command>...

Commands:
  deps         Install the distro packages the build needs (dnf/AlmaLinux 8).
  zstd         Build and install the static zstd the binaries link in.
  configure    Configure the CMake build.
  build        Build librocjitsu_dbi_hooks.so and rj_waitcheck.
  stage        Copy the binaries into the bundle and write its metadata.
  smoke-test   Check the staged bundle against its runtime contract.
  summary      Print a Markdown summary of the staged bundle.
  all          zstd, configure, build, stage, smoke-test.
  -h, --help   Show this help.

Environment:
  ROCJITSU_SOURCE_DIR, ROCJITSU_BUILD_DIR, ROCJITSU_STAGE_DIR,
  ROCJITSU_BUILD_TYPE, ROCJITSU_BUILD_JOBS, ROCJITSU_BUILD_IMAGE,
  ROCJITSU_RETENTION_DAYS, ZSTD_PREFIX

See the comment at the top of this script for what each one defaults to.
EOF
}

log() {
  printf '==> %s\n' "$*"
}

# `::error::` turns into an annotation on the job; elsewhere it is noise.
die() {
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    printf '::error::%s\n' "$*" >&2
  else
    printf 'error: %s\n' "$*" >&2
  fi
  exit 1
}

require() {
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
  done
}

set_output() {
  [[ -n "${GITHUB_OUTPUT:-}" ]] || return 0
  printf '%s=%s\n' "$1" "$2" >>"$GITHUB_OUTPUT"
}

build_jobs() {
  if [[ -n "${ROCJITSU_BUILD_JOBS:-}" ]]; then
    printf '%s\n' "$ROCJITSU_BUILD_JOBS"
    return
  fi
  # Halve the worker count to avoid OOM and excessive system load on the shared
  # scale set, matching rocjitsu-corpus-tests.yml.
  printf '%s\n' "$(( ($(nproc) + 1) / 2 ))"
}

git_head() {
  git -C "$SOURCE_DIR" "$@" 2>/dev/null || true
}

commit_sha() {
  printf '%s\n' "${GITHUB_SHA:-$(git_head rev-parse HEAD)}"
}

artifact_name() {
  local commit
  commit="$(commit_sha)"
  printf 'rocjitsu-sanitizers-%s-%s\n' "$BUILD_TYPE" "${commit:0:12}"
}

# AlmaLinux 8 ships libzstd-devel without a static archive or any of upstream's
# CMake package files, so the prefix built by `zstd` below is the only place
# find_package(zstd) can pick up a static zstd::libzstd_static. The library
# directory name follows the distro, so accept either spelling.
zstd_cmake_dir() {
  local dir
  for dir in "$ZSTD_PREFIX/lib64/cmake/zstd" "$ZSTD_PREFIX/lib/cmake/zstd"; do
    if [[ -d "$dir" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  die "no zstd CMake package under $ZSTD_PREFIX; run '$0 zstd' first"
}

cmd_deps() {
  # The manylinux image already carries CMake, Ninja, and gcc-toolset-13.
  # zlib-static supplies the PIC libz.a the published binaries link in; zstd
  # has no static package on AlmaLinux 8 and is built by `zstd` instead.
  command -v dnf >/dev/null 2>&1 ||
    die "deps only knows dnf; install a static zlib and jq with your own package manager"
  log "installing build dependencies"
  dnf install -y -q zlib-devel zlib-static jq
}

cmd_zstd() {
  require curl tar sha256sum cmake

  log "building a static zstd $ZSTD_VERSION"
  local src="$WORK_DIR/zstd-src"
  local build="$WORK_DIR/zstd-build"
  local tarball="$WORK_DIR/zstd-$ZSTD_VERSION.tar.gz"

  rm -rf "$src" "$build"
  mkdir -p "$src"
  curl -sSfL -o "$tarball" \
    "https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz"
  printf '%s  %s\n' "$ZSTD_SHA256" "$tarball" | sha256sum --check --quiet
  tar -xzf "$tarball" -C "$src" --strip-components=1

  # Build it PIC and static so the project's find_package(zstd) consumes the
  # exported zstd::libzstd_static target unmodified.
  cmake -S "$src/build/cmake" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ZSTD_PREFIX" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DZSTD_BUILD_SHARED=OFF \
    -DZSTD_BUILD_STATIC=ON \
    -DZSTD_BUILD_PROGRAMS=OFF \
    -DZSTD_BUILD_TESTS=OFF
  cmake --build "$build" -j "$(nproc)"
  cmake --install "$build"
}

cmd_configure() {
  require cmake

  log "configuring $BUILD_TYPE build in $BUILD_DIR"
  # Link every non-glibc dependency statically -- libstdc++ and libgcc from the
  # toolchain, zlib and zstd from the packages above -- so the published
  # binaries need nothing from the host but glibc itself. The libstdc++ half
  # matches rocjitsu-docker-build.sh. Symbols stay hidden: both targets already
  # link with --exclude-libs,ALL, so the absorbed libraries cannot collide with
  # a host copy loaded into the same process.
  #
  # Keep FetchContent out of the source tree so the checkout stays clean.
  cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DFETCHCONTENT_BASE_DIR="$WORK_DIR/deps" \
    -Dzstd_DIR="$(zstd_cmake_dir)" \
    -DZLIB_USE_STATIC_LIBS=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    -DBUILD_TESTING=OFF
}

cmd_build() {
  require cmake

  local jobs
  jobs="$(build_jobs)"
  log "building rocjitsu_dbi_hooks and rj_waitcheck with $jobs jobs"
  cmake --build "$BUILD_DIR" \
    --target rocjitsu_dbi_hooks rj_waitcheck \
    -j "$jobs"
}

# The compiler CMake actually configured, which is not always whatever `g++`
# resolves to on PATH -- the project defaults to amdclang++ when it finds one.
build_compiler() {
  local cxx=""
  if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cxx="$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$BUILD_DIR/CMakeCache.txt" | head -1)"
  fi
  if [[ -x "$cxx" ]]; then
    "$cxx" --version 2>/dev/null | head -1
  else
    printf 'unknown\n'
  fi
}

# Every library the loader will go looking for on the host, one per line.
needed_libraries() {
  objdump -p "$WAITCHECK_STAGED" "$HOOK_STAGED" |
    awk '/NEEDED/ {print $2}' |
    sort -u
}

cmd_stage() {
  require objdump jq sha256sum

  [[ -f "$HOOK_BUILT" ]] || die "missing build output: $HOOK_BUILT"
  [[ -f "$WAITCHECK_BUILT" ]] || die "missing build output: $WAITCHECK_BUILT"

  log "staging the bundle in $STAGE_DIR"
  rm -rf "$STAGE_DIR"
  mkdir -p "$STAGE_DIR/lib" "$STAGE_DIR/bin"
  cp "$HOOK_BUILT" "$HOOK_STAGED"
  cp "$WAITCHECK_BUILT" "$WAITCHECK_STAGED"

  # Record enough provenance to tie a downloaded bundle back to a commit and to
  # the toolchain baseline the binaries were linked against.
  #
  # Report the highest versioned glibc symbol the binaries actually reference
  # rather than the build environment's own glibc: the build image is newer
  # than what the artifacts require, and the difference decides whether a given
  # host can run them.
  local glibc_version host_glibc compiler_version needed
  glibc_version="$(
    objdump -T "$HOOK_STAGED" "$WAITCHECK_STAGED" |
      grep -o 'GLIBC_[0-9.]*' |
      sed 's/^GLIBC_//' |
      sort -V |
      tail -1
  )"
  host_glibc="$(ldd --version | head -1 | awk '{print $NF}')"
  # The rest of the runtime contract: every library the loader will go looking
  # for on the host. Static linking should leave only glibc here, and
  # smoke-test enforces that, but record it either way so the bundle documents
  # itself.
  needed="$(needed_libraries | jq -Rn '[inputs]')"
  compiler_version="$(build_compiler)"

  local repository run_id run_url
  repository="${GITHUB_REPOSITORY:-}"
  run_id="${GITHUB_RUN_ID:-}"
  run_url=""
  if [[ -n "$repository" && -n "$run_id" ]]; then
    run_url="https://github.com/$repository/actions/runs/$run_id"
  fi

  jq -n \
    --arg repository "$repository" \
    --arg ref "${GITHUB_REF_NAME:-$(git_head rev-parse --abbrev-ref HEAD)}" \
    --arg commit "$(commit_sha)" \
    --arg run_id "$run_id" \
    --arg run_number "${GITHUB_RUN_NUMBER:-}" \
    --arg run_attempt "${GITHUB_RUN_ATTEMPT:-}" \
    --arg build_type "$BUILD_TYPE" \
    --arg compiler "$compiler_version" \
    --arg glibc "$glibc_version" \
    --arg host_glibc "$host_glibc" \
    --arg image "$BUILD_IMAGE" \
    --arg run_url "$run_url" \
    --argjson needed "$needed" \
    '{
      repository: $repository,
      branch: $ref,
      commit: $commit,
      run_id: $run_id,
      run_number: $run_number,
      run_attempt: $run_attempt,
      build_type: $build_type,
      compiler: $compiler,
      minimum_glibc: $glibc,
      shared_library_dependencies: $needed,
      build_image: $image,
      build_image_glibc: $host_glibc,
      run_url: $run_url
    }' >"$STAGE_DIR/MANIFEST.json"

  # Digests are relative to the bundle root so the download script can verify
  # them after extraction.
  (
    cd "$STAGE_DIR"
    find bin lib -type f -print0 |
      sort -z |
      xargs -0 sha256sum >sha256sums.txt
  )

  set_output artifact_name "$(artifact_name)"
  set_output stage_dir "$STAGE_DIR"
}

cmd_smoke_test() {
  require objdump nm sha256sum

  [[ -f "$HOOK_STAGED" ]] || die "nothing staged at $STAGE_DIR; run '$0 stage' first"

  log "smoke testing the staged bundle"
  # Confirm the tool starts and the hook resolves every symbol it needs before
  # publishing binaries that users download unbuilt.
  "$WAITCHECK_STAGED" --help >/dev/null

  # `ldd -r` reports unresolved symbols on stdout but still exits 0, so inspect
  # its output rather than trusting the status.
  local ldd_output
  ldd_output="$(ldd -r "$HOOK_STAGED" 2>&1)"
  printf '%s\n' "$ldd_output"
  if grep -q "undefined symbol" <<<"$ldd_output"; then
    die "$HOOK_NAME has unresolved symbols"
  fi

  # Hold the documented runtime contract: glibc and nothing else. A new
  # dependency that slips past static linking would otherwise surface only as a
  # load failure on a user's minimal host.
  local unexpected
  unexpected="$(needed_libraries | grep -Ev "$ALLOWED_NEEDED" || true)"
  if [[ -n "$unexpected" ]]; then
    die "binaries depend on host libraries beyond glibc: ${unexpected//$'\n'/ }"
  fi

  local exported
  exported="$(
    nm -D --defined-only "$HOOK_STAGED" |
      awk '{print $NF}' |
      grep -E "$FORBIDDEN_EXPORTS" || true
  )"
  if [[ -n "$exported" ]]; then
    die "the hook exports absorbed compression symbols: ${exported//$'\n'/ }"
  fi

  (
    cd "$STAGE_DIR"
    sha256sum --check --quiet sha256sums.txt
  )
  log "MANIFEST.json"
  cat "$STAGE_DIR/MANIFEST.json"
}

cmd_summary() {
  local out="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
  {
    echo "### rocjitsu sanitizer artifacts"
    echo
    echo "| Field | Value |"
    echo "| --- | --- |"
    echo "| Artifact | \`$(artifact_name)\` |"
    echo "| Build type | \`$BUILD_TYPE\` |"
    echo "| Commit | \`$(commit_sha)\` |"
    echo "| Retention | $RETENTION_DAYS days |"
    if [[ -n "${GITHUB_RUN_ID:-}" ]]; then
      echo
      echo "Download with:"
      echo
      echo '```sh'
      echo "python3 emulation/rocjitsu/scripts/download_sanitizer_artifacts.py --run ${GITHUB_RUN_ID}"
      echo '```'
    fi
  } >>"$out"
}

cmd_all() {
  cmd_zstd
  cmd_configure
  cmd_build
  cmd_stage
  cmd_smoke_test
}

main() {
  [[ $# -gt 0 ]] || { usage >&2; exit 2; }

  local command
  for command in "$@"; do
    case "$command" in
      -h|--help) usage; exit 0 ;;
    esac
  done

  mkdir -p "$WORK_DIR"

  for command in "$@"; do
    case "$command" in
      deps) cmd_deps ;;
      zstd) cmd_zstd ;;
      configure) cmd_configure ;;
      build) cmd_build ;;
      stage) cmd_stage ;;
      smoke-test) cmd_smoke_test ;;
      summary) cmd_summary ;;
      all) cmd_all ;;
      *) die "unknown command: $command" ;;
    esac
  done
}

main "$@"
