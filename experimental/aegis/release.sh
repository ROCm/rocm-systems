#!/bin/bash
# Build a self-contained binary release tarball.
#
# Usage: ./release.sh [--arch gfx950]
#
# Produces: aegisbit-<version>-<arch>.tar.gz
#   aegisbit/
#   ├── lib/libaegisbit.so
#   └── tools/aegisbit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ARCH="${2:-gfx950}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# Extract version from CMakeLists.txt
VERSION=$(grep -oP 'project\(AegisBit VERSION \K[0-9.]+' "${SCRIPT_DIR}/CMakeLists.txt")
if [ -z "$VERSION" ]; then
  echo "Error: could not extract version from CMakeLists.txt"
  exit 1
fi

RELEASE_NAME="aegisbit-${VERSION}-${ARCH}"
STAGING_DIR="${BUILD_DIR}/${RELEASE_NAME}"

echo "=== AegisBit Release Builder ==="
echo "  Version:  ${VERSION}"
echo "  Arch:     ${ARCH}"
echo "  Output:   ${RELEASE_NAME}.tar.gz"
echo ""

# Build
echo "Building..."
"$SCRIPT_DIR/build.sh" build

# Verify shared library exists
SO_PATH="${BUILD_DIR}/src/libaegisbit.so"
if [ ! -f "$SO_PATH" ]; then
  echo "Error: libaegisbit.so not found at ${SO_PATH}"
  echo "GPU support (HIP + HSA runtime) is required for the shared library."
  exit 1
fi

# Stage release layout
echo "Staging release..."
rm -rf "$STAGING_DIR"
mkdir -p "${STAGING_DIR}/lib" "${STAGING_DIR}/tools"

cp "$SO_PATH" "${STAGING_DIR}/lib/libaegisbit.so"
cp "${SCRIPT_DIR}/tools/aegisbit" "${STAGING_DIR}/tools/aegisbit"
cp "${SCRIPT_DIR}/tools/smoke-test.sh" "${STAGING_DIR}/tools/smoke-test.sh"
chmod +x "${STAGING_DIR}/tools/aegisbit" "${STAGING_DIR}/tools/smoke-test.sh"

# Verify the binary is self-contained (no LLVM .so deps)
echo "Checking dependencies..."
LLVM_DEPS=$(ldd "${STAGING_DIR}/lib/libaegisbit.so" 2>/dev/null | grep -i "llvm" || true)
if [ -n "$LLVM_DEPS" ]; then
  echo "WARNING: libaegisbit.so has dynamic LLVM dependencies:"
  echo "$LLVM_DEPS"
  echo "The binary release should have LLVM statically linked."
  exit 1
fi

# Print dependency summary
echo ""
echo "Runtime dependencies (must be present on target system):"
ldd "${STAGING_DIR}/lib/libaegisbit.so" 2>/dev/null | grep -E "rocprofiler|hsa-runtime" | sed 's/^/  /'
echo ""

# Report glibc floor
GLIBC_MAX=$(readelf -V "${STAGING_DIR}/lib/libaegisbit.so" 2>/dev/null \
  | grep -oP 'GLIBC_\K[0-9.]+' | sort -V | tail -1)
GLIBCXX_MAX=$(readelf -V "${STAGING_DIR}/lib/libaegisbit.so" 2>/dev/null \
  | grep -oP 'GLIBCXX_\K[0-9.]+' | sort -V | tail -1)
echo "Minimum system requirements:"
echo "  glibc   >= ${GLIBC_MAX:-unknown}"
echo "  libstdc++ (GLIBCXX) >= ${GLIBCXX_MAX:-unknown}"
echo ""

# Create tarball
echo "Creating tarball..."
tar czf "${BUILD_DIR}/${RELEASE_NAME}.tar.gz" -C "${BUILD_DIR}" "${RELEASE_NAME}"

# Verify
SIZE=$(du -h "${BUILD_DIR}/${RELEASE_NAME}.tar.gz" | cut -f1)
echo ""
echo "=== Release built successfully ==="
echo "  ${BUILD_DIR}/${RELEASE_NAME}.tar.gz  (${SIZE})"
echo ""
echo "Install and verify with:"
echo "  tar xf ${RELEASE_NAME}.tar.gz"
echo "  ./${RELEASE_NAME}/tools/aegisbit --dry-run -- echo hello"
echo "  ./${RELEASE_NAME}/tools/smoke-test.sh   # full GPU verification"

# Cleanup staging
rm -rf "$STAGING_DIR"
