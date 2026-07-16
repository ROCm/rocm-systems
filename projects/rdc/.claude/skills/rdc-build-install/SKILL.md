---
name: rdc-build-install
description: "Build and install RDC from source. Use when: building locally, installing before tests, pre-review build verification, build + install + verify. Requires GRPC_ROOT to be set."
---

# Build & Install RDC

Builds RDC from source, packages it, and installs locally. Used by the review agent as a pre-step before dispatching subagents, and can be invoked independently.

## Prerequisites

- CMake 3.15+, GCC 5.4+, and a C/C++ toolchain installed
- gRPC 1.78.1 pre-built at `$GRPC_ROOT` (required for `BUILD_STANDALONE=ON`)
- AMDSMI installed at `/opt/rocm` (required at runtime)
- `sudo` access for package install
- Working directory must be the rdc workspace root (`projects/rdc/`)

## Build Commands

All commands assume `$WORKSPACE` is the rdc workspace root (where `CMakeLists.txt` lives).

### Step 1: Clean

```bash
cd "$WORKSPACE"
sudo rm -rf build
```

### Step 2: Configure & Build

```bash
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGRPC_ROOT="${GRPC_ROOT:-/opt/grpc}" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j "$(nproc)"
make package
```

If `GRPC_ROOT` is not set or gRPC is not available, build without standalone mode:

```bash
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DBUILD_STANDALONE=OFF \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Step 3: Uninstall Previous

```bash
sudo dpkg --remove --force-depends rdc-tests 2>/dev/null || true
sudo dpkg --remove --force-depends rdc 2>/dev/null || true
```

### Step 4: Install

```bash
# Detect package format
PKG_FMT=$(command -v dpkg >/dev/null && echo deb || echo rpm)

if [ "$PKG_FMT" = "deb" ]; then
  sudo dpkg -i build/rdc*99999-local_amd64.deb 2>/dev/null || \
  sudo dpkg -i build/rdc*.deb
else
  sudo rpm -Uvh --force build/rdc*99999-local*x86_64.rpm 2>/dev/null || \
  sudo rpm -Uvh --force build/rdc*.rpm
fi
```

### Step 5: Verify

```bash
/opt/rocm/bin/rdci --version 2>/dev/null || rdci --version
```

## One-Shot Command

For use in scripts or as a single terminal command (DEB systems with gRPC):

```bash
cd "$WORKSPACE" && \
sudo rm -rf build && \
mkdir -p build && cd build && \
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGRPC_ROOT="${GRPC_ROOT:-/opt/grpc}" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
make -j "$(nproc)" && \
make package && \
sudo dpkg --remove --force-depends rdc-tests 2>/dev/null || true && \
sudo dpkg --remove --force-depends rdc 2>/dev/null || true && \
sudo dpkg -i rdc*99999-local_amd64.deb && \
rdci --version
```

## Output

On success, capture and report:
- **Build time** (cmake + make duration)
- **Package files** produced (names + sizes)
- **Installed version** (output of `rdci --version`)
- **Any warnings** from cmake or make (even if build succeeded) — pass these to the build and tests subagents

On failure, capture and report:
- **Which step failed** (configure, build, package, install)
- **Full error output** from the failing command
- **This is ❌ BLOCKING** — a build failure stops the review

## RPM Variant

On RPM-based systems:

```bash
# Uninstall
sudo rpm -e rdc-tests 2>/dev/null || true
sudo rpm -e rdc 2>/dev/null || true

# Install
sudo rpm -Uvh --force build/rdc*99999-local*x86_64.rpm
```

Detect which to use: `command -v dpkg >/dev/null && echo deb || echo rpm`

## Notes

- `BUILD_STANDALONE=OFF` skips rdcd/rdci but still builds librdc.so — useful when gRPC is not available
- gRPC compilation is slow — strongly recommend ccache: `cmake .. -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ...`
- If AMDSMI is not installed at `/opt/rocm`, set `-DCMAKE_PREFIX_PATH` to the AMDSMI install prefix
