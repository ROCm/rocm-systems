---
name: cuid-build-install
description: "Build and install cuid from source. Use when: building locally, installing before tests, pre-review build verification, build + install + verify."
---

# Build & Install cuid

Builds cuid from source and installs locally. Used by the review agent as a pre-step before dispatching subagents, and can be invoked independently.

## Prerequisites

- CMake, make, and a C/C++ toolchain installed
- `sudo` access for install
- Working directory must be the cuid workspace root

## Build Commands

All commands assume `$WORKSPACE` is the cuid workspace root (where `CMakeLists.txt` lives).
The default install prefix is `${ROCM_DIR}/core`, which is `/opt/rocm/core` unless
`-DROCM_DIR` or `-DCMAKE_INSTALL_PREFIX` says otherwise.

### Step 1: Uninstall Previous

Only possible from the build tree that installed it, since it reads `install_manifest.txt`:

```bash
cd "$WORKSPACE/build" && sudo make uninstall 2>/dev/null || true
```

### Step 2: Clean

```bash
cd "$WORKSPACE"
sudo rm -rf build
```

### Step 3: Configure & Build

```bash
mkdir -p build && cd build
cmake ..
make -j "$(nproc)"
sudo make install
```

### Step 4: Verify

```bash
ls /opt/rocm/core/lib/libamdcuid_static.a \
   /opt/rocm/core/include/amdcuid/amd_cuid.h \
   /opt/rocm/core/lib/cmake/amdcuid/amdcuid-config.cmake \
   /usr/lib/tmpfiles.d/amdcuid.conf
```

## One-Shot Command

For use in scripts or as a single terminal command:

```bash
cd "$WORKSPACE" && \
(cd build 2>/dev/null && sudo make uninstall 2>/dev/null || true) && \
sudo rm -rf build && \
mkdir -p build && cd build && \
cmake .. && \
make -j "$(nproc)" && \
sudo make install && \
ls /opt/rocm/core/lib/libamdcuid_static.a
```

## Output

On success, capture and report:
- **Build time** (cmake + make duration)
- **Installed files** (the archive, header and CMake package under `/opt/rocm/core`, and the tmpfiles.d rule)
- **Any warnings** from cmake or make (even if build succeeded)

On failure, capture and report:
- **Which step failed** (configure, build, install)
- **Full error output** from the failing command
- **This is ❌ BLOCKING** — a build failure stops the review
