---
name: cuid-build-install
description: "Build and install cuid from source. Use when: building locally, installing before tests, pre-review build verification, build + install + verify."
---

# Build & Install cuid

Builds cuid from source and installs locally. Used by the review agent as a pre-step before dispatching subagents, and can be invoked independently.

## Prerequisites

- CMake, make, and a C/C++ toolchain installed
- `sudo` access for package install
- Working directory must be the cuid workspace root

## Build Commands

All commands assume `$WORKSPACE` is the cuid workspace root (where `CMakeLists.txt` lives).

### Step 1: Clean

```bash
cd "$WORKSPACE"
sudo rm -rf build
```

### Step 2: Uninstall Previous

```bash
sudo rm -rf /opt/amdcuid 2>/dev/null || true
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
/opt/amdcuid/bin/amdcuid_tool --version
```

## One-Shot Command

For use in scripts or as a single terminal command:

```bash
cd "$WORKSPACE" && \
sudo rm -rf build && \
mkdir -p build && cd build && \
sudo apt remove -y amdcuid 2>/dev/null || true && \
cmake .. && \
make -j "$(nproc)" && \
sudo make install && \
/opt/amdcuid/bin/amdcuid_tool --version
```

## Output

On success, capture and report:
- **Build time** (cmake + make duration)
- **Package files** produced (names + sizes)
- **Installed version** (output of `/opt/amdcuid/bin/amdcuid_tool --version`)
- **Any warnings** from cmake or make (even if build succeeded)

On failure, capture and report:
- **Which step failed** (configure, build, package, install)
- **Full error output** from the failing command
- **This is ❌ BLOCKING** — a build failure stops the review
