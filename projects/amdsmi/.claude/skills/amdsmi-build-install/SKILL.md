---
name: amdsmi-build-install
description: "Build and install amd-smi from source. Use when: building locally, installing before tests, pre-review build verification, build + install + verify."
---

# Build & Install amd-smi

Builds amd-smi from source, packages it, and installs locally. Used by the review agent as a pre-step before dispatching subagents, and can be invoked independently.

## Prerequisites

- CMake, make, and a C/C++ toolchain installed
- `sudo` access for package install
- Working directory must be the amd-smi workspace root

## Build Commands

All commands assume `$WORKSPACE` is the amd-smi workspace root (where `CMakeLists.txt` lives).

### Step 1: Clean

```bash
cd "$WORKSPACE"
sudo rm -rf esmi_ib_library esmi_ib_library_temp build
```

### Step 2: Configure & Build

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j "$(nproc)"
make package
```

### Step 3: Uninstall Previous

```bash
sudo python3 -m pip uninstall -y amdsmi 2>/dev/null || true
sudo dpkg --remove --force-depends amd-smi-lib-tests 2>/dev/null || true
sudo dpkg --remove --force-depends amd-smi-lib 2>/dev/null || true
sudo rm -f /usr/local/bin/amd-smi
```

### Step 4: Install

```bash
sudo dpkg -i build/amd-smi-lib*99999-local_amd64.deb
sudo dpkg -i build/amd-smi-lib-tests*99999-local_amd64.deb
sudo ln -sf /opt/rocm/bin/amd-smi /usr/local/bin/amd-smi
```

### Step 5: Verify

Run all verification checks. Every check must pass — any failure is ❌ BLOCKING.

```bash
# Version check — hash must match HEAD of the branch/PR
VERSION_OUTPUT=$(amd-smi version)
echo "$VERSION_OUTPUT"
INSTALLED_HASH=$(echo "$VERSION_OUTPUT" | grep -oP '\+\K[0-9a-f]+')
EXPECTED_HASH=$(git rev-parse --short=10 HEAD)
if [[ "$INSTALLED_HASH" != "$EXPECTED_HASH" ]]; then
  echo "❌ Hash mismatch: installed=$INSTALLED_HASH expected=$EXPECTED_HASH"
  exit 1
fi

# Library is loadable
python3 -c "import amdsmi; print('Python import OK')"

# Shared library exists
ls -la /opt/rocm/lib/libamd_smi.so

# CLI is on PATH
which amd-smi
```

If the installed version string doesn't match the version from `cmake_modules/version_util.sh`, report as ⚠️ IMPORTANT.

## One-Shot Command

For use in scripts or as a single terminal command:

```bash
cd "$WORKSPACE" && \
sudo rm -rf esmi_ib_library esmi_ib_library_temp build && \
mkdir -p build && cd build && \
cmake .. -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
make -j "$(nproc)" && \
make package && \
sudo python3 -m pip uninstall -y amdsmi 2>/dev/null || true && \
sudo dpkg --remove --force-depends amd-smi-lib-tests 2>/dev/null || true && \
sudo dpkg --remove --force-depends amd-smi-lib 2>/dev/null || true && \
sudo rm -f /usr/local/bin/amd-smi && \
sudo dpkg -i amd-smi-lib*99999-local_amd64.deb && \
sudo dpkg -i amd-smi-lib-tests*99999-local_amd64.deb && \
sudo ln -sf /opt/rocm/bin/amd-smi /usr/local/bin/amd-smi && \
amd-smi version
```

## Output

On success, capture and report:
- **Build time** (cmake + make duration)
- **Package files** produced (names + sizes)
- **Installed version** (output of `amd-smi version`)
- **Any warnings** from cmake or make (even if build succeeded)

On failure, capture and report:
- **Which step failed** (configure, build, package, install)
- **Full error output** from the failing command
- **This is ❌ BLOCKING** — a build failure stops the review

## RPM Variant

On RPM-based systems, replace the install steps:

```bash
# Uninstall
sudo rpm -e amd-smi-lib-tests 2>/dev/null || true
sudo rpm -e amd-smi-lib 2>/dev/null || true
sudo rm -f /usr/local/bin/amd-smi

# Install
sudo rpm -Uvh --force build/amd-smi-lib*99999-local*x86_64.rpm
sudo rpm -Uvh --force build/amd-smi-lib-tests*99999-local*x86_64.rpm
sudo ln -sf /opt/rocm/bin/amd-smi /usr/local/bin/amd-smi
```

Detect which to use: `command -v dpkg >/dev/null && echo deb || echo rpm`