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

## CI Build Matrix

The local build commands below target one host. CI builds the same artifact across the matrix in [.github/workflows/amdsmi-build.yml](.github/workflows/amdsmi-build.yml) (`os:` lists at lines 28, 142, 366, 568). Image strings come from GitHub repo variables (`vars.<OS>_DOCKER_IMAGE`) and resolve to AMD-internal registries. Run `gh variable get <LABEL>_DOCKER_IMAGE` to fetch a CI image (requires `gh auth login` against `github.com/ROCm/rocm-systems` AND read access to the repo's variables AND on AMD VPN); off-VPN or unauthenticated callers get the public fallback in the table below.

| CI label | Public fallback | Format |
|----------|-----------------|--------|
| `Ubuntu20` | `ubuntu:20.04` | DEB |
| `Ubuntu22` | `ubuntu:22.04` | DEB |
| `Ubuntu24` | `ubuntu:24.04` | DEB |
| `Debian10` | `debian:10` | DEB |
| `SLES` | `registry.suse.com/bci/bci-base:15.6` | RPM |
| `RHEL8` | `registry.access.redhat.com/ubi8/ubi:latest` | RPM |
| `RHEL9` | `registry.access.redhat.com/ubi9/ubi:latest` | RPM |
| `RHEL10` | `registry.access.redhat.com/ubi10/ubi:latest` | RPM |
| `AzureLinux3` | `mcr.microsoft.com/azurelinux/base/core:3.0` | RPM |
| `AlmaLinux8` | `almalinux:8` | RPM |

Resolve the CI image with:

```bash
img=$(gh variable get UBUNTU22_DOCKER_IMAGE 2>/dev/null) || img=ubuntu:22.04
```

For multi-OS install validation, see the [amdsmi-packaging-test](../amdsmi-packaging-test/SKILL.md) skill.

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