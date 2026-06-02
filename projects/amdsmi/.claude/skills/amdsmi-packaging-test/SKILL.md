---
name: amdsmi-packaging-test
description: "Integration testing for amd-smi Python packaging. Use when: testing package installs, multi-OS testing, manylinux wheel builds, offline install verification, glibc/Python version compatibility, post-install smoke tests."
---

# Packaging & Integration Test — amd-smi

End-to-end testing of amd-smi package builds, installs, and Python imports across multiple OS containers. Covers offline (system package) installs, wheel builds, and manylinux compatibility.

## When to Use

- After changes to packaging files (`DEBIAN/postinst.in`, `RPM/post.in`, `py-interface/CMakeLists.txt`, `src/CMakeLists.txt`)
- After changes to Python wrapper/loader (`amdsmi_wrapper.py`, `amdsmi_init.py`)
- After changes to wheel build tooling (`tools/build_wheel_debian.py`, `tools/build_wheel_rpm.py`)
- After changes to `tools/generator.py` or `py-interface/setup.py.in` / `py-interface/pyproject.toml.in`
- When reviewing PRs that touch `BUILD_PYTHON_WHEEL` cmake paths
- Before merging any Python packaging PR

## Prerequisites

- Docker installed and running
- Build artifacts available (run `amdsmi-build-install` skill first, or use manylinux build below)
- `$WORKSPACE` is the amd-smi workspace root

---

## 1. Manylinux Container Build

Build packages inside a manylinux container to produce both RPM and DEB artifacts with a known glibc baseline.

```bash
docker run --rm \
  -v "$WORKSPACE:/src" \
  quay.io/pypa/manylinux_2_28_x86_64 \
  bash -c '
set -e
yum install -y -q libdrm-devel rpm-build dpkg cmake3 ninja-build python3-pip 2>&1 | tail -3
python3 -m pip install --quiet "setuptools>=59.0" wheel

cd /src
rm -rf build-manylinux
cmake -S . -B build-manylinux \
    -G Ninja \
    -DBUILD_TESTS=OFF \
    -DENABLE_LDCONFIG=OFF \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCPACK_GENERATOR="RPM;DEB" 2>&1 | tail -15
# NOTE: ENABLE_LDCONFIG=OFF is intentional. Real users typically do not enable
# ldconfig for amd-smi, and leaving it ON during testing masks bugs in the
# wrapper's library-resolution logic (the linker becomes a safety net that
# hides path-resolution failures). Test the no-ldconfig path that real
# deployments use.

cmake --build build-manylinux -j"$(nproc)" 2>&1 | tail -15
cd build-manylinux && cpack 2>&1 | tail -10
ls -la *.rpm *.deb 2>/dev/null
'
```

**Expected outputs:**
- `amd-smi-lib_<version>_amd64.deb`
- `amd-smi-lib-<version>.el8.x86_64.rpm`
- `amd-smi-lib-tests_<version>_amd64.deb`  (if `-DBUILD_TESTS=ON`)
- `amd-smi-lib-tests-<version>.el8.x86_64.rpm`  (if `-DBUILD_TESTS=ON`)

**Key fact:** manylinux_2_28 targets glibc 2.28. Packages built here are compatible with RHEL 8+, Ubuntu 20.04+, SLES 15+.

---

## 2. Multi-OS Install Testing

Test package installation and Python import across multiple OS containers. This is the core validation: packages must install and `import amdsmi` must succeed on every target OS.

### Target Matrix

This matrix mirrors `.github/workflows/amdsmi-build.yml` (`os:` lists at lines 28, 142, 366, 568). The CI workflow resolves images from GitHub repo variables (`vars.<OS>_DOCKER_IMAGE`) which point at AMD-internal registries. Use `gh variable get <LABEL>_DOCKER_IMAGE` to fetch the CI image (requires `gh auth login` against `github.com/ROCm/rocm-systems`, read access to the repo's variables, and AMD VPN); off-VPN or unauthenticated callers get the public fallback below.

| CI label | Public fallback | Format | Min Python | Min glibc |
|----------|-----------------|--------|------------|-----------|
| `Ubuntu20` | `ubuntu:20.04` | DEB | 3.8 | 2.31 |
| `Ubuntu22` | `ubuntu:22.04` | DEB | 3.10 | 2.35 |
| `Ubuntu24` | `ubuntu:24.04` | DEB | 3.12 | 2.39 |
| `Debian10` | `debian:10` (needs `archive.debian.org` sources) | DEB | 3.7 | 2.28 |
| `SLES` | `registry.suse.com/bci/bci-base:15.6` | RPM | 3.6 | 2.31 |
| `RHEL8` | `registry.access.redhat.com/ubi8/ubi:latest` | RPM | 3.6.8 | 2.28 |
| `RHEL9` | `registry.access.redhat.com/ubi9/ubi:latest` | RPM | 3.9 | 2.34 |
| `RHEL10` | `registry.access.redhat.com/ubi10/ubi:latest` | RPM | 3.12 | 2.39 |
| `AzureLinux3` | `mcr.microsoft.com/azurelinux/base/core:3.0` | RPM | 3.12 | 2.38 |
| `AlmaLinux8` | `almalinux:8` | RPM | 3.6 | 2.28 |

### Multi-OS Test Script

Set `$BUILD_DIR` to the directory containing built `.rpm` and `.deb` files. The `ci_image` helper prefers the CI image (via `gh variable get`) and falls back to the public image if `gh` isn't authenticated or the variable isn't set.

```bash
BUILD_DIR="$WORKSPACE/build-manylinux"  # or $WORKSPACE/build
RPM_NAME=$(ls "$BUILD_DIR"/amd-smi-lib-*[!tests]*.rpm 2>/dev/null | head -1 | xargs basename)
DEB_NAME=$(ls "$BUILD_DIR"/amd-smi-lib_*.deb 2>/dev/null | grep -v tests | head -1 | xargs basename)

# Resolve a CI image: prefer vars.<LABEL>_DOCKER_IMAGE, else public fallback.
ci_image() {
  local label=$1 fallback=$2 img
  img=$(gh variable get "${label}_DOCKER_IMAGE" 2>/dev/null) || img=""
  echo "${img:-$fallback}"
}

for spec in \
  "ubuntu20|Ubuntu20|ubuntu:20.04|deb|$DEB_NAME" \
  "ubuntu22|Ubuntu22|ubuntu:22.04|deb|$DEB_NAME" \
  "ubuntu24|Ubuntu24|ubuntu:24.04|deb|$DEB_NAME" \
  "debian10|Debian10|debian:10|deb|$DEB_NAME" \
  "sles|SLES|registry.suse.com/bci/bci-base:15.6|rpm|$RPM_NAME" \
  "rhel8|RHEL8|registry.access.redhat.com/ubi8/ubi:latest|rpm|$RPM_NAME" \
  "rhel9|RHEL9|registry.access.redhat.com/ubi9/ubi:latest|rpm|$RPM_NAME" \
  "rhel10|RHEL10|registry.access.redhat.com/ubi10/ubi:latest|rpm|$RPM_NAME" \
  "azurelinux3|AzureLinux3|mcr.microsoft.com/azurelinux/base/core:3.0|rpm|$RPM_NAME" \
  "almalinux8|AlmaLinux8|almalinux:8|rpm|$RPM_NAME"; do

  IFS='|' read -r name label fallback fmt pkg <<< "$spec"
  img=$(ci_image "$label" "$fallback")
  echo ""
  echo "##### $name ($img / $fmt / $pkg) #####"
  docker run --rm -e PKG="$pkg" -e FMT="$fmt" \
    -v "$BUILD_DIR:/work:ro" \
    "$img" bash -c '
      set +e
      if [ "$FMT" = "rpm" ]; then
        yum install -y -q python3 python3-pip 2>&1 | tail -1
      else
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq 2>&1 | tail -1
        apt-get install -y -qq python3 python3-pip 2>&1 | tail -1
      fi
      python3 --version
      python3 -c "import setuptools; print(\"setuptools:\", setuptools.__version__)" 2>&1
      python3 -m pip --version

      echo ""
      echo "--- install pkg ---"
      if [ "$FMT" = "rpm" ]; then
        rpm -ivh --nodeps /work/$PKG 2>&1 | tail -15
      else
        dpkg --force-depends -i /work/$PKG 2>&1 | tail -15
      fi

      echo ""
      echo "--- pip list amdsmi ---"
      python3 -m pip list --disable-pip-version-check 2>/dev/null | grep -i amdsmi || echo "amdsmi NOT in pip list"

      echo ""
      echo "--- import amdsmi ---"
      python3 -c "import amdsmi; print(\"OK:\", amdsmi.__file__)" 2>&1 | tail -3

      echo ""
      echo "--- amdsmi_init smoke ---"
      python3 -c "
import amdsmi
try:
    amdsmi.amdsmi_init()
    print(\"init OK\")
    handles = amdsmi.amdsmi_get_processor_handles()
    print(\"handles:\", len(handles))
    amdsmi.amdsmi_shut_down()
except Exception as e:
    print(\"init result:\", type(e).__name__, str(e)[:80])
" 2>&1 | tail -5
    '
done
```

### Pass/Fail Criteria

For each OS container, verify **all four checkpoints**:

| # | Checkpoint | Pass | Fail |
|---|-----------|------|------|
| 1 | Package installs | No errors from `dpkg -i` / `rpm -ivh` | ❌ Install error |
| 2 | `pip list` shows amdsmi | `amdsmi <version>` in output | ❌ Missing from pip |
| 3 | `import amdsmi` succeeds | Prints `OK: <path>` | ❌ ImportError |
| 4 | `amdsmi_init()` doesn't crash | Prints `init OK` (0 handles OK in container — no GPU) | ❌ Python crash/traceback |

**Expected in containers without GPU:** `init OK` with `handles: 0`. The `libdrm_amdgpu.so.1` warning is normal and expected — it means the library gracefully handles missing GPU drivers.

**Critical regressions to watch for:**
- `ModuleNotFoundError: No module named 'amdsmi'` — postinst wheel install failed
- `ImportError: libamd_smi.so` — library not found by wrapper
- `SyntaxError` on Python 3.6.8 — f-strings OK, but walrus operator (`:=`) or `Union[X | Y]` breaks

---

## 3. Offline vs Online Install Testing

Test that package install works without internet access (offline) — the postinst script must use the bundled wheel, not download from PyPI.

### Offline Install Test

```bash
docker run --rm --network none \
  -e PKG="$DEB_NAME" \
  -v "$BUILD_DIR:/work:ro" \
  ubuntu:22.04 bash -c '
    export DEBIAN_FRONTEND=noninteractive
    # NOTE: cannot apt-get update (no network) — use what is in the base image
    dpkg --force-depends -i /work/$PKG 2>&1 | tail -15
    python3 -c "import amdsmi; print(\"OK:\", amdsmi.__file__)" 2>&1
'
```

**Pass:** `import amdsmi` succeeds. The postinst script installs from the bundled wheel at `/opt/rocm/share/amd_smi/amdsmi-*.whl`.

**Fail:** Any `pip install` step tries to reach the network → hangs or errors. This means the postinst is not using `--no-index --find-links` or equivalent.

### Online Install Test (comparison)

Same as section 2 — the standard multi-OS test runs with network access. Compare behavior to offline: both should produce identical `pip list` output.

---

## 4. Wheel Build Testing

Test the standalone wheel build scripts (`tools/build_wheel_debian.py`, `tools/build_wheel_rpm.py`) in their intended environments.

### Debian Wheel Build

```bash
docker run --rm \
  -v "$WORKSPACE:/src" \
  quay.io/pypa/manylinux_2_28_x86_64 \
  bash -c '
    cd /src
    python3 tools/build_wheel_debian.py --project-dir /src --all-pythons 2>&1 | tail -30
    ls -la /src/wheelhouse/*.whl 2>/dev/null || echo "NO WHEELS PRODUCED"
'
```

### RPM Wheel Build

```bash
docker run --rm \
  -v "$WORKSPACE:/src" \
  quay.io/pypa/manylinux_2_28_x86_64 \
  bash -c '
    cd /src
    python3 tools/build_wheel_rpm.py --project-dir /src 2>&1 | tail -30
    ls -la /src/wheelhouse/*.whl 2>/dev/null || echo "NO WHEELS PRODUCED"
'
```

### Wheel Validation

After building wheels, verify:

```bash
# Check wheel metadata
python3 -m zipfile -l wheelhouse/*.whl | head -20

# Check platform tags (should be manylinux_2_28 after auditwheel repair)
ls wheelhouse/*.whl

# Install and test
pip install wheelhouse/amdsmi-*.whl
python3 -c "import amdsmi; print(amdsmi.__file__)"
```

### System Package + Wheel Conflict Matrix

The dual-library design (`BUILD_PYTHON_WHEEL`) ships `libamd_smi.so` in the system package and a SONAME-renamed `libamd_smi_python.so` inside the wheel. The two MUST be safely co-installable in the same Python process. Run [tests/run_amdsmi_pkg_conflict_test.py](tests/run_amdsmi_pkg_conflict_test.py) to validate all four scenarios in one container:

```bash
python3 tests/run_amdsmi_pkg_conflict_test.py \
    --build-dir build-manylinux \
    --wheel-dir build-manylinux/py-interface/python_package \
    --image ubuntu:22.04
```

The driver verifies, in order:

| # | Scenario | Pass criterion |
|---|----------|----------------|
| 1 | system pkg ALONE | `_loaded_lib_path` is `/opt/rocm/lib64/libamd_smi.so`; `amdsmi_init` runs |
| 2 | wheel ALONE | `_loaded_lib_path` is `<site>/amdsmi/libamd_smi_python.so`; `amdsmi_init` runs |
| 3 | system pkg + wheel | both .so files present, SONAMEs differ (`libamd_smi.so.<X>` vs `libamd_smi_python.so`); `amdsmi_init` runs |
| 4 | forced dual-load via `ctypes.CDLL(..., RTLD_GLOBAL)` | `amdsmi_init` + `amdsmi_shut_down` complete with no SIGSEGV |

**Critical regression to watch for:** if scenario 3's SONAME assertion ever fails, the two-library guarantee is broken and the wheel is one `pip install` away from segfaulting any system that already has the .deb/.rpm installed. This is the failure mode `BUILD_PYTHON_WHEEL` exists to prevent.

> Note: requires a wheel built with `-DBUILD_PYTHON_WHEEL=ON` (the wheel-build scripts already do this; a default `cmake .. && make` does NOT).

---

## 5. Python Version Compatibility

The amdsmi package must work across Python 3.6.8 through 3.12+.

### Version-Specific Concerns

| Python | OS | Concern |
|--------|----|---------|
| 3.6.8 | RHEL 8, SLES 15 | No walrus `:=`, no `Union[X \| Y]`, no `str.removeprefix()`. `setuptools` may be very old (39.x). |
| 3.8 | Ubuntu 20.04 | Oldest Debian-based target. `pip` may need `--break-system-packages` on newer pip versions. |
| 3.10 | Ubuntu 22.04 | Middle ground — generally trouble-free. |
| 3.11+ | SLES (via zypper) | May need explicit zypper install of `python311` package. |
| 3.12 | Ubuntu 24.04, AzureLinux 3 | `distutils` removed — `setuptools` must provide it. |

### Quick Syntax Check

Verify no 3.7+ syntax leaks into code that must run on 3.6:

```bash
python3.6 -c "import py_compile; py_compile.compile('py-interface/amdsmi_wrapper.py', doraise=True)"
python3.6 -c "import py_compile; py_compile.compile('amdsmi_cli/amdsmi_init.py', doraise=True)"
```

Or in a container:

```bash
docker run --rm -v "$WORKSPACE:/src:ro" \
  registry.access.redhat.com/ubi8/ubi:latest \
  python3 -c "
import py_compile, sys
for f in ['py-interface/amdsmi_wrapper.py', 'amdsmi_cli/amdsmi_init.py', 'tools/generator.py']:
    try:
        py_compile.compile('/src/' + f, doraise=True)
        print(f'OK: {f}')
    except py_compile.PyCompileError as e:
        print(f'FAIL: {f}: {e}')
        sys.exit(1)
"
```

---

## 6. glibc Compatibility

Shared libraries built on a newer glibc will fail to load on older systems.

### Minimum glibc Target: 2.28 (manylinux_2_28)

This covers RHEL 8+, Ubuntu 20.04+, Debian 10+, SLES 15+.

### Verify glibc Requirements

After building `libamd_smi.so`:

```bash
# Check GLIBC version requirements of the built library
objdump -T build/libamd_smi.so | grep GLIBC | sed 's/.*GLIBC_//' | sort -Vu
# or
readelf -V build/libamd_smi.so | grep GLIBC | awk '{print $3}' | sort -Vu
```

**Pass:** Highest GLIBC version ≤ 2.28.
**Fail:** Any GLIBC version > 2.28 → library won't load on RHEL 8 / Debian 10.

### Build in manylinux for safety

Building inside `quay.io/pypa/manylinux_2_28_x86_64` guarantees glibc 2.28 ceiling. Always use the manylinux container build (section 1) for release artifacts.

---

## 7. Build Automation Script Testing

Test `tests/amdsmi_build/run_amdsmi_build.py` which mirrors the CI workflow locally:

```bash
# Ubuntu / Debian
sudo python3 tests/amdsmi_build/run_amdsmi_build.py --package-manager apt --os-label Ubuntu22

# RHEL / AlmaLinux
sudo python3 tests/amdsmi_build/run_amdsmi_build.py --package-manager dnf --package-format rpm --os-label RHEL9

# SLES (auto-bootstraps Python 3.11 if system Python is 3.6)
sudo python3 tests/amdsmi_build/run_amdsmi_build.py --package-manager zypper --package-format rpm --os-label SLES
```

The script should:
1. Configure & build
2. Package (DEB or RPM)
3. Install the package
4. Run `amd-smi version`
5. Run `import amdsmi` smoke test
6. Optionally build and verify wheel

---

## Quick Reference: One-Shot Multi-OS Validation

Fastest way to validate packaging changes — run after `amdsmi-build-install` skill or manylinux build:

```bash
BUILD_DIR="$WORKSPACE/build-manylinux"
# Expand to all targets as needed; minimum: rhel8 + ubuntu22 + ubuntu24
for spec in \
  "ubuntu22|Ubuntu22|ubuntu:22.04|deb" \
  "rhel9|RHEL9|registry.access.redhat.com/ubi9/ubi:latest|rpm" \
  "azurelinux3|AzureLinux3|mcr.microsoft.com/azurelinux/base/core:3.0|rpm"; do
  IFS='|' read -r name label fallback fmt <<< "$spec"
  img=$(gh variable get "${label}_DOCKER_IMAGE" 2>/dev/null) || img="$fallback"
  pkg=$(ls "$BUILD_DIR"/*amd-smi-lib*[!tests]*.$fmt 2>/dev/null | head -1)
  [[ -z "$pkg" ]] && echo "SKIP $name: no .$fmt package" && continue
  echo "=== $name ==="
  docker run --rm -v "$pkg:/pkg.$(basename "$pkg" | rev | cut -d. -f1 | rev):ro" "$img" bash -c '
    set +e; fmt="'"$fmt"'"
    if [[ "$fmt" == "rpm" ]]; then yum install -y -q python3 2>&1|tail -1; rpm -ivh --nodeps /pkg.rpm 2>&1|tail -5
    else export DEBIAN_FRONTEND=noninteractive; apt-get update -qq 2>&1|tail -1; apt-get install -y -qq python3 python3-pip 2>&1|tail -1; dpkg --force-depends -i /pkg.deb 2>&1|tail -5; fi
    python3 -c "import amdsmi; print(\"PASS:\", amdsmi.__file__)" 2>&1
  '
done
```

---

## Severity (when used by review agents)

| Marker | Use for |
|--------|---------|
| **❌ BLOCKING** | Import fails on any target OS, package won't install, offline install broken |
| **⚠️ IMPORTANT** | Missing OS from test matrix, glibc > 2.28, Python 3.6 syntax error |
| **💡 SUGGESTION** | Additional OS targets, auditwheel repair for manylinux tags |
| **📋 FUTURE WORK** | ARM64 packaging, conda packaging |
