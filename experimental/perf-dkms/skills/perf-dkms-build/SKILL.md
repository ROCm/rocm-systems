---
name: perf-dkms-build
description: Build, install, and test perf-dkms kernel module on remote target
trigger:
  paths: ["experimental/perf-dkms/**"]
  keywords: ["build perf-dkms", "build perf dkms", "perf-dkms", "amdgpu_pmu"]
---

Build, install, and test the perf-dkms kernel module on a remote target machine.

## Overview

perf-dkms is a Linux kernel DKMS module that exposes AMD GPU hardware performance counters through the Linux perf subsystem. It produces a kernel module (`amdgpu_pmu.ko`) via a CMake-wrapped Kbuild system.

**Source location:** `experimental/perf-dkms/` on `develop` branch, or `projects/perf-dkms/` on older branches. After cloning on the remote, verify which path exists and use that throughout.

**Module dependency:** The `amdgpu_pmu` module depends on the `amdgpu` kernel driver. Ensure `amdgpu` is loaded before attempting `insmod`.

## User Prompts

**Before proceeding, ask the user:**

1. **Remote target machine?** (required, `user@hostname` format, e.g. `ben@192.168.0.66`)
2. **Remote staging directory?** (default: `~/tmp/perf-dkms-build`)
3. **Kernel version override?** (leave blank for auto-detect via `uname -r` on remote)
4. **Build type?** (Debug or Release, default: Debug)
5. **Path to perf binary?** (default: `perf` from PATH. On custom kernels, perf is often built from the kernel source tree -- e.g. `~/linux/tools/perf/perf` or `/path/to/kernel-build/tools/perf/perf`. Ask the user where it is.)

## Step 1: Stage and Push

Ensure current perf-dkms changes are on a branch that the remote can access:

```bash
# Record current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"

# Check for uncommitted changes
git status experimental/perf-dkms/ projects/perf-dkms/ 2>/dev/null

# If there are uncommitted changes, warn the user and ask to commit or stash
# Push to origin so the remote can checkout
git push origin $BRANCH
```

## Step 2: Remote Setup

SSH to the target and prepare the staging directory:

```bash
# Get the git remote URL
REMOTE_URL=$(git remote get-url origin)

# Create staging directory
ssh $TARGET "mkdir -p $STAGING_DIR"

# Clone or update the repo
ssh $TARGET "
  if [ -d $STAGING_DIR/rocm-systems/.git ]; then
    cd $STAGING_DIR/rocm-systems
    git fetch origin
    git checkout $BRANCH
    git pull origin $BRANCH
  else
    cd $STAGING_DIR
    git clone $REMOTE_URL rocm-systems
    cd rocm-systems
    git checkout $BRANCH
  fi
"
```

**Important:** Determine the correct git remote URL by running `git remote get-url origin` locally before the SSH clone step.

## Step 3: Prerequisite Check (on remote)

Run these checks on the remote machine via SSH:

```bash
ssh $TARGET "
  echo '=== Build Tools ==='
  which cmake && cmake --version || echo 'MISSING: cmake (apt install cmake)'
  which make || echo 'MISSING: make (apt install build-essential)'
  which gcc && gcc --version | head -1 || echo 'MISSING: gcc (apt install build-essential)'

  echo ''
  echo '=== Kernel Headers ==='
  KVER=\${KERNEL_VERSION_OVERRIDE:-\$(uname -r)}
  echo \"Kernel version: \$KVER\"
  if [ -f /lib/modules/\$KVER/build/Makefile ]; then
    echo \"Kernel headers: OK (/lib/modules/\$KVER/build/)\"
  else
    echo \"MISSING: Kernel headers for \$KVER\"
    echo \"Install with: sudo apt install linux-headers-\$KVER\"
  fi

  echo ''
  echo '=== Kernel Patch Check ==='
  if [ -f /lib/modules/\$KVER/build/Module.symvers ]; then
    KFD_EXPORTS=\$(grep -c 'kfd_' /lib/modules/\$KVER/build/Module.symvers 2>/dev/null || echo 0)
    echo \"KFD exports found: \$KFD_EXPORTS\"
    if [ \"\$KFD_EXPORTS\" -eq 0 ]; then
      echo 'WARNING: No KFD exports found in Module.symvers.'
      echo 'The kernel may not have the required perf-dkms patch applied.'
      echo 'The module will compile but may not function fully at runtime.'
    fi
  else
    echo 'WARNING: Module.symvers not found, cannot verify kernel patch'
  fi

  echo ''
  echo '=== perf tools ==='
  # Check user-specified path first, then PATH
  PERF_BIN=\${PERF_PATH:-perf}
  if [ -x \"\$PERF_BIN\" ]; then
    echo \"perf binary: \$PERF_BIN\"
    \$PERF_BIN --version 2>/dev/null || echo '(version check failed)'
  elif which perf >/dev/null 2>&1; then
    echo \"perf binary: \$(which perf)\"
    perf --version 2>/dev/null || echo '(version check failed)'
  else
    echo 'perf not found in PATH.'
    echo 'On custom kernels, perf is often built from the kernel source tree.'
    echo 'Check: ~/linux/tools/perf/perf or /path/to/kernel-build/tools/perf/perf'
    echo 'Provide the path via the PERF_PATH user prompt.'
  fi

  echo ''
  echo '=== amdgpu driver ==='
  lsmod | grep amdgpu && echo 'amdgpu driver: loaded' || echo 'WARNING: amdgpu driver not loaded. Module depends on amdgpu.'
  ls /dev/kfd 2>/dev/null && echo 'KFD device: OK' || echo 'WARNING: /dev/kfd not found'
"
```

If cmake, gcc, or kernel headers are missing, report what's needed and stop. Do not proceed with the build. Missing perf tools or amdgpu warnings are non-blocking -- the module can still be built.

## Step 4: Locate Source and Build (on remote)

First, determine which path the perf-dkms source is at on the checked-out branch:

```bash
ssh $TARGET "
  # Detect source path
  if [ -d $PERF_DKMS_DIR/src ]; then
    PERF_DKMS_DIR=$PERF_DKMS_DIR
  elif [ -d $STAGING_DIR/rocm-systems/projects/perf-dkms/src ]; then
    PERF_DKMS_DIR=$STAGING_DIR/rocm-systems/projects/perf-dkms
  else
    echo 'ERROR: perf-dkms source not found in experimental/ or projects/'
    exit 1
  fi
  echo \"Using: \$PERF_DKMS_DIR\"

  cd \$PERF_DKMS_DIR

  # Clean previous build
  rm -rf build

  # Configure CMake
  cmake -B build \
    -DBUILD_KERNEL_MODULE=ON \
    -DBUILD_USERSPACE_TOOLS=ON \
    -DBUILD_TESTS=ON \
    -DENABLE_DEBUG=ON
  # Add if kernel version override provided:
  # -DKERNEL_VERSION=\$KERNEL_VERSION_OVERRIDE

  # Build
  cmake --build build --parallel \$(nproc)

  # Check what was produced
  echo ''
  echo '=== Build Output ==='
  if [ -f build/src/amdgpu_pmu.ko ]; then
    echo 'Module: build/src/amdgpu_pmu.ko'
    ls -lh build/src/amdgpu_pmu.ko
  elif [ -f build/src/amdgpu_pmu.o ]; then
    echo 'Module: build/src/amdgpu_pmu.o'
    ls -lh build/src/amdgpu_pmu.o
  else
    echo 'ERROR: No module output found!'
    ls -la build/src/ | grep amdgpu_pmu
  fi
"
```

**Note:** Use the detected `$PERF_DKMS_DIR` path for all subsequent steps. On `develop`, the path is `experimental/perf-dkms/`. On older branches, it may be `projects/perf-dkms/`.

## Step 5: Run Userspace Unit Tests (on remote)

These tests validate packet generation and counter registry logic without needing the module loaded. Note: Not all test source files may be built by CMake -- only tests that appear in the CMake configuration will have executables.

```bash
ssh $TARGET "
  cd \$PERF_DKMS_DIR

  echo '=== Userspace Unit Tests ==='
  echo 'Looking for test executables...'
  ls build/src/aql_c/tests/test_* 2>/dev/null || echo 'No test executables found'
  echo ''

  PASS=0; FAIL=0; SKIP=0

  for t in \
    build/src/aql_c/tests/test_pm4 \
    build/src/aql_c/tests/test_pm4_packets \
    build/src/aql_c/tests/test_counter_registry \
    build/src/aql_c/tests/test_packet_generation \
    build/src/aql_c/tests/test_gfx12 \
    build/src/aql_c/tests/test_dimension_decode \
    build/src/aql_c/tests/test_dimension_helpers \
    build/src/aql_c/tests/test_aql_queue; do
    if [ -x \"\$t\" ]; then
      echo \"Running: \$(basename \$t)\"
      if \$t 2>&1; then
        PASS=\$((PASS + 1))
      else
        echo \"  FAIL (exit code: \$?)\"
        FAIL=\$((FAIL + 1))
      fi
    else
      SKIP=\$((SKIP + 1))
    fi
  done

  echo ''
  echo \"Unit test results: \$PASS passed, \$FAIL failed, \$SKIP skipped\"
"
```

## Step 6: Install Module (on remote)

```bash
ssh $TARGET "
  cd $PERF_DKMS_DIR

  # Determine module file
  if [ -f build/src/amdgpu_pmu.ko ]; then
    MODULE=build/src/amdgpu_pmu.ko
  elif [ -f build/src/amdgpu_pmu.o ]; then
    MODULE=build/src/amdgpu_pmu.o
  else
    echo 'ERROR: No module file found'
    exit 1
  fi

  # Unload existing module
  sudo rmmod amdgpu_pmu 2>/dev/null || true
  sleep 1

  # Load the module
  sudo insmod \$MODULE debug_enable=1 timer_period_ms=100
  sleep 2

  # Verify
  echo '=== Module Status ==='
  lsmod | grep amdgpu_pmu || echo 'ERROR: Module not loaded'
  echo ''
  echo '=== Kernel Messages ==='
  dmesg | tail -20
  echo ''
  echo '=== Sysfs Events ==='
  ls /sys/bus/event_source/devices/amdgpu_pmu/events/ 2>/dev/null || echo 'WARNING: No sysfs events directory'
"
```

## Step 7: Run Integration Tests (on remote)

```bash
ssh $TARGET "
  cd $PERF_DKMS_DIR

  PERF_BIN=\${PERF_PATH:-perf}

  echo '=== Perf Integration ==='
  \$PERF_BIN list 2>/dev/null | grep amdgpu_pmu || echo 'No amdgpu_pmu events in perf list'

  echo ''
  echo '=== Basic Counter Test ==='
  sudo \$PERF_BIN stat -e amdgpu_pmu/sq_waves/ sleep 1 2>&1 || echo 'perf stat failed'

  echo ''
  echo '=== Test Scripts ==='
  if [ -x test/load_test.sh ]; then
    echo 'Running load_test.sh...'
    sudo bash test/load_test.sh 2>&1 || echo 'load_test.sh failed'
  fi

  if [ -x test/perf_test.sh ]; then
    echo 'Running perf_test.sh...'
    sudo bash test/perf_test.sh 2>&1 || echo 'perf_test.sh failed'
  fi
"
```

## Step 8: Cleanup (on remote)

```bash
ssh $TARGET "
  sudo rmmod amdgpu_pmu 2>/dev/null || true
  echo 'Module unloaded'
"
```

## Step 9: Report Results

Display a summary of all results:
- Kernel version used (auto-detected or overridden)
- Build status (success/fail, .ko vs .o)
- Unit test results (pass/fail counts)
- Module load status
- Perf event registration status
- Integration test results
- Any warnings from prerequisite checks

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No rule to make target 'modules'` | Kernel headers missing: `apt install linux-headers-$(uname -r)` |
| `insmod: ERROR: could not insert module: File exists` | Module already loaded; run `sudo rmmod amdgpu_pmu` first |
| `insmod: ERROR: could not insert module` | Check `dmesg`; may need kernel patch, conflicting module, or missing `amdgpu` dependency |
| `No events found in perf list` | Module loaded but PMU registration failed; check `dmesg` for errors |
| `Permission denied on perf stat` | Use `sudo` or set `kernel.perf_event_paranoid=0` |
| SSH connection refused | Verify target is reachable: `ssh $TARGET echo ok` |
| Kernel headers version mismatch | Install correct headers: `apt install linux-headers-$KERNEL_VERSION` |
| `cmake: command not found` on remote | Install cmake: `apt install cmake` |
| Build fails with missing headers | Source may be stale; `git fetch && git checkout` again |
| Module loads but no GPU detected | Verify GPU is present: `lspci \| grep -i amd` and KFD is working: `ls /dev/kfd` |
| `vermagic` mismatch on insmod | Kernel version used for build doesn't match running kernel |
| perf not found | On custom kernels, perf is often built from the kernel source tree (e.g. `~/linux/tools/perf/perf`). Ask the user for the path. |
| Unit tests crash (segfault/abort) | Known issue: some userspace tests have memory bugs; module itself still works |
| test scripts fail with wrong paths | Test scripts assume specific relative paths; run from project root |
| `perf-dkms not found` on remote | Check both `experimental/perf-dkms/` and `projects/perf-dkms/` |

Report all results to the user.
