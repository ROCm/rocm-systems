# perf-dkms-build Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Claude Code skill that builds, installs, and tests the perf-dkms kernel module on a remote target machine, then update documentation based on discoveries.

**Architecture:** Single `SKILL.md` file following existing build skill patterns (hsa-build, hip-build). The skill handles: push branch to origin, SSH to remote to checkout/build/install/test, report results. No hardcoded hosts or paths.

**Tech Stack:** Bash (SSH, git, cmake, make, insmod, perf), YAML frontmatter for skill metadata.

**Spec:** `docs/superpowers/specs/2026-04-13-perf-dkms-build-skill-design.md`

---

### Task 1: Create the perf-dkms-build Skill File

**Files:**
- Create: `/home/bewelton/skills/perf-dkms-build/SKILL.md`

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p /home/bewelton/skills/perf-dkms-build
```

- [ ] **Step 2: Write the SKILL.md file**

Create `/home/bewelton/skills/perf-dkms-build/SKILL.md` with this content:

```markdown
---
name: perf-dkms-build
description: Build, install, and test perf-dkms kernel module on remote target
trigger:
  paths: ["experimental/perf-dkms/**"]
  keywords: ["build perf-dkms", "build perf dkms", "perf-dkms", "amdgpu_pmu"]
---

Build, install, and test the perf-dkms kernel module on a remote target machine.

## Overview

perf-dkms is a Linux kernel DKMS module at `experimental/perf-dkms/` that exposes AMD GPU hardware performance counters through the Linux perf subsystem. It produces a kernel module (`amdgpu_pmu.ko`) via a CMake-wrapped Kbuild system.

## User Prompts

**Before proceeding, ask the user:**

1. **Remote target machine?** (required, `user@hostname` format, e.g. `ben@192.168.0.66`)
2. **Remote staging directory?** (default: `~/tmp/perf-dkms-build`)
3. **Kernel version override?** (leave blank for auto-detect via `uname -r` on remote)
4. **Build type?** (Debug or Release, default: Debug)

## Step 1: Stage and Push

Ensure current perf-dkms changes are on a branch that the remote can access:

```bash
# Record current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"

# Check for uncommitted changes in experimental/perf-dkms/
git status experimental/perf-dkms/

# If there are uncommitted changes, warn the user and ask to commit or stash
# Push to origin so the remote can checkout
git push origin $BRANCH
```

## Step 2: Remote Setup

SSH to the target and prepare the staging directory:

```bash
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
    git clone <REPO_URL> rocm-systems
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
  which perf && perf --version || echo 'MISSING: perf (apt install linux-tools-common linux-tools-\$(uname -r))'
"
```

If prerequisites are missing, report what's needed and stop. Do not proceed with the build.

## Step 4: Build (on remote)

```bash
ssh $TARGET "
  cd $STAGING_DIR/rocm-systems/experimental/perf-dkms

  # Clean previous build
  rm -rf build

  # Configure CMake
  cmake -B build \
    -DBUILD_KERNEL_MODULE=ON \
    -DBUILD_USERSPACE_TOOLS=ON \
    -DBUILD_TESTS=ON \
    -DENABLE_DEBUG=ON  # Use OFF for Release builds
    # Add if kernel version override provided:
    # -DKERNEL_VERSION=\$KERNEL_VERSION_OVERRIDE

  # Build
  cmake --build build --parallel \$(nproc)

  # Check what was produced
  echo ''
  echo '=== Build Output ==='
  if [ -f build/src/amdgpu_pmu.ko ]; then
    echo 'Module: build/src/amdgpu_pmu.ko (standard kernel)'
    ls -lh build/src/amdgpu_pmu.ko
  elif [ -f build/src/amdgpu_pmu.o ]; then
    echo 'Module: build/src/amdgpu_pmu.o (development kernel)'
    echo 'NOTE: Development kernels produce .o instead of .ko'
    ls -lh build/src/amdgpu_pmu.o
  else
    echo 'ERROR: No module output found!'
    ls -la build/src/ | grep amdgpu_pmu
  fi
"
```

**Development kernel note:** Kernels like `6.17.0-rc*` may produce `.o` files instead of `.ko`. Both work with `insmod`. The skill detects which was produced and adjusts accordingly.

## Step 5: Run Userspace Unit Tests (on remote)

These tests validate packet generation and counter registry logic without needing the module loaded:

```bash
ssh $TARGET "
  cd $STAGING_DIR/rocm-systems/experimental/perf-dkms

  echo '=== Userspace Unit Tests ==='
  PASS=0; FAIL=0

  for test in \
    build/src/aql_c/tests/test_pm4_packets \
    build/src/aql_c/tests/test_counter_registry \
    build/src/aql_c/tests/test_packet_generation \
    build/src/aql_c/tests/test_gfx12 \
    build/src/aql_c/tests/test_dimension_decode \
    build/src/aql_c/tests/test_dimension_helpers \
    build/src/aql_c/tests/test_aql_queue; do
    if [ -x \"\$test\" ]; then
      echo \"Running: \$(basename \$test)\"
      if \$test; then
        echo \"  PASS\"
        PASS=\$((PASS + 1))
      else
        echo \"  FAIL\"
        FAIL=\$((FAIL + 1))
      fi
    else
      echo \"SKIP: \$test (not found or not executable)\"
    fi
  done

  echo ''
  echo \"Unit test results: \$PASS passed, \$FAIL failed\"
"
```

## Step 6: Install Module (on remote)

```bash
ssh $TARGET "
  cd $STAGING_DIR/rocm-systems/experimental/perf-dkms

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
  cd $STAGING_DIR/rocm-systems/experimental/perf-dkms

  echo '=== Perf Integration ==='
  perf list 2>/dev/null | grep amdgpu_pmu || echo 'No amdgpu_pmu events in perf list'

  echo ''
  echo '=== Basic Counter Test ==='
  sudo perf stat -e amdgpu_pmu/sq_waves/ sleep 1 2>&1 || echo 'perf stat failed'

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
| `.o` instead of `.ko` | Development kernel -- use `insmod amdgpu_pmu.o` instead (handled automatically) |
| `insmod: ERROR: could not insert module` | Check `dmesg`; may need kernel patch or conflicting module loaded |
| `No events found in perf list` | Module loaded but PMU registration failed; check `dmesg` for errors |
| `Permission denied on perf stat` | Use `sudo` or set `kernel.perf_event_paranoid=0` |
| SSH connection refused | Verify target is reachable: `ssh $TARGET echo ok` |
| Kernel headers version mismatch | Install correct headers: `apt install linux-headers-$KERNEL_VERSION` |
| `cmake: command not found` on remote | Install cmake: `apt install cmake` |
| Build fails with missing headers | Source may be stale; `git fetch && git checkout` again |
| Module loads but no GPU detected | Verify GPU is present: `lspci \| grep -i amd` and KFD is working: `ls /dev/kfd` |
| `vermagic` mismatch on insmod | Kernel version used for build doesn't match running kernel |

Report all results to the user.
```

- [ ] **Step 3: Verify skill file was created correctly**

```bash
cat /home/bewelton/skills/perf-dkms-build/SKILL.md | head -5
# Expected: YAML frontmatter starting with ---
```

- [ ] **Step 4: Commit the skill**

```bash
cd /home/bewelton/skills
git add perf-dkms-build/SKILL.md
git commit -m "feat: add perf-dkms-build skill for remote kernel module build/test"
```

Note: If the skills directory is not a git repo, skip this step. The skill is ready to use.

---

### Task 2: Test the Skill - Remote Setup

Test the skill workflow manually against `ben@192.168.0.66` to validate each step and discover issues.

**Prerequisites:**
- Task 1 complete (skill file exists)
- SSH access to `ben@192.168.0.66` works
- Current branch has perf-dkms changes

- [ ] **Step 1: Verify SSH connectivity**

```bash
ssh ben@192.168.0.66 "echo 'SSH OK'; uname -r; hostname"
```

Expected: Connection succeeds, prints kernel version and hostname.

- [ ] **Step 2: Check git remote URL**

```bash
cd /home/bewelton/rocm-systems
git remote get-url origin
```

Record the URL -- it will be needed for the remote clone step.

- [ ] **Step 3: Ensure current branch is pushed**

```bash
git push origin $(git rev-parse --abbrev-ref HEAD)
```

- [ ] **Step 4: Create staging directory on remote**

```bash
ssh ben@192.168.0.66 "mkdir -p ~/tmp/explore-build"
```

- [ ] **Step 5: Clone or checkout the repo on remote**

```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
REMOTE_URL=$(git remote get-url origin)

ssh ben@192.168.0.66 "
  cd ~/tmp/explore-build
  if [ -d rocm-systems/.git ]; then
    cd rocm-systems
    git fetch origin
    git checkout $BRANCH
    git pull origin $BRANCH
  else
    git clone $REMOTE_URL rocm-systems
    cd rocm-systems
    git checkout $BRANCH
  fi
"
```

Record any issues (auth, branch not found, disk space, etc.).

- [ ] **Step 6: Verify perf-dkms directory exists on remote**

```bash
ssh ben@192.168.0.66 "ls ~/tmp/explore-build/rocm-systems/experimental/perf-dkms/"
```

Expected: Should list `CMakeLists.txt`, `src/`, `test/`, `README.md`, etc.

---

### Task 3: Test the Skill - Prerequisites and Build

- [ ] **Step 1: Run prerequisite check on remote**

```bash
ssh ben@192.168.0.66 "
  echo '=== Build Tools ==='
  which cmake && cmake --version || echo 'MISSING: cmake'
  which make || echo 'MISSING: make'
  which gcc && gcc --version | head -1 || echo 'MISSING: gcc'

  echo ''
  echo '=== Kernel Headers ==='
  KVER=\$(uname -r)
  echo \"Kernel version: \$KVER\"
  ls /lib/modules/\$KVER/build/Makefile 2>/dev/null && echo 'Headers: OK' || echo 'MISSING headers'

  echo ''
  echo '=== KFD Exports ==='
  grep -c 'kfd_' /lib/modules/\$KVER/build/Module.symvers 2>/dev/null || echo '0 KFD exports'

  echo ''
  echo '=== perf ==='
  which perf && perf --version || echo 'MISSING: perf'
"
```

Record any missing prerequisites. If any are missing, install them before continuing.

- [ ] **Step 2: Build the module on remote**

```bash
ssh ben@192.168.0.66 "
  cd ~/tmp/explore-build/rocm-systems/experimental/perf-dkms
  rm -rf build
  cmake -B build \
    -DBUILD_KERNEL_MODULE=ON \
    -DBUILD_USERSPACE_TOOLS=ON \
    -DBUILD_TESTS=ON \
    -DENABLE_DEBUG=ON
  cmake --build build --parallel \$(nproc) 2>&1
"
```

Record:
- Did it produce `.ko` or `.o`?
- Any build warnings or errors?
- Build time?

- [ ] **Step 3: Check build output**

```bash
ssh ben@192.168.0.66 "
  cd ~/tmp/explore-build/rocm-systems/experimental/perf-dkms
  ls -lh build/src/amdgpu_pmu.* 2>/dev/null
  ls build/src/aql_c/tests/ 2>/dev/null
  ls build/src/aql_c/tools/ 2>/dev/null
"
```

Record which files were produced.

---

### Task 4: Test the Skill - Unit Tests and Module Load

- [ ] **Step 1: Run userspace unit tests on remote**

```bash
ssh ben@192.168.0.66 "
  cd ~/tmp/explore-build/rocm-systems/experimental/perf-dkms
  PASS=0; FAIL=0; SKIP=0

  for test in \
    build/src/aql_c/tests/test_pm4_packets \
    build/src/aql_c/tests/test_counter_registry \
    build/src/aql_c/tests/test_packet_generation \
    build/src/aql_c/tests/test_gfx12 \
    build/src/aql_c/tests/test_dimension_decode \
    build/src/aql_c/tests/test_dimension_helpers \
    build/src/aql_c/tests/test_aql_queue; do
    if [ -x \"\$test\" ]; then
      echo \"--- \$(basename \$test) ---\"
      if \$test 2>&1; then
        PASS=\$((PASS + 1))
      else
        FAIL=\$((FAIL + 1))
      fi
    else
      echo \"SKIP: \$(basename \$test)\"
      SKIP=\$((SKIP + 1))
    fi
  done
  echo \"\"
  echo \"Results: \$PASS pass, \$FAIL fail, \$SKIP skip\"
"
```

Record pass/fail/skip counts and any failure output.

- [ ] **Step 2: Load the module on remote**

```bash
ssh ben@192.168.0.66 "
  cd ~/tmp/explore-build/rocm-systems/experimental/perf-dkms

  # Determine module file
  MODULE=\$(ls build/src/amdgpu_pmu.ko build/src/amdgpu_pmu.o 2>/dev/null | head -1)
  echo \"Using module: \$MODULE\"

  sudo rmmod amdgpu_pmu 2>/dev/null || true
  sleep 1
  sudo insmod \$MODULE debug_enable=1 timer_period_ms=100
  sleep 2

  echo '=== lsmod ==='
  lsmod | grep amdgpu_pmu

  echo '=== dmesg ==='
  dmesg | tail -30

  echo '=== sysfs ==='
  ls /sys/bus/event_source/devices/amdgpu_pmu/events/ 2>/dev/null || echo 'No sysfs events'
"
```

Record: module load success, dmesg output, sysfs events listed.

- [ ] **Step 3: Run perf integration tests on remote**

```bash
ssh ben@192.168.0.66 "
  echo '=== perf list ==='
  perf list 2>/dev/null | grep amdgpu_pmu

  echo '=== perf stat test ==='
  sudo perf stat -e amdgpu_pmu/sq_waves/ sleep 1 2>&1
"
```

Record: which events are visible, perf stat output.

- [ ] **Step 4: Run existing test scripts on remote**

```bash
ssh ben@192.168.0.66 "
  cd ~/tmp/explore-build/rocm-systems/experimental/perf-dkms

  echo '=== test_module.sh ==='
  sudo bash test_module.sh 2>&1 || echo 'test_module.sh failed'

  echo '=== test/load_test.sh ==='
  cd test
  sudo bash load_test.sh 2>&1 || echo 'load_test.sh failed'

  echo '=== test/perf_test.sh ==='
  sudo bash perf_test.sh 2>&1 || echo 'perf_test.sh failed'
"
```

Record: which tests pass, which fail, and why.

**Known issue:** `load_test.sh` checks for `pmu_stub` in sysfs but the module now registers as `amdgpu_pmu`. `perf_test.sh` uses old event names (`cycles/`, `instructions/`) instead of real counter names (`sq_waves`, `gl2c_hit`). These test scripts may need updating.

- [ ] **Step 5: Cleanup -- unload module**

```bash
ssh ben@192.168.0.66 "sudo rmmod amdgpu_pmu 2>/dev/null || true"
```

---

### Task 5: Iterate on Skill Based on Test Results

After Tasks 2-4, review what went wrong and update the skill.

- [ ] **Step 1: Document all discoveries**

Create a file `~/ai/task_info/perf-dkms-build-skill-testing.md` with:
- Kernel version on remote machine
- `.ko` vs `.o` behavior observed
- Which prerequisite checks passed/failed
- Build warnings or issues
- Unit test results
- Module load behavior
- Perf integration results
- Test script issues (outdated sysfs paths, event names)
- Any SSH or git issues

- [ ] **Step 2: Update SKILL.md based on discoveries**

Edit `/home/bewelton/skills/perf-dkms-build/SKILL.md` to fix any issues found during testing:
- Adjust build commands if they failed
- Add missing prerequisite checks
- Fix paths or commands that didn't work
- Add new troubleshooting entries
- Correct any assumptions about `.ko` vs `.o`

- [ ] **Step 3: Re-test the updated skill**

Re-run the build and test steps from Tasks 3-4 to verify the fixes work.

- [ ] **Step 4: Commit updated skill**

```bash
cd /home/bewelton/skills
git add perf-dkms-build/SKILL.md
git commit -m "fix: update perf-dkms-build skill based on testing feedback"
```

---

### Task 6: Update README.md on origin/develop

Update `experimental/perf-dkms/README.md` based on discoveries from testing.

**Files:**
- Modify: `experimental/perf-dkms/README.md` (on a branch based on origin/develop)

- [ ] **Step 1: Switch to develop or create a branch from develop**

```bash
cd /home/bewelton/rocm-systems
git fetch origin develop
git checkout -b bewelton/perf-dkms-docs origin/develop
```

- [ ] **Step 2: Read current README.md**

Read `experimental/perf-dkms/README.md` to understand current state.

- [ ] **Step 3: Update README.md with discovered nuances**

Based on testing results, add or update these sections:
- **Prerequisites**: Exact packages needed, kernel header requirements
- **Building**: Note `.o` vs `.ko` behavior on development kernels, kernel version override syntax
- **Remote Build**: Document the remote build workflow (push branch, SSH checkout, build, test)
- **Troubleshooting**: Add any new issues discovered during testing
- **Kernel Patch**: Clarify what the kernel patch provides and how to check if it's applied

Use the Edit tool to make minimal, targeted changes. Do not rewrite sections that are already accurate.

- [ ] **Step 4: Commit README changes**

```bash
git add experimental/perf-dkms/README.md
git commit -m "[perf-dkms] Update README with build nuances from testing"
```

---

### Task 7: Update QUICKSTART.md on origin/develop

**Files:**
- Modify: `experimental/perf-dkms/QUICKSTART.md` (on the same branch as Task 6)

- [ ] **Step 1: Read current QUICKSTART.md**

Read `experimental/perf-dkms/QUICKSTART.md` to understand current state.

- [ ] **Step 2: Update QUICKSTART.md if needed**

Based on testing results, ensure the quickstart guide:
- Has accurate build commands that actually work
- Mentions `.o` vs `.ko` for dev kernels
- Lists correct event names for perf stat examples
- Has accurate prerequisite installation commands

If the QUICKSTART.md is already accurate, skip this task.

- [ ] **Step 3: Commit if changes were made**

```bash
git add experimental/perf-dkms/QUICKSTART.md
git commit -m "[perf-dkms] Update QUICKSTART with tested build steps"
```

---

### Task 8: Fix Outdated Test Scripts (if applicable)

During Task 4, `load_test.sh` and `perf_test.sh` were identified as using outdated names (`pmu_stub` sysfs path, `cycles/instructions/` event names). If testing confirms these are broken, fix them.

**Files:**
- Modify: `experimental/perf-dkms/test/load_test.sh` (on the same branch)
- Modify: `experimental/perf-dkms/test/perf_test.sh` (on the same branch)

- [ ] **Step 1: Fix load_test.sh sysfs path**

Change `pmu_stub` references to `amdgpu_pmu` in `test/load_test.sh`:
- The sysfs check uses `/sys/bus/event_source/devices/pmu_stub` but should use `/sys/bus/event_source/devices/amdgpu_pmu`

- [ ] **Step 2: Fix perf_test.sh event names**

Update `test/perf_test.sh` to use real counter names:
- Replace `${PMU_NAME}/cycles/` with `${PMU_NAME}/sq_waves/`
- Replace `${PMU_NAME}/instructions/` with `${PMU_NAME}/sq_instructions/`
- Replace `${PMU_NAME}/cache-misses/` with `${PMU_NAME}/gl2c_miss/`
- Replace `${PMU_NAME}/bandwidth/` with `${PMU_NAME}/gl2c_hit/`
- Update multi-event test to use real counter names

- [ ] **Step 3: Test the fixed scripts on remote**

```bash
ssh ben@192.168.0.66 "
  cd ~/tmp/explore-build/rocm-systems/experimental/perf-dkms
  # Reload module first
  sudo rmmod amdgpu_pmu 2>/dev/null || true
  MODULE=\$(ls build/src/amdgpu_pmu.ko build/src/amdgpu_pmu.o 2>/dev/null | head -1)
  sudo insmod \$MODULE debug_enable=1 timer_period_ms=100
  sleep 2

  cd test
  sudo bash load_test.sh 2>&1
  sudo bash perf_test.sh 2>&1
"
```

- [ ] **Step 4: Commit test script fixes**

```bash
git add experimental/perf-dkms/test/load_test.sh experimental/perf-dkms/test/perf_test.sh
git commit -m "[perf-dkms] Fix outdated sysfs paths and event names in test scripts"
```

---

### Task 9: Final Verification and Cleanup

- [ ] **Step 1: Run full end-to-end test using the skill**

Invoke the perf-dkms-build skill with:
- Target: `ben@192.168.0.66`
- Staging: `~/tmp/explore-build`
- Kernel: auto-detect
- Build: Debug

Verify all steps complete successfully.

- [ ] **Step 2: Clean up remote staging**

```bash
ssh ben@192.168.0.66 "sudo rmmod amdgpu_pmu 2>/dev/null || true"
```

Do NOT delete the staging directory -- the user may want to keep it for future work.

- [ ] **Step 3: Update task file with final status**

Update `~/ai/task_info/perf-dkms-build-skill-testing.md` with:
- Mark task as COMPLETED
- Final test results
- Any remaining known issues
- Summary of all changes made

- [ ] **Step 4: Push documentation branch**

```bash
git push origin bewelton/perf-dkms-docs
```

Report the branch name to the user so they can create a PR.
