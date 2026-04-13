# perf-dkms-build Skill Design

**Date:** 2026-04-13
**Status:** Draft
**Author:** Ben Welton + Claude

## Summary

A single, self-contained Claude Code skill (`perf-dkms-build/SKILL.md`) that handles the full lifecycle of building, installing, and testing the perf-dkms kernel module on a remote target machine. The skill is generic with no hardcoded hosts or paths -- all values are user-specified at invocation.

## Context

perf-dkms is a Linux kernel DKMS module that exposes AMD GPU hardware performance counters through the Linux perf subsystem. It lives at `experimental/perf-dkms/` in the rocm-systems repo and produces a kernel module (`amdgpu_pmu.ko`) via a CMake-wrapped Kbuild system.

Building kernel modules has kernel-version-specific nuances (header matching, `.o` vs `.ko` output on dev kernels, cross-compilation). The skill must capture these details so that any developer can build and test without tribal knowledge.

## Design Decisions

1. **Single monolithic skill** -- Matches existing skill patterns (hsa-build, hip-build). One `SKILL.md` file covers push, SSH checkout, build, install, test.
2. **Generic, no hardcoded hosts** -- All target machines and paths are user-specified. Skill can be committed to the repo.
3. **Manual insmod only** -- No DKMS registration workflow. Simpler for development iteration.
4. **Both kernel modes** -- Default to `uname -r` auto-detect on remote, but allow explicit `KERNEL_VERSION` override for cross-compilation.
5. **Kernel patch check-and-warn** -- Verify kernel exports exist, warn if missing, but don't block the build.

## Skill Structure

### Metadata

```yaml
---
name: perf-dkms-build
description: Build, install, and test perf-dkms kernel module on remote target
trigger:
  paths: ["experimental/perf-dkms/**"]
  keywords: ["build perf-dkms", "build perf dkms", "perf-dkms", "amdgpu_pmu"]
---
```

### User Prompts (Before Build)

The skill asks the user for:

1. **Remote target machine** -- `user@hostname` (required, no default)
2. **Remote staging directory** -- where to clone/checkout (default: `~/tmp/perf-dkms-build`)
3. **Kernel version override** -- leave blank for `uname -r` auto-detect on remote
4. **Build type** -- Debug (default, includes `-DDEBUG -g`) or Release

### Workflow Steps

#### Step 1: Stage & Push

- Ensure current changes are committed or stashed
- Push current branch to origin
- Record the branch name and commit hash

#### Step 2: Remote Setup

- SSH to target machine
- Create staging directory if it doesn't exist
- Clone the repo into staging dir (first time) or `git fetch origin && git checkout <branch>` (subsequent runs)

#### Step 3: Prerequisite Check (on remote)

```bash
# Build tools
which cmake && cmake --version
which make
which gcc

# Kernel headers
KERNEL_VERSION=${OVERRIDE:-$(uname -r)}
ls /lib/modules/$KERNEL_VERSION/build/Makefile

# Kernel patch exports (check and warn)
grep -c 'kfd_' /lib/modules/$KERNEL_VERSION/build/Module.symvers 2>/dev/null
```

If prerequisites are missing, report what's needed and stop.

#### Step 4: Build (on remote)

```bash
cd <staging_dir>/experimental/perf-dkms

cmake -B build \
  -DBUILD_KERNEL_MODULE=ON \
  -DBUILD_USERSPACE_TOOLS=ON \
  -DBUILD_TESTS=ON \
  -DENABLE_DEBUG=ON   # or OFF for Release
  # If kernel version override:
  # -DKERNEL_VERSION=$KERNEL_VERSION

cmake --build build --parallel $(nproc)
```

Check for output:
- `build/src/amdgpu_pmu.ko` (standard kernels)
- `build/src/amdgpu_pmu.o` (development kernels -- use this for insmod)

#### Step 5: Install (on remote)

```bash
# Unload existing module
sudo rmmod amdgpu_pmu 2>/dev/null || true

# Load the built module
sudo insmod build/src/amdgpu_pmu.ko  # or .o for dev kernels

# Verify
lsmod | grep amdgpu_pmu
dmesg | tail -20
```

#### Step 6: Test (on remote)

**Phase 1: Userspace unit tests** (no module needed)
```bash
./build/src/aql_c/tests/test_pm4_packets
./build/src/aql_c/tests/test_counter_registry
./build/src/aql_c/tests/test_packet_generation
./build/src/aql_c/tests/test_gfx12
./build/src/aql_c/tests/test_dimension_decode
./build/src/aql_c/tests/test_dimension_helpers
```

**Phase 2: Module load verification**
```bash
ls /sys/bus/event_source/devices/amdgpu_pmu/events/
perf list | grep amdgpu_pmu
```

**Phase 3: Perf integration**
```bash
perf stat -e amdgpu_pmu/sq_waves/ sleep 1
```

**Phase 4: Existing test scripts**
```bash
./test/load_test.sh
./test/perf_test.sh
```

#### Step 7: Report

Display summary:
- Build status (success/failure, .ko vs .o)
- Module load status
- Unit test results (pass/fail counts)
- Perf integration results
- Kernel version used
- Any warnings from prerequisite checks

### Troubleshooting Table

| Issue | Fix |
|-------|-----|
| `No rule to make target 'modules'` | Kernel headers missing: `apt install linux-headers-$(uname -r)` |
| `.o` instead of `.ko` | Development kernel -- use `insmod amdgpu_pmu.o` instead |
| `insmod: ERROR: could not insert module` | Check `dmesg`; may need kernel patch or conflicting module |
| `No events found in perf list` | Module loaded but PMU registration failed; check `dmesg` |
| `Permission denied on perf stat` | Use `sudo` or set `kernel.perf_event_paranoid=0` |
| SSH connection refused | Verify target is reachable and SSH key is set up |
| Kernel headers version mismatch | Install correct headers: `apt install linux-headers-$KERNEL_VERSION` |
| `cmake: command not found` on remote | Install cmake: `apt install cmake` |
| Build fails with missing `aql_queue.h` | Source may be out of date; `git fetch && git checkout` again |

### README/Install Guide Updates

After testing, update these files based on discoveries:
- `experimental/perf-dkms/README.md` -- Notes about dev kernel `.o` vs `.ko`, kernel version requirements, remote build workflow, prerequisite installation
- `experimental/perf-dkms/QUICKSTART.md` -- Ensure it reflects actual tested build steps

## Implementation Plan (High Level)

1. Create `/home/bewelton/skills/perf-dkms-build/SKILL.md` with the skill content
2. Test the skill by running it against `ben@192.168.0.66`
3. Iterate on the skill based on build/test results
4. Update `experimental/perf-dkms/README.md` with discovered nuances
5. Update `experimental/perf-dkms/QUICKSTART.md` if needed

## Success Criteria

- Skill can be invoked to build perf-dkms on any remote machine with kernel headers
- Module loads successfully and exposes perf events
- Full test suite passes (unit tests + load test + perf test)
- Kernel version override works for cross-compilation scenarios
- README/QUICKSTART reflect actual build experience
