# AI NIC Simulation

This document explains how to build, run, and test the AMD AI NIC (Pensando Pollara)
simulation on a machine **without physical AI NIC hardware**, including inside a
container.

---

## Table of Contents

1. [Overview](#overview)
2. [How it works](#how-it-works)
3. [Repository layout](#repository-layout)
4. [Build instructions](#build-instructions)
5. [Manual test (quick sanity check)](#manual-test-quick-sanity-check)
6. [Automated pytest suite](#automated-pytest-suite)
7. [Full rocprofiler-systems end-to-end test](#full-rocprofiler-systems-end-to-end-test)
8. [Container quick-start recipe](#container-quick-start-recipe)

---

## Overview

AMD AI NIC cards (Pensando Pollara) expose their metrics through the Linux
**sysfs** pseudo-filesystem at paths like:

```
/sys/bus/pci/devices/<bdf>/
/sys/class/net/<iface>/
/sys/class/infiniband/<ib_dev>/ports/<n>/hw_counters/
```

The `amdsmi` library reads these paths at device-discovery time and on every
metrics collection call.  Because the paths are **hardcoded** in the original
code, running on a machine without the hardware (e.g. a CI container) makes
any test of the collection pipeline impossible.

The simulation adds:

* A **configurable sysfs root** (`SMI_NIC_SYSFS_ROOT` env var) — every sysfs
  path in amdsmi is prefixed with this value instead of the real `/`.
* A **fake sysfs tree** (`fake_sysfs.py`) — a Python script that builds a
  minimal but complete tree that satisfies all amdsmi traversals.
* A **NIC simulator** (`nic_simulator.py`) — a background process that
  periodically increments the `hw_counters` files so the sampled values are
  non-zero and rising, just like a real card under load.
* A **manual info tool** (`amd_smi_ainic_info`) — a C++ binary that calls
  every public amdsmi NIC API once and prints the result; useful for a quick
  human-readable sanity check.
* A **developer script** (`run_simulated_ainic.sh`) — sets everything up and
  runs `amd_smi_ainic_info` in one shot.
* **pytest tests** — three automated tests that cover sysfs structure,
  simulator behaviour, and the full amdsmi read path.

---

## How it works

### `SMI_NIC_SYSFS_ROOT`

Every C++ file that previously used a hardcoded `/sys/...` path now calls
`smi_sysfs_root()` (declared in `smi_sysfs.h`, implemented in `smi_sysfs.cpp`).
That helper reads `SMI_NIC_SYSFS_ROOT` from the environment:

* If unset (or empty) → returns `""` → paths are `/sys/...` (real hardware).
* If set to `/tmp/ainic-sim-sysfs.XYZ` → paths become
  `/tmp/ainic-sim-sysfs.XYZ/sys/...` (fake tree).

Modified files:

| File | Change |
|------|--------|
| `src/nic/ai-nic/amdsmi_unified/inc/smi_sysfs.h` | Declares `smi_sysfs_root()` |
| `src/nic/ai-nic/amdsmi_unified/src/smi_sysfs.cpp` | Implements `smi_sysfs_root()` |
| `src/nic/ai-nic/amdsmi_unified/src/smi_nic_system.cpp` | Uses root for `net_path_` and `pci_path_` |
| `src/nic/ai-nic/amdsmi_unified/src/smi_nic.cpp` | Uses root for NUMA node path |
| `src/nic/ai-nic/amdsmi_unified/src/smi_nic_subsystem.cpp` | Uses root for driver paths |
| `src/amd_smi/amd_smi.cc` | Uses root for `hw_counters` lookup via class/net |
| `src/nic/ai-nic/amdsmi_unified/src/smi_nic_interface.cpp` | `smi_get_nic_driver_info`: treat ethtool failure as non-fatal (returns "N/A" instead of error) |

### Fake sysfs tree

`fake_sysfs.py::create(root)` builds this layout under `root/`:

```
sys/
├── bus/
│   ├── pci/
│   │   ├── devices/
│   │   │   ├── 0000:e2:00.0 -> ../../../devices/pci0000:e0/0000:e2:00.0
│   │   │   └── 0000:e2:00.1 -> ../../../devices/pci0000:e0/0000:e2:00.0/0000:e2:00.1
│   │   └── drivers/ionic/
│   │       └── 0000:e2:00.1 -> (symlink, for driver_loaded() check)
│   └── auxiliary/drivers/ionic_rdma.rdma/
│       └── ionic_rdma.rdma.0 -> (symlink, for RDMA driver_loaded() check)
├── class/
│   ├── net/enp226s0/
│   │   ├── address, carrier, mtu, operstate, speed, ifindex, ...
│   │   ├── statistics/{rx_bytes, tx_bytes}
│   │   └── device -> ../../../devices/pci0000:e0/0000:e2:00.0/0000:e2:00.1
│   └── infiniband/rocep226s0 -> (symlink to real IB device dir)
├── devices/
│   ├── pci0000:e0/
│   │   └── 0000:e2:00.0/          ← Pensando PCIe bridge (vendor=0x1dd8 dev=0x0008)
│   │       ├── vendor, device, subsystem_vendor, subsystem_device, revision, ...
│   │       └── 0000:e2:00.1/      ← Pensando port (vendor=0x1dd8 dev=0x1002)
│   │           ├── vendor, device, numa_node, ...
│   │           ├── ionic_rdma.rdma.0/  (auxiliary device for driver check)
│   │           └── infiniband/rocep226s0/
│   │               ├── node_guid, node_type, sys_image_guid, fw_ver
│   │               └── ports/1/
│   │                   ├── state, max_mtu, active_mtu
│   │                   └── hw_counters/
│   │                       ├── rx_rdma_ucast_bytes   ← simulator writes here
│   │                       ├── tx_rdma_ucast_bytes
│   │                       └── ... (10 counters total)
│   └── system/node/node0/cpulist
```

### NIC simulator

`nic_simulator.py` runs in the background, reading each counter file and
writing `value + delta` every `--interval` seconds (default 0.1 s).
It exits cleanly on `SIGTERM` / `SIGINT`.

```
python3 nic_simulator.py <hw_counters_dir> [--interval 0.1]
```

---

## Repository layout

```
projects/amdsmi/
├── example/
│   ├── amd_smi_ainic_example.cc   ← manual info tool source
│   ├── CMakeLists.txt             ← builds amd_smi_ainic_info target
│   └── run_simulated_ainic.sh     ← one-shot developer test script
└── src/
    ├── amd_smi/amd_smi.cc
    └── nic/ai-nic/amdsmi_unified/
        ├── inc/smi_sysfs.h
        └── src/
            ├── smi_sysfs.cpp
            ├── smi_nic_system.cpp
            ├── smi_nic.cpp
            ├── smi_nic_subsystem.cpp
            └── smi_nic_interface.cpp

projects/rocprofiler-systems/tests/pytest/
├── conftest.py           ← registers ainic_sim pytest mark
├── fake_sysfs.py         ← builds the fake sysfs tree
├── nic_simulator.py      ← background counter updater
└── test_ainic_sim.py     ← pytest test suite (4 tests)
```

---

## Build instructions

### Prerequisites (Ubuntu 22.04 container)

```bash
apt-get install -y \
    build-essential cmake git \
    libnl-3-dev libnl-genl-3-dev libmnl-dev \
    python3 python3-pip
pip3 install pytest
```

### Build amdsmi only (fastest — sufficient for manual testing)

```bash
cd /path/to/rocm-systems

cmake -S projects/amdsmi \
      -B projects/amdsmi/build \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DROCM_DIR=/opt/rocm

cmake --build projects/amdsmi/build \
      --target amd_smi_ainic_info \
      -j$(nproc)
```

The binary will be at:

```
projects/amdsmi/build/example/amd_smi_ainic_info
```

### Build rocprofiler-systems (required for full end-to-end test)

```bash
cmake -S projects/rocprofiler-systems \
      -B projects/rocprofiler-systems/build \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -Damdsmi_DIR=projects/amdsmi/build \
      -DROCM_DIR=/opt/rocm

cmake --build projects/rocprofiler-systems/build \
      --target rocprof-sys-sample \
      -j$(nproc)
```

---

## Manual test (quick sanity check)

The `run_simulated_ainic.sh` script does everything in one command:

```bash
cd projects/amdsmi/example

./run_simulated_ainic.sh /path/to/amd_smi_ainic_info
# or, if the binary is in PATH or a standard build dir:
./run_simulated_ainic.sh
```

It will:
1. Create a temp fake sysfs tree.
2. Start the NIC simulator in the background.
3. Run `amd_smi_ainic_info` with `SMI_NIC_SYSFS_ROOT` set.
4. Print all discovered AI NIC parameters (ASIC, bus, driver, NUMA, ports,
   RDMA devices, and live hw_counters).
5. Kill the simulator and clean up on exit.

Expected output excerpt:

```
SMI_NIC_SYSFS_ROOT = /tmp/ainic-sim-sysfs.XXXXX  (simulation mode)
amd-smi initialized

=== AI NIC device 0 (socket 0, device 0) ===

  --- ASIC ---
  Vendor name                  : N/A
  Product name                 : N/A
  ...

  --- RDMA Devices ---
  1 RDMA device(s)
    [rdma_dev 0: rocep226s0]
      node GUID                  : 0xaabbccddeeff0011
      ...
        --- hw_counters (10) ---
        rx_rdma_ucast_bytes              : 42
        tx_rdma_ucast_bytes              : 42
        ...

Total AI NIC devices found: 1
amd-smi shut down
```

---

## Automated pytest suite

```bash
cd projects/rocprofiler-systems/tests/pytest

# Run all four tests (3 pass/skip, 1 requires build):
python3 -m pytest test_ainic_sim.py -v

# Run only the two infrastructure tests (no binary needed):
python3 -m pytest test_ainic_sim.py -v -k "sysfs or simulator"

# Run the amdsmi direct test (needs amd_smi_ainic_info on PATH):
export PATH=$PATH:/path/to/projects/amdsmi/build/example
python3 -m pytest test_ainic_sim.py::TestAINICSim::test_amdsmi_reads_simulated_nic_counters -v
```

### What each test does

| Test | Needs | Validates |
|------|-------|-----------|
| `test_fake_sysfs_structure` | nothing | Symlinks, file contents, and directory layout of the fake sysfs tree |
| `test_nic_simulator_increments_counters` | nothing | Simulator actually increments `hw_counter` files over time |
| `test_amdsmi_reads_simulated_nic_counters` | `amd_smi_ainic_info` binary | Our patched amdsmi discovers the device, reads RDMA counters, and returns non-zero values while the simulator runs |
| `test_rocprof_sys_sample_collects_nic_metrics` | dev build + `AINIC_TEST_ROCPROF=1` | Full rocprofiler-systems pipeline collects NIC metrics from the simulation |

### Enabling the rocprof-sys-sample test

This test requires a `rocprof-sys-sample` binary built against the patched
amdsmi (the installed system version does not support `SMI_NIC_SYSFS_ROOT`):

```bash
export AINIC_TEST_ROCPROF=1
export PATH=/path/to/rocprofiler-systems/build/bin:$PATH
python3 -m pytest test_ainic_sim.py -v
```

---

## Full rocprofiler-systems end-to-end test

Once both projects are built (see [Build instructions](#build-instructions)):

```bash
export SMI_NIC_SYSFS_ROOT=$(mktemp -d)
python3 -c "
import sys
sys.path.insert(0, 'projects/rocprofiler-systems/tests/pytest')
import fake_sysfs
from pathlib import Path
fake_sysfs.create(Path('$SMI_NIC_SYSFS_ROOT'))
"

HW=$SMI_NIC_SYSFS_ROOT/sys/devices/pci0000:e0/0000:e2:00.0/0000:e2:00.1/infiniband/rocep226s0/ports/1/hw_counters
python3 projects/rocprofiler-systems/tests/pytest/nic_simulator.py "$HW" &
SIM=$!

SMI_NIC_SYSFS_ROOT=$SMI_NIC_SYSFS_ROOT \
  AINIC_TEST_ROCPROF=1 \
  PATH=projects/rocprofiler-systems/build/bin:$PATH \
  python3 -m pytest projects/rocprofiler-systems/tests/pytest/test_ainic_sim.py -v

kill $SIM
rm -rf "$SMI_NIC_SYSFS_ROOT"
```

---

## Container quick-start recipe

This is the complete procedure to go from a clean Ubuntu 22.04 container to a
passing AI NIC simulation test, assuming the rocm-systems repository is mounted
at `/mnt`.

```bash
# 1. Install build dependencies
apt-get update && apt-get install -y \
    build-essential cmake git ninja-build \
    libnl-3-dev libnl-genl-3-dev libmnl-dev \
    python3 python3-pip
pip3 install pytest

# 2. Build amdsmi (only the info binary, fast build)
cmake -S /mnt/projects/amdsmi \
      -B /tmp/amdsmi-build \
      -GNinja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DROCM_DIR=/opt/rocm

ninja -C /tmp/amdsmi-build amd_smi_ainic_info

# 3. Quick manual sanity check
cd /mnt/projects/amdsmi/example
./run_simulated_ainic.sh /tmp/amdsmi-build/example/amd_smi_ainic_info

# 4. Run the automated tests
export PATH=$PATH:/tmp/amdsmi-build/example
cd /mnt/projects/rocprofiler-systems/tests/pytest
python3 -m pytest test_ainic_sim.py -v
```

Expected result: **3 PASSED, 1 SKIPPED** (the rocprof-sys-sample test is
skipped until `AINIC_TEST_ROCPROF=1` and a dev build are available).

---

## Running `test_ainic_perf.py` without hardware

`test_ainic_perf.py` tests full end-to-end AI NIC performance sampling via
`rocprof-sys-sample`.  It was originally written for systems with real Pensando
Pollara hardware, but it now supports simulation via `SMI_NIC_SYSFS_ROOT`.

### Prerequisites

* A dev build of **rocprofiler-systems** that links against the patched amdsmi
  (the one with `SMI_NIC_SYSFS_ROOT` support and the non-fatal ethtool path).
* `wget` must be on the `PATH` (the test uses it as the profiled workload).
* The `setup_ainic_sim.sh` script (in the same directory as the tests).

### Workflow: two terminals

**Terminal 1 — start the simulation** (keep it running for the duration of the tests):

```bash
cd projects/rocprofiler-systems/tests/pytest
./setup_ainic_sim.sh
```

The script prints a line like:

```
    export SMI_NIC_SYSFS_ROOT=/tmp/ainic-sim-12345
```

**Terminal 2 — run the tests:**

```bash
# Copy the export line from Terminal 1:
export SMI_NIC_SYSFS_ROOT=/tmp/ainic-sim-12345
export ROCPROFSYS_BUILD_DIR=/mnt/projects/rocprofiler-systems/build/debug

cd projects/rocprofiler-systems/tests/pytest
pytest -m ainic test_ainic_perf.py -v
```

The `ainic_perf_env` fixture automatically forwards `SMI_NIC_SYSFS_ROOT` into
the `rocprof-sys-sample` child process so that amdsmi uses the fake sysfs.
The `ainic_available` session fixture skips the test automatically if neither
real hardware nor `SMI_NIC_SYSFS_ROOT` is set.

### What the test validates

| Check | How |
|-------|-----|
| 10 AI NIC Perfetto counter tracks written | `assert_perfetto()` with `AINIC_PERFETTO_COUNTER_NAMES` |
| 10 AI NIC track names in ROCpd SQLite DB | queries `rocpd_string_*` tables for `ainic_*` strings |

### Simulation vs real hardware

| | Simulation | Real hardware |
|---|---|---|
| `SMI_NIC_SYSFS_ROOT` | set to fake sysfs root | unset |
| Counter source | `nic_simulator.py` (increments files every 50 ms) | Pensando Pollara NIC |
| Workload | `wget` download (2 files) | same |
| Expected result | PASSED (counters non-zero) | PASSED (counters non-zero) |

> **Tip:** The download URLs in the test point to real release artifacts. In an
> air-gapped environment, replace them with any HTTP endpoint that takes several
> seconds to respond (e.g. a local Python HTTP server serving a large file).

> **Note:** `ROCPROFSYS_USE_PROCESS_SAMPLING` must be `ON` (which is the default
> in the fixture) because `ROCPROFSYS_USE_AINIC` is in the `process_sampling`
> category.  Setting process sampling to `OFF` silently disables AINIC even
> when `ROCPROFSYS_USE_AINIC=ON`.
