---
name: amdsmi-improvement-evaluation
description: A structured approach to evaluating AMD SMI's current state and identifying improvement opportunities.
---

# AMD SMI Improvement Evaluation

Build a practical understanding of AMD SMI and identify high-value improvement areas for ROCm users who need GPU monitoring, diagnostics, and management. This includes validating whether AMD SMI reports correct GPU information across different AMD GPU models and driver environments.

## Current Source Notes

- `ROCm/amdsmi` still exists, but its README says the repository is deprecated and that source work has moved to `ROCm/rocm-systems/projects/amdsmi`.
- The latest public AMD SMI documentation is published under ROCm Software 7.2.2 / AMD SMI 26.2.2.
- The documentation frames AMD SMI as a unified user-space interface for GPU management and monitoring in HPC environments.
- The documented interfaces include C++, Python, Go, CLI usage, Docker usage, and Reliability, Availability, Serviceability (RAS) concepts.

## Official Example Inventory

From `ROCm/rocm-systems` on the `develop` branch, the current example directory
contains:

- `CMakeLists.txt`
- `amd_smi_afid_example.py`
- `amd_smi_cper.cc`
- `amd_smi_cper_example.py`
- `amd_smi_drm_example.cc`
- `amd_smi_nic.cc`
- `amd_smi_nodrm_example.cc`
- `amd_smi_violation_example.py`
- `amdsmi_esmi_intg_example.cc`

These examples give us an initial test set for assessing build friction, runtime requirements, documentation clarity, and API ergonomics.

## Investigation Tracks

### 1. CLI Experience

Questions:

- How easy is it to discover the right `amd-smi` command for common tasks?
- Are outputs readable for humans and stable enough for scripts?
- Are errors actionable when permissions, drivers, or supported hardware are missing?
- Is JSON or machine-readable output available where operators need it?

Evidence to collect:

- `amd-smi --help`
- Common query commands for devices, utilization, power, temperature, clocks, firmware, ECC/RAS, topology, and processes
- Behavior with missing GPU access or unsupported host environment

### 2. Python API Experience

Questions:

- How much boilerplate is needed to initialize, enumerate devices, query metrics, and handle errors?
- Do Python return shapes feel consistent and typed enough for downstream tools?
- Can a user build a simple exporter or health check without reading C++ API details?

Evidence to collect:

- Minimal device inventory script
- Minimal utilization/power/thermal polling script
- Error-handling behavior and exception patterns
- Differences between Python examples and docs

### 3. C++ API Experience

Questions:

- Are headers, linking, and CMake integration straightforward?
- Are API names and structs discoverable?
- Are lifetime, initialization, and cleanup rules obvious?
- Are example binaries focused enough to teach one concept at a time?

Evidence to collect:

- Build steps for official C++ examples
- CMake dependency behavior with installed ROCm
- Compile/link errors on a clean environment
- Required privileges and driver/library paths

### 4. Observability Coverage

Questions:

- Which operational signals are easy to retrieve?
- Which signals are available only through the CLI, only through APIs, or only through privileged contexts?
- Are important health signals missing, unclear, or difficult to correlate?

Signal categories:

- Device identity and firmware
- Utilization and process attribution
- Memory usage and bandwidth
- Power, energy, clocks, and performance levels
- Temperature and throttling
- ECC, RAS, CPER, XGMI, and violation data
- Topology, NUMA locality, and NIC integration

### 5. Output Correctness

Questions:

- Does AMD SMI report the correct GPU model, ASIC name, device ID, revision, VRAM size, memory type, firmware versions, PCIe information, and serial/UUID identifiers?
- Do reported clock, power, temperature, utilization, memory, and throttling values match trusted system sources within expected tolerances?
- Are incorrect fields isolated to specific GPU families, ROCm versions, kernel drivers, firmware versions, containers, or VM environments?
- Does the CLI disagree with the Python, C++, or Go APIs for the same GPU?
- When AMD SMI cannot determine a value, does it return an explicit unsupported or unavailable state instead of a misleading value?

Baseline comparison sources:

- `amd-smi`
- AMD SMI Python API
- AMD SMI C++ API examples
- `rocminfo`
- `rocm-smi`, where available
- Linux sysfs under `/sys/class/drm/card*/device`
- `lspci -nn`, `lspci -vv`, and PCI IDs
- Kernel logs from `dmesg`
- Vendor specifications for the tested GPU model

Fields to validate:

- GPU marketing name / product name
- ASIC name
- Device ID, vendor ID, subsystem ID, revision ID
- BDF / PCI bus address
- VRAM size and memory vendor/type
- Firmware and VBIOS versions
- Driver and ROCm versions
- Power, energy, clocks, and performance level
- Temperature sensors and throttling state
- Utilization and active process reporting
- ECC/RAS support and counters

Evidence to collect:

- Exact command or API call
- Raw AMD SMI output
- Raw baseline output
- Expected value
- Observed AMD SMI value
- Whether the value is wrong, missing, stale, rounded, mislabeled, or ambiguously named
- GPU model, ROCm version, kernel version, driver version, OS, and environment

### 6. Documentation and Example Quality

Questions:

- Does the documentation guide a first-time user from install to a working script?
- Are examples named and scoped clearly?
- Are expected outputs shown?
- Are environment assumptions explicit?
- Are version differences visible enough?

Evidence to collect:

- Missing prerequisites
- Broken links or stale repository references
- Ambiguous install/build instructions
- Example commands that cannot be copied directly

## Improvement Backlog Template

For each issue or opportunity, capture:

- Title
- Type of User affected
- Scenario
- Current behavior
- Expected or better behavior
- Evidence
- Suggested improvement
- Impact: high / medium / low
- Effort: high / medium / low

## Sources

- AMD SMI source repository: <https://github.com/ROCm/amdsmi>
- AMD SMI active super-repo path: <https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi>
- AMD SMI documentation: <https://rocm.docs.amd.com/projects/amdsmi/en/latest/>
- AMD SMI examples: <https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi/example>
