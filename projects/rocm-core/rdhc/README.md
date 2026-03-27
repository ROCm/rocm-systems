# ROCm Deployment Health Check (RDHC)

## Overview

RDHC is a comprehensive health check tool for ROCm deployments. It validates GPU presence, driver status, kernel parameters, library dependencies, and tests installed ROCm components.

## Features

- **Cross-Platform Support**: Works on Ubuntu, RHEL, and SLES distributions
- **Comprehensive Testing**: GPU validation, driver checks, library dependencies, kernel parameters, and component-specific tests
- **Dynamic Component Detection**: Automatically identifies installed ROCm components
- **Flexible Reporting**: Pretty table output and JSON export options
- **Configurable Verbosity**: Support for verbose, normal, and silent modes
- **Optional check control**: Skip cluster-only or environment-limited tests; relax kernel parameter strictness (see [Optional CLI behavior])

## Test Categories

### Default Tests (Quick Mode)

1. **GPU Presence** - Detects AMD GPUs in the system
2. **AMDGPU Driver** - Validates driver installation and initialization
3. **Kernel Parameters** - Checks ROCm-related kernel settings
4. **rocminfo** - Validates ROCm information utility
5. **rocm_agent_enumerator** - Checks GPU agent enumeration
6. **amd-smi** - Tests AMD System Management Interface
7. **Library Dependencies** - Validates ROCm library dependencies
8. **Environment Variables** - Checks ROCm-related environment settings
9. **Multinode Cluster Readiness** - Validates network and MPI configuration
10. **Atomic Operations** - Checks if atomic operations are enabled on GPUs

Tests **9** and **10** can be skipped when not relevant (single-node systems, containers without full PCI/RDMA).
Kernel parameter checks (**3**) can optionally be reported as warnings only instead of failures.

### Component Tests (--all mode)

Tests installed ROCm components by compiling and executing example programs:

- HIP (hipcc, hip-runtime-amd)
- Math Libraries (hipBLAS, hipFFT, rocBLAS, rocFFT, etc.)
- Primitives (hipCUB, rocPRIM, rocThrust)
- Solvers (hipSOLVER, rocSOLVER, rocSPARSE)
- Deep Learning (MIOpen)
- Applications (from rocm-examples repository)

## Output

The tool provides three types of output:

1. **Console Output** - Real-time test progress and results
2. **Summary Tables** - Formatted tables showing:
   - General system information
   - GPU device information
   - Firmware version information
   - Test results with status and details
3. **JSON Export** - Detailed results in JSON format for further analysis

## Install dependency pip packages

```bash
sudo pip3 install -r requirements.txt
```

## Usage

```bash
./rdhc.py -h
usage: sudo -E rdhc.py [options]

ROCm Deployment Health Check Tool

optional arguments:
  -h, --help            show this help message and exit
  --quick               Run quick tests only (default)
  --all                 Default tests + Compile and executes simple program for each component.
  -v, --verbose         Enable verbose output
  -s, --silent          Silent mode (errors only)
  -j FILE, --json FILE  Export results to JSON file
  -d DIR, --dir DIR     Directory path for temporary files (default: /tmp/rdhc/)
  --rocm-install-prefix DIR  ROCm installation prefix. If set, this path is used; otherwise `ROCM_PATH` env or `/opt/rocm` is used.
  --skip-multinode-readiness   Skip multinode cluster readiness test (e.g. single-node or no RDMA stack)
  --skip-atomic-operations     Skip PCIe atomic operations test (e.g. containers or limited lspci)
  --skip-optional-cluster-checks  Same as `--skip-multinode-readiness` and `--skip-atomic-operations` together
  --kernel-params-warnings-only   Treat kernel parameter check failures as warnings instead of errors
```

### Usage examples

```bash
# Run quick test (default tests only)
sudo -E ./rdhc.py

# Run all tests including compile and execute the rocm-example program for each component
sudo -E ./rdhc.py --all

# Run all tests with verbose output
sudo -E ./rdhc.py --all -v

# Enable verbose output
sudo -E ./rdhc.py -v

# Run in silent mode (only errors shown)
sudo -E ./rdhc.py -s

# Export results to a specific JSON file
sudo -E ./rdhc.py --all --json rdhc-results.json

# Specify a directory for temp files and logs (default: /tmp/rdhc/)
sudo -E ./rdhc.py -d /home/user/rdhc-dir/

# Custom install prefix
sudo -E ./rdhc.py --rocm-install-prefix /usr/local/rocm

# Custom prefix and run all tests
sudo -E ./rdhc.py --rocm-install-prefix /usr/local/rocm --all -v

# Skip multinode + atomic checks (single-node or container-friendly)
sudo -E ./rdhc.py --skip-optional-cluster-checks

# Or skip only one of them
sudo -E ./rdhc.py --skip-multinode-readiness
sudo -E ./rdhc.py --skip-atomic-operations

# Kernel cmdline/sysctl checks: mismatches count as warnings, not FAIL
sudo -E ./rdhc.py --kernel-params-warnings-only

# Combine with other options
sudo -E ./rdhc.py --all --skip-optional-cluster-checks --kernel-params-warnings-only -v
```

## Optional CLI behavior

These options improve usability when some checks are not applicable or cannot run reliably:

| Option | Purpose |
|--------|---------|
| `--skip-multinode-readiness` | Skips the **Multinode Cluster Readiness** test (MPI, NIC drivers, RDMA modules). Use on **single-node** hosts or systems **without** an InfiniBand/RDMA stack. |
| `--skip-atomic-operations` | Skips the **Atomic Operations** test (`lspci`-based). Use in **containers** or hosts where **PCIe capability** details are not visible without full access. |
| `--skip-optional-cluster-checks` | Equivalent to passing **both** `--skip-multinode-readiness` and `--skip-atomic-operations`. |
| `--kernel-params-warnings-only` | For the **Kernel Parameters** test, entries that would normally count as **errors** are treated as **warnings** so the overall check does not **FAIL** solely on kernel/cmdline tuning differences. |

Skipped tests are recorded with status **NOT TESTED** and a short reason in the report/JSON.

## RDHC Environment VARIABLES

RDHC tool will use the following ENV variables and act accordingly if they are set.

```bash
# ROCm installation path can be set by the below ENV variable. Default is "/opt/rocm/"
export ROCM_PATH="/opt/rocm"

# For library dependency validation, the lib search depth can be set by the below ENV.
# Default is full depth. It checks for all the lib files in ROCM_PATH/lib/ folder recursively.
export LIBDIR_MAX_DEPTH=""

# if you want to check the libs only from the ROCM_PATH/lib/ folder set the depth as 1.
export LIBDIR_MAX_DEPTH=1
```

## Troubleshooting

### Python Package Installation Issues (Ubuntu 24.04)

If `sudo pip3 install` fails with an "externally-managed-environment" error (common in Ubuntu 24.04), use a Python virtual environment instead:

```bash
# Create a virtual environment (one-time setup)
python3 -m venv ~/rdhc-venv

# Activate the virtual environment
source ~/rdhc-venv/bin/activate

# Install required packages
pip3 install -r requirements.txt
```

**Note for Ubuntu 24.04 users:** Due to enhanced security policies, `sudo -E` does not preserve the virtual environment PATH. Replace all `sudo -E` commands with `sudo --preserve-env=PATH` in the usage examples above.

For example:

```bash
# Instead of: sudo -E ./rdhc.py
# Use:
source ~/rdhc-venv/bin/activate
sudo --preserve-env=PATH ./rdhc.py

# Run all tests
sudo --preserve-env=PATH ./rdhc.py --all

# Run with verbose output
sudo --preserve-env=PATH ./rdhc.py -v
```

---

The tool is designed to be easily extended with additional component tests by adding new test methods following the naming convention `test_check_component_name()`.
