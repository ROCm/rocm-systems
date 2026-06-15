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
usage: ./rdhc.py [options]

ROCm Deployment Health Check Tool

optional arguments:
  -h, --help            show this help message and exit
  --quick               Run quick tests only (default)
  --all                 Default tests + Compile and executes simple program for each component.
  -v, --verbose         Enable verbose output
  -s, --silent          Silent mode (errors only)
  -j FILE, --json FILE  Export results to JSON file
  -d DIR, --dir DIR     Directory path for temporary files (must not exist; created with mode 0700)
  --cleanup             Remove auto-generated temp directory after tests complete
  --rocm-install-prefix DIR  ROCm installation prefix. If set, this path is used; otherwise `ROCM_PATH` env or `/opt/rocm` is used.
  --skip-multinode-readiness   Skip multinode cluster readiness test (e.g. single-node or no RDMA stack)
  --skip-atomic-operations     Skip PCIe atomic operations test (e.g. containers or limited lspci)
  --skip-optional-cluster-checks  Same as `--skip-multinode-readiness` and `--skip-atomic-operations` together
  --kernel-params-warnings-only   Treat kernel parameter check failures as warnings instead of errors
```

### Usage examples

```bash
# Run quick test (default tests only)
./rdhc.py

# Run all tests including compile and execute the rocm-example program for each component
./rdhc.py --all

# Run all tests with verbose output
./rdhc.py --all -v

# Enable verbose output
./rdhc.py -v

# Run in silent mode (only errors shown)
./rdhc.py -s

# Export results to a specific JSON file
./rdhc.py --all --json rdhc-results.json

# Specify a directory for temp files (directory must not exist)
./rdhc.py --all -d /home/user/rdhc-run1/

# Auto-generate temp directory and clean it up after tests
./rdhc.py --all --cleanup

# Custom install prefix
./rdhc.py --rocm-install-prefix /usr/local/rocm

# Custom prefix and run all tests
./rdhc.py --rocm-install-prefix /usr/local/rocm --all -v

# Skip multinode + atomic checks (single-node or container-friendly)
./rdhc.py --skip-optional-cluster-checks

# Or skip only one of them
./rdhc.py --skip-multinode-readiness
./rdhc.py --skip-atomic-operations

# Kernel cmdline/sysctl checks: mismatches count as warnings, not FAIL
./rdhc.py --kernel-params-warnings-only

# Combine with other options
./rdhc.py --all --skip-optional-cluster-checks --kernel-params-warnings-only -v
```

**Note on privileges:** Most checks run without elevated privileges. Some checks (e.g., `lspci -vvv` for atomic operations) may require root access. Run as root when necessary, or use `--skip-atomic-operations` if running unprivileged.

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

# Run the tool (virtual environment activated)
./rdhc.py
./rdhc.py --all
```

### Temporary Directory Behavior

When using `--all` mode, RDHC clones and builds rocm-examples:

- **Without `-d` flag**: A secure temporary directory is auto-generated in `/tmp` (e.g., `/tmp/rdhc-abc123/`) with mode `0700`. By default, this directory persists after the run. Use `--cleanup` to remove it automatically.
- **With `-d` flag**: The specified directory is created with mode `0700`. The directory **must not already exist** (for security reasons). If you need to reuse a directory, remove it first: `rm -rf /path/to/dir`

Example workflow for repeated runs:

```bash
# Auto-generated temp dir (persists)
./rdhc.py --all
# Creates /tmp/rdhc-abc123/ (check logs or report for actual path)

# Auto-generated temp dir with automatic cleanup
./rdhc.py --all --cleanup
# Creates and removes temp dir after tests complete

# Custom directory (must not exist)
./rdhc.py --all -d /tmp/my-rdhc-run1/   # First run with this path
rm -rf /tmp/my-rdhc-run1/                # Clean up before next run
./rdhc.py --all -d /tmp/my-rdhc-run1/   # Can reuse after cleanup

# Note: --cleanup has no effect when -d is specified
```

---

The tool is designed to be easily extended with additional component tests by adding new test methods following the naming convention `test_check_component_name()`.
