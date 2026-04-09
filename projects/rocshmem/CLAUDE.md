# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROCm OpenSHMEM (rocSHMEM) is a GPU-centric networking library that provides an OpenSHMEM-like interface for AMD GPUs. It enables intra-kernel networking, allowing GPU kernels to directly perform communication operations without host intervention.

## Architecture

rocSHMEM has three backend implementations, selectable at build time:

1. **IPC Backend** (`src/ipc/`): Uses GPU load/store operations for communication within a single node
2. **Reverse Offload (RO) Backend** (`src/reverse_offload/`): Forwards GPU networking requests to host-side MPI/OpenSHMEM runtime
3. **GDA Backend** (`src/gda/`): GPU Direct Async - allows GPU to communicate directly with NIC without CPU proxy
   - Vendor-specific implementations in `src/gda/bnxt/`, `src/gda/ionic/`, `src/gda/mlx5/`

### Key Components

- **Context Management** (`src/context*.{cpp,hpp}`): Manages device and host-side communication contexts
- **Memory Management** (`src/memory/`): Symmetric heap allocation with device memory allocators (dlmalloc or Pow2Bins)
- **Bootstrap** (`src/bootstrap/`): Initialization and PE discovery
- **Synchronization** (`src/sync/`): Barriers and fences
- **Teams** (`src/team*.{cpp,hpp}`): Team-based collective operations
- **Templates** (`src/templates*.hpp`): Template implementations for RMA, AMO, and collective operations

Public API defined in `include/rocshmem/`:
- `rocshmem.hpp`: Main header with host and device interfaces
- `rocshmem_RMA.hpp`, `rocshmem_AMO.hpp`, `rocshmem_COLL.hpp`: Operation categories
- `rocshmem_config.h`: Generated configuration (from `cmake/rocshmem_config.h.in`)

Version is defined in `include/rocshmem/rocshmem.hpp` as `constexpr char VERSION[]`.

## Building

rocSHMEM uses CMake and requires ROCm, HIP, and HSA runtime. Out-of-source builds are mandatory.

### Dependencies

Install UCX and Open MPI dependencies:
```bash
./scripts/install_dependencies.sh
# Set INSTALL_DIR to customize installation location (default: ./install)
```

### Build Configurations

Pre-configured build scripts in `scripts/build_configs/`:

**Reverse Offload + Network (typical development):**
```bash
mkdir build && cd build
../scripts/build_configs/ro_net
```

**IPC (single node):**
```bash
mkdir build && cd build
../scripts/build_configs/ipc_single
```

**GDA (GPU Direct Async):**
```bash
mkdir build && cd build
../scripts/build_configs/gda
# Specify NIC vendor: gda_mlx5, gda_bnxt, or gda_ionic
```

**All backends:**
```bash
mkdir build && cd build
../scripts/build_configs/all_backends
```

### Manual CMake Configuration

```bash
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/rocshmem \
    -DUSE_RO=ON \
    -DUSE_IPC=OFF \
    -DUSE_GDA=OFF \
    -DBUILD_FUNCTIONAL_TESTS=ON \
    -DBUILD_UNIT_TESTS=ON \
    ..
cmake --build . --parallel 8
cmake --install .
```

### Key CMake Options

**Backends:**
- `USE_RO`: Enable Reverse Offload backend (default: ON)
- `USE_IPC`: Enable IPC backend (default: OFF)
- `USE_GDA`: Enable GDA backend (default: OFF)
- `GDA_MLX5`, `GDA_BNXT`, `GDA_IONIC`: Enable specific NIC vendor support

**Build Options:**
- `BUILD_FUNCTIONAL_TESTS`: Build functional tests - requires MPI (default: OFF)
- `BUILD_UNIT_TESTS`: Build unit tests - requires MPI (default: OFF)
- `BUILD_EXAMPLES`: Build examples (default: ON)
- `BUILD_TOOLS`: Build rocshmem_info tool (default: ON)
- `BUILD_LOCAL_GPU_TARGET_ONLY`: Build only for detected GPUs (default: OFF)

**Memory Configuration:**
- `USE_HEAP_DEVICE_FINEGRAIN`: GPU memory in finegrain mode (default: ON)
- `USE_HEAP_DEVICE_COARSEGRAIN`: GPU memory in coarsegrain mode (default: OFF)
- `USE_ALLOC_DLMALLOC`: Use dlmalloc allocator (default: ON)
- `USE_ALLOC_POW2BINS`: Use legacy Pow2Bins allocator (default: OFF)

**Debug/Profile:**
- `BUILD_DEBUG_LEVEL_TRACE_HOST`: Compile in host-side trace logging (LOG_TRACE) (default: OFF)
- `BUILD_DEBUG_LEVEL_TRACE_DEVICE`: Compile in device-side trace logging (LOGD_TRACE) (default: OFF)
- `BUILD_DEBUG_LEVEL_DEVICE`: Compile in device-side logging — ROCSHMEM_*_DEVICE printf and legacy GPU_DPRINTF (default: OFF). Note: abort() in ROCSHMEM_ABORT_DEVICE/ROCSHMEM_ERROR_DEVICE is always compiled regardless.
- `PROFILE`: Enable statistics and timing (default: OFF)
- `BUILD_CODE_COVERAGE`: Enable code coverage flags (gcc only) (default: OFF)

**GPU Targets:**
Default targets: gfx90a, gfx1100, gfx1201, gfx942, gfx950 (ROCm 7+)
Override with: `-DGPU_TARGETS="gfx90a;gfx942"`

## Testing

### Functional Tests

Located in `tests/functional_tests/`. Tests for RMA operations, atomics, collectives, barriers, etc.

Run via driver script:
```bash
cd build
../scripts/functional_tests/driver.sh
```

The driver script supports running specific tests by name (matches TestType enum in `tests/functional_tests/tester.hpp`):
```bash
# Run specific test types
../scripts/functional_tests/driver.sh get put barrierall
```

### Unit Tests

Located in `tests/unit_tests/`. Smaller focused tests using MPI for multi-PE setup.

Run via driver script:
```bash
cd build
../scripts/unit_tests/driver.sh
```

### Examples

Located in `examples/`. Built by default unless `BUILD_EXAMPLES=OFF`.

## Contributing

- Base branches on `develop` (development branch)
- Pull requests must target `develop` branch
- All commits require `Signed-off-by` line (use `git commit -s`)
- Follow Developer's Certificate of Origin

## Documentation

Online documentation: https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/

Build docs locally (from `docs/` directory):
```bash
# HTML
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

# PDF (requires LaTeX)
sphinx-build -M latexpdf . _build
```

## Environment Variables

rocSHMEM behavior is controlled via environment variables defined in `src/envvar.hpp` and `src/envvar.cpp`. Variables are grouped into categories with distinct prefixes:

| Category | Prefix | Description |
|---|---|---|
| Core | `ROCSHMEM` | Core configuration |
| Bootstrap | `ROCSHMEM_BOOTSTRAP` | Initialization/discovery |
| Reverse Offload | `ROCSHMEM_RO` | RO backend options |
| GDA | `ROCSHMEM_GDA` | GPU Direct Async options |

### Key Variables

**Core (`ROCSHMEM_*`):**
- `ROCSHMEM_DEBUG_LEVEL`: Debug/verbosity level. Values: `NONE` (default), `ERROR`, `VERSION`, `WARN`, `ENV` or `ENV:MODIFIED`, `ENV:ALL`, `ENV:FULL`, `INFO`, `TRACE`. The `ENV*` levels print environment variable state at startup. Levels are ordered by increasing verbosity; each level includes all messages from less verbose levels.
- `ROCSHMEM_HEAP_SIZE`: Symmetric heap size in bytes per PE (default: 1 GiB)
- `ROCSHMEM_BACKEND`: Backend selection (`ipc`, `ro`, `gda`); empty = auto-detect
- `ROCSHMEM_MAX_NUM_CONTEXTS`: Max contexts an application can use (default: 32)
- `ROCSHMEM_MAX_NUM_TEAMS`: Max teams (default: 40)
- `ROCSHMEM_USE_IB_HCA`: NIC to bind to (e.g., `bnxt_re0`); empty = auto-detect
- `ROCSHMEM_DISABLE_MIXED_IPC`: Force network conduit even when IPC is available

**Bootstrap (`ROCSHMEM_BOOTSTRAP_*`):**
- `ROCSHMEM_BOOTSTRAP_TIMEOUT`: Initialization timeout in seconds (default: 5)
- `ROCSHMEM_BOOTSTRAP_SOCKET_FAMILY`: Socket family (`AF_UNSPEC`, `AF_INET`, `AF_INET6`)
- `ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME`: Network interface for bootstrap (e.g., `eno8303`)

**Reverse Offload (`ROCSHMEM_RO_*`):**
- `ROCSHMEM_RO_PROGRESS_DELAY`: Progress engine delay in microseconds (default: 3)

**GDA (`ROCSHMEM_GDA_*`):**
- `ROCSHMEM_GDA_PROVIDER`: NIC vendor (`bnxt`, `ionic`, `mlx5`); empty = auto-detect
- `ROCSHMEM_GDA_ENABLE_DMABUF`: Enable dmabuf memory registration (default: false)
- `ROCSHMEM_GDA_PCIE_RELAXED_ORDERING`: Enable PCIe Relaxed Ordering (default: false)

### Printing Environment Variables Programmatically

`src/envvar.hpp` exposes `rocshmem::envvar::print_envvars(mode, os)` with three modes:
- `print_mode::MODIFIED` (default): print only non-default variables
- `print_mode::ALL_VALUES`: print all variables with their current values
- `print_mode::FULL_DOCUMENTATION`: print all variables with description, default, and current value

### rocshmem_info Tool

The `rocshmem_info` tool (built when `BUILD_TOOLS=ON`) displays build info and environment variables:

```bash
rocshmem_info                  # build info + modified env vars (default)
rocshmem_info --env:all        # build info + all env vars with values
rocshmem_info --env:full       # build info + env vars with full documentation
rocshmem_info -o output.txt    # write output to a file
```

## Code Organization Notes

- Device-side code uses `__device__` and HIP kernel launches
- Host-side initialization in `src/rocshmem.cpp`
- GPU-side runtime in `src/rocshmem_gpu.cpp`
- Backend-specific context implementations: `context_{ro,ipc,gda}_*.{cpp,hpp}`
- Proxy classes (e.g., `*_proxy.hpp`) provide device-side interfaces to host-allocated resources
