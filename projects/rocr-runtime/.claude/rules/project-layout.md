# Project Layout

| Component | Directory | Key Files |
|-----------|-----------|-----------|
| HSA Runtime (ROCr) | `runtime/hsa-runtime/` | Core runtime implementation |
| Public HSA headers | `runtime/hsa-runtime/inc/` | `hsa.h`, `hsa_ext_*.h` (public HSA API) |
| Core runtime | `runtime/hsa-runtime/core/` | Runtime, signals, queues, memory |
| ROCt Thunk (libhsakmt) | `libhsakmt/` | Thunk library, KFD ioctl wrappers |
| Thunk headers | `libhsakmt/include/` | `hsakmt.h`, `hsakmttypes.h` |
| Runtime tests | `rocrtst/` | HSA runtime validation and performance tests |
| Thunk tests | `libhsakmt/tests/kfdtest/` | ROCt validation tests |
| Build helpers | `cmake_modules/` | CMake utilities |
| Samples | `samples/` | Example HSA applications |

# Architecture Layers

```
┌─────────────────────────────────────┐
│  HSA Applications                   │
├─────────────────────────────────────┤
│  HSA Runtime (ROCr)                 │  ← runtime/ directory
│  - libhsa-runtime64.so              │
│  - Core, Signals, Queues, Memory   │
├─────────────────────────────────────┤
│  ROCt Thunk (libhsakmt)             │  ← libhsakmt/ directory
│  - Static library (linked into     │
│    libhsa-runtime64)                │
│  - KFD ioctl wrappers              │
├─────────────────────────────────────┤
│  AMDGPU Kernel Driver (ROCk)        │
└─────────────────────────────────────┘
```

# Build Artifacts

| Artifact | Description | Location after install |
|----------|-------------|------------------------|
| `libhsa-runtime64.so` | HSA runtime shared library (default) | `$PREFIX/lib/` |
| `libhsa-runtime64.a` | HSA runtime static library (if BUILD_SHARED_LIBS=OFF) | `$PREFIX/lib/` |
| `hsa*.h` | HSA public headers | `$PREFIX/include/hsa/` |
| `hsa-runtime64-config.cmake` | CMake package config | `$PREFIX/lib/cmake/hsa-runtime64/` |

# Test Suites

| Suite | Purpose | Build Location | Tests |
|-------|---------|----------------|-------|
| **rocrtst** | HSA runtime validation, performance, functional tests | `rocrtst/suites/test_common/build/` | Signal, memory, queue, agent tests |
| **kfdtest** | ROCt thunk validation tests | `libhsakmt/tests/kfdtest/build/` | Topology, memory, events, ioctl tests |

## Building rocrtst

```bash
cd rocrtst/suites/test_common
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="<rocm>;<llvm>" \
      -DROCM_DIR="<rocm>" \
      -DOPENCL_DIR="<rocm>" ..
make
make rocrtst_kernels
```

## Building kfdtest

```bash
cd libhsakmt/tests/kfdtest
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="<rocm>" \
      -DROCM_DIR="<rocm>" ..
make
```

# Key CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `BUILD_SHARED_LIBS` | ON | Build libhsa-runtime64 as shared (.so) vs static (.a) |
| `CMAKE_BUILD_TYPE` | Debug | Release/Debug/RelWithDebInfo |
| `CMAKE_INSTALL_PREFIX` | /opt/rocm | Install location |

# Packaging

| Format | Path | Files |
|--------|------|-------|
| RPM | `RPM/` | Post-install scripts |
| DEB | `DEB/` | Packaging metadata, changelog, copyright |

# Critical Paths

- **HSA API headers:** `runtime/hsa-runtime/inc/hsa*.h` — all public HSA API
- **ROCt API headers:** `libhsakmt/include/hsakmt*.h` — thunk API (internal to runtime)
- **CMake:** `CMakeLists.txt`, `runtime/CMakeLists.txt`, `libhsakmt/CMakeLists.txt`
- **Packaging:** `RPM/`, `DEB/` — must stay in sync with CMake install targets

# Quick Checks

## Check HSA API usage
```bash
FUNC="hsa_signal_create"
grep -rn "$FUNC" runtime/hsa-runtime/inc/ runtime/hsa-runtime/core/ rocrtst/
```

## Check libhsakmt linkage
```bash
# libhsakmt should be static and linked into libhsa-runtime64
nm -D build/libhsa-runtime64.so | grep hsaKmt  # Should show hsaKmt symbols
```

## Check test coverage
```bash
# Find all tests referencing a function
grep -rn "hsa_signal_wait" rocrtst/suites/
```
