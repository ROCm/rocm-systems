# Device Linker Context

## Environment

| | |
|---|---|
| **Machine** | MI300A (ringo) |
| **GPU Target** | gfx942 |
| **GPUs Available** | 4 |
| **ROCm Version** | 7.0 |
| **Build Path** | `/work/lmeadows/rocm-systems/projects/rccl/build/release` |

## Current Status

| Test | Status |
|------|--------|
| Single-GPU | WORKING |
| Multi-GPU | **HANGING** (kernel never completes) |
| System RCCL Multi-GPU | WORKING |

## Current Problem: Multi-GPU Kernel Hang

**Symptoms:**
- Single-GPU AllReduce works correctly
- Multi-GPU AllReduce hangs indefinitely during `hipStreamSynchronize()`
- Initialization succeeds, channels connect, kernel launches
- Kernel never completes

**Likely Root Cause:**
Connection pointers in `ncclShmem.groups[].recvConns[]` and `sendConns[]` are NULL or invalid.
These are populated from `channel->peers[peer]` in the Primitives constructor.

**Key code path (`prims_simple.h`):**
```cpp
if (flags & (RoleWaitRecv|RolePostRecv)) loadRecvConn(channel->peers[peer], ...);
if (flags & (RoleWaitSend|RolePostSend)) loadSendConn(channel->peers[peer], ...);
```

**Investigation approach:**
Use `rocgdb` to inspect LDS contents at hang point. Debug info is now working correctly
(DWARF5 line tables, proper function symbols).

## Build & Test

```bash
# Full rebuild
cd /work/lmeadows/rocm-systems/projects/rccl/build/release
rm -rf specialized_objs device_linker_output
mkdir -p specialized_objs device_linker_output
make -j

# Test single-GPU (should pass)
cd tools/device_linker/smoke_test
LD_LIBRARY_PATH=/work/lmeadows/rocm-systems/projects/rccl/build/release ./test_single_gpu

# Test multi-GPU (currently hangs)
LD_LIBRARY_PATH=/work/lmeadows/rocm-systems/projects/rccl/build/release ./test_two_gpu_simple
```

## Critical Constraint: Compilation Flags

All compilation units MUST have identical flags for consistent structure layouts:

| Flag | Purpose | Affects Layout |
|------|---------|----------------|
| `DEVICE_LINKER` | Function table dispatch | No |
| `ENABLE_FAULT_INJECTION` | Adds faults field to ncclShmemData | **Yes** |
| `ENABLE_WARP_SPEED` | Adds warpComm/warpChannel fields | **Yes** |
| `ENABLE_LL128` | Protocol selection | Possibly |

If structure layouts mismatch between host, dispatcher, and specialized kernels, 
the kernel will access wrong memory offsets and crash or hang.

## Key Documents

| Document | Purpose |
|----------|---------|
| `DEVICE_LINKER_REDESIGN.md` | **Primary design doc** - architecture, ELF layout, implementation phases, lessons learned, common mistakes |
| `LDS_LAYOUT.md` | Shared memory layout reference - actual offsets from disassembly analysis |
| `IFC_VS_DEVICE_LINKER_COMPARISON.md` | Detailed ELF comparison with production build |
| `BUILD_PROCESS.md` | Build pipeline diagrams and data flow |

## Key Source Files

| File | Purpose |
|------|---------|
| `tools/device_linker/device_linker.cpp` | The device linker tool - merges ELFs, patches debug info |
| `src/device/common.cu` | Dispatcher kernels (`ncclDevKernel_Generic_*`) |
| `src/device/common.h` | Function table declarations, dispatch logic |
| `src/device/prims_simple.h` | Primitives constructor - where connection pointers are loaded |
| `src/device/prims_ll.h` | LL protocol primitives |
| `src/device/generate_specialized.py` | Generates specialized kernel source files |

## Previously Fixed Issues

These are documented in detail in `DEVICE_LINKER_REDESIGN.md`:

1. COMGR "Cannot Find Global Var Sizes" - `__clang_gpu_used_external` symbol
2. PC-relative addressing - preserved .rodata layout
3. Helper function extraction - complete code blocks
4. ENABLE_WARP_SPEED mismatch - consistent flags
5. Shared memory limit - `extern __shared__` for dynamic allocation
6. DWARF5 debug line string offsets - `patchDwarf5StringOffsets()`
7. Specialized kernel symbol addresses - `func_offset` correction
