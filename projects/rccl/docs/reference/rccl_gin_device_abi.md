# RCCL GIN Device Bitcode ABI

The `rccl_gin_*` functions below are part of RCCL's device bitcode (the
`bindings/ir` build, enabled with `-DEMIT_LLVM_IR=ON`). They ship in the same
artifact as the other `nccl*` device thunks; there is no separate GIN-only
option.

## Building and installing

```text
cmake -DEMIT_LLVM_IR=ON -DBITCODE_GPU_ARCHS="gfx950;gfx942" ...
```

- `BITCODE_GPU_ARCHS` takes comma- or semicolon-separated targets. Feature
  suffixes (`gfx950:sramecc+:xnack-`) are passed to `--offload-arch` and
  stripped from the file name. When empty, only `BITCODE_LIB_ARCH` (default
  `gfx950`) is built.
- One `librccl_device_<arch>.bc` is produced per architecture. The first
  architecture is also copied to the unsuffixed `librccl_device.bc`.
- Files are installed to `${CMAKE_INSTALL_LIBDIR}/rccl/`, next to the existing
  `librccl_device.bc`, together with `nccl_device_wrapper.h`.
- Exports are the functions marked `NCCL_IR_EXPORT` (`__attribute__((used))`),
  not an explicit list. The build fails if any of them is not defined with
  external linkage in the final IR, or if a required `rccl_gin_*` symbol is
  missing (`bindings/ir/verify_exports.cmake`).
- Discovery by a consumer (for example an environment-variable directory
  override) is the consumer's responsibility; RCCL installs files but defines no
  lookup variable.

No architecture is listed here as validated. See
`rccl_gin_bitcode_feasibility.md` for the gfx942 status.

The artifact is configuration-specific. A build with `ENABLE_ROCSHMEM_GIN=ON`
is compiled with the matching backend defines and must be consumed with an RCCL
host library built with that option; do not mix artifacts from different
backend builds under the same installation directory.

## Functions

All symbols are unmangled `extern "C" __device__` functions. Window and
communicator arguments are opaque device pointers; offsets, counts, IDs, and
operations are scalar values. Each call acts as a single thread
(`ncclCoopThread`) and waits use relaxed, 64-bit polling. The `signal_op`
values are RCCL's `ncclGinSignalOp_t` values (`ncclGinSignalInc` or
`ncclGinSignalAdd`); `signal_arg` is only meaningful for `ncclGinSignalAdd`.

```text
void rccl_gin_put(void* dev_comm, int context_index, int peer,
                  void* dst_window, size_t dst_offset,
                  void* src_window, size_t src_offset, size_t bytes)
void rccl_gin_put_signal(void* dev_comm, int context_index, int peer,
                         void* dst_window, size_t dst_offset,
                         void* src_window, size_t src_offset, size_t bytes,
                         uint32_t signal, int signal_op, uint64_t signal_arg)
void rccl_gin_put_counter(void* dev_comm, int context_index, int peer,
                          void* dst_window, size_t dst_offset,
                          void* src_window, size_t src_offset, size_t bytes,
                          uint32_t counter)
void rccl_gin_wait_signal(void* dev_comm, int context_index,
                          uint32_t signal, uint64_t least)
void rccl_gin_wait_counter(void* dev_comm, int context_index,
                           uint32_t counter, uint64_t least)
void rccl_gin_flush(void* dev_comm, int context_index)
```

`rccl_gin_flush` waits until every put previously issued on the context is
locally complete (acquire ordering, the same default as `ncclGin::flush`). Call
it before a producer kernel exits.

`dev_comm` points to a live `ncclDevComm` created with GIN resources
(`ginForceEnable` or non-zero `ginSignalCount`/`ginCounterCount`), and
windows must be RCCL-registered. The functions do not initialize host-side
communication resources.

## Known limitations

- The put wrappers call `ncclGinPut_v2`, which cannot express strong signals or
  windows containing HOST_NUMA segments. `ncclGinPut_v3` can.

## Tests

`bindings/ir/test/IR_gin_mpi_test.cpp` cases `D1_PutSignal_TwoRanks` and
`D2_PutCounterThenSignal_TwoRanks` drive these functions on two ranks. They
require GIN-capable hardware and are not run by default.
