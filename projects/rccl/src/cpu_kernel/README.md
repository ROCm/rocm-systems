# RCCL CPU kernel launcher

Optional host-side execution path for RCCL collective device kernels when GPU
compute is scarce but CPU cores are available.

## Enable at runtime

```bash
export RCCL_CPU_KERNEL_ENABLE=1
```

Optional:

```bash
# Max blocks (channels) to run in parallel (default: active channel count)
export RCCL_CPU_KERNEL_THREADS=16

# Confirm CPU launches in logs (rank 0 only)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,ENV
```

When active, rank 0 prints lines like:

```text
Launching RCCL collective kernel on CPU (grid=... block=... nWorkBatches=... stream=...)
```

## Design

- **Isolation**: all logic lives under `src/cpu_kernel/`. The only integration
  point in the main tree is a guarded branch in `ncclLaunchKernel()` inside
  `enqueue.cc`.
- **Ordering**: `cpu_memory_model.h` implements MI300 (gfx942) store/barrier
  semantics used by RCCL device code (`RELAXED` peer stores, block-scope
  release/acquire barriers, `SEQ_CST` abort polling).
- **Launch contract**: work is scheduled with `cudaLaunchHostFunc` on the same
  user stream as GPU kernels, preserving stream ordering with surrounding HIP
  calls and proxy follow-up in `ncclLaunchKernelAfter_NoCuda()`.

## Supported plans

CPU launch is used only when:

- `RCCL_CPU_KERNEL_ENABLE` is set
- Plan is not CE, symmetric, persistent, or persistent work storage

## Collective execution

When enabled, `cpu_dispatch.cc` decodes `funcId` via `cpu_func_decode.cc` (using
the same `ncclDevFuncNameToId` table as device code) and runs host equivalents
of ring/tree collectives through `cpu_primitives.cc` (SIMPLE protocol, MI300
ordering) and `cpu_coll_exec.cc`.

Supported collectives (RING/TREE/PAT/CollNet/NVLS channel work via ring primitives):

- AllReduce, Broadcast, Reduce, AllGather, ReduceScatter
- SendRecv / P2p
- AlltoAllPivot / GDA (ring/simple path)

LL/LL128 batches are executed using the SIMPLE CPU transport path (same
algorithms; LL fifo timing is not reimplemented on host).

Device pointers in user buffers are staged through `cpu_mem.cc` when not
host-accessible.

## Files

| File | Role |
|------|------|
| `cpu_kernel_launcher.cc` | Env params, stream enqueue, thread pool |
| `cpu_kernel_main.cc` | Host `ncclKernelMain` loop |
| `cpu_work_loader.cc` | Host `loadWorkBatchToShmem` |
| `cpu_dispatch.cc` | Work dispatch / barriers |
| `cpu_dev_comm_mirror.cc` | Device comm/channel mirror for host reads |
| `cpu_memory_model.h` | MI300 memory ordering primitives |
