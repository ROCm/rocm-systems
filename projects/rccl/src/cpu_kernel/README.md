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

## Limitations

Device collective bodies (`RunWorkColl`, `Primitives`, etc.) are not yet fully
ported to host. The CPU path executes kernel **orchestration** (channel mapping,
work-batch loading, barriers, counter writeback) with MI300 memory fences.
Intranode transport still relies on the existing proxy/network path where
applicable. Full correctness for all algorithms/protocols requires extending
`cpu_dispatch.cc` with host ports of the device primitive templates.

## Files

| File | Role |
|------|------|
| `cpu_kernel_launcher.cc` | Env params, stream enqueue, thread pool |
| `cpu_kernel_main.cc` | Host `ncclKernelMain` loop |
| `cpu_work_loader.cc` | Host `loadWorkBatchToShmem` |
| `cpu_dispatch.cc` | Work dispatch / barriers |
| `cpu_dev_comm_mirror.cc` | Device comm/channel mirror for host reads |
| `cpu_memory_model.h` | MI300 memory ordering primitives |
