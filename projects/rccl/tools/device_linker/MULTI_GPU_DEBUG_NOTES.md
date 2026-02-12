# Multi-GPU zeros: what was tried (device linker build)

## Symptom
- 2+ GPU collectives (AllReduce, batched AllReduce) return zeros; 1-GPU is skipped.
- Dispatcher runs, correct device function is called (jumptable OK), but batches appear not to be processed.

## Changes made (you can revert any)

### 1. HIP launch: pass shared memory size (src/enqueue.cc)
- **Was:** `hipExtLaunchKernel(..., extra, 0, ...)` — passed 0 for sharedMemBytes.
- **Now:** Pass `smem` (rcclShmemDynamicSize) so the kernel gets dynamic LDS for `extern __shared__ ncclShmemPerWarp[]`.
- **HIP clamp:** On HIP we clamp `smem` to `min(device MaxSharedMemoryPerBlockOptin - attr.sharedSizeBytes)` so total LDS does not exceed device limit.

### 2. HIP launch: kernel args in device memory (src/enqueue.cc, init.cc, include/comm.h)
- **Idea:** On discrete GPU, the kernel receives a *pointer* to the args. That pointer was host-allocated (`plan->kernelArgs` from ncclMemoryStackAlloc → malloc). The device cannot read host memory, so the kernel was reading garbage/zeros.
- **Now:** 
  - In init: allocate `comm->kernelArgsBufDev` (device buffer, size `workArgsBytes`) when `__HIP_PLATFORM_AMD__` and `__HIPCC__`.
  - **Before any launch** (so both the single-stream path and the fallback path use it): `hipMemcpyAsync(comm->kernelArgsBufDev, plan->kernelArgs, plan->kernelArgsSize, ...)` on the launch stream, then set `extra[0] = kernelArgsBufDev`. So all HIP launch paths pass the device pointer.

### 3. Smoke tests: 2 channels (tools/device_linker/smoke_test/run_tests.sh)
- Export `NCCL_MIN_NCHANNELS=2` and `NCCL_MAX_NCHANNELS=2` (default) so runs use 2 channels (16 waves) for easier debugging.

### 4. Device-side debug dump (kernel → host)
- **Purpose:** See what the kernel actually sees (args, first batch) without attaching a debugger.
- **Mechanism:**  
  - `ncclDevKernelArgs` has an optional `debugOut` (device pointer).  
  - At init (HIP): allocate `comm->kernelDebugBufDev` (64 bytes).  
  - Plan build: `plan->kernelArgs->debugOut = comm->kernelDebugBufDev`.  
  - In the kernel (`ncclKernelMain`): thread 0 block 0 writes 8× uint64: `comm`, `channelMask.masks[0]`, `workStorageType`, and first batch `offsetBitset`, `workType`, `funcId`, `offsetBase`, `flags`.  
  - After the single-stream HIP launch: host syncs, copies the 64 bytes back, and prints one `INFO(NCCL_ALL, ...)` line with those values.
- **How to see it:** Run with `NCCL_DEBUG=INFO` (or higher). The line appears after each kernel launch on the single-stream path.
- **Interpretation:** If `comm` is 0 or `offsetBitset` is 0, the kernel is not seeing the expected args/batch. A second line dumps the first work struct after load: `sendbuff`, `recvbuff`, `nWorks`, `countLo`. If `sendbuff` or `recvbuff` is 0 (or looks like a host address), the kernel is reading/writing the wrong buffers and the result will be wrong (e.g. zeros).

### 5. Optional device debug traps (main thread only)

- **Purpose:** Stop in ROCgdb at specific points to confirm execution reaches them. If a trap never fires, that code path is not run.
- **Mechanism:** In `ncclKernelMain` (device/common.h), three optional `__builtin_trap()` points are guarded by `(tid == 0 && blockIdx.x == 0 && args->debugOut)`. Before trapping, the trap id (1, 2, or 3) is written to `args->debugOut[0]` so you can confirm which trap in the debugger.
- **Trap points (all enabled by one flag `NCCL_DEVICE_DEBUG_TRAP`):**  
  - **0:** Generic kernel entry (start of `ncclDevKernel_Generic_*` in common.cu).  
  - **1:** Right after args copy (before writing the normal debug dump).  
  - **2:** After work batch load and sync, before the `while (ncclShmem.aborted == 0)` loop.  
  - **3:** Right before the indirect call `ncclDevFuncTable_2[ncclShmem.funcId]()`.  
  - **4:** Just before kernel exit (end of `ncclKernelMain`).
- **Enable:** Build with `-DNCCL_DEVICE_DEBUG_TRAP` (e.g. `./install.sh -l --device-linker --device-debug-trap` or `NCCL_DEVICE_DEBUG_TRAP=1`). Run under ROCgdb; trap id (s_trap 0–4) shows which point.

## Result
- Smoke tests and RCCL unit test (AllReduce.OutOfPlace) still fail: output 0.0 instead of expected sum.
- Full device-linker pipeline was run, then `make rccl`; same failure. So the merged device binary is current.
- hipStreamSynchronize after memcpy was tried; reverted (no change). So async ordering is not the cause.
- Root cause remains unknown; kernel-args device copy is in place for all HIP launch paths.

## What to try next
1. **Confirm HIP path:** Log or assert in the `planner->numStreams == 1 && !plan->persistent` branch to ensure it’s taken for the smoke test.
2. **Confirm kernel args copy:** Log after `hipMemcpyAsync` and ensure `comm->kernelArgsBufDev != nullptr` at launch.
3. **Use device debug dump:** Run with `NCCL_DEBUG=INFO` and check the "NCCL kernel debug:" line after launch; interpret `comm`, `offsetBitset`, etc. (see §4 above).
4. **Full rebuild:** Run the full device-linker pipeline and rebuild from scratch:  
   `rm -rf build; ./install.sh -l --device-linker` then rebuild smoke tests and run.
5. **NCCL_DEBUG:** Run with `NCCL_DEBUG=INFO` or `NCCL_DEBUG=WARN` to see if init or launch report errors.
6. **Sync before launch (debug only):** Temporarily add `hipStreamSynchronize(launchStream)` after the memcpy and before the launch to rule out async ordering.

## Env for 2 channels
```bash
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=2
```
