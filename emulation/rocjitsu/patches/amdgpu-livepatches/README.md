# amdgpu kernel livepatches for rocjitsu vfio-user

These kernel live patches enable the `amdgpu` driver to load and the KFD stack to
initialize when the GPU is a rocjitsu vfio-user emulated device (no real hardware).

Unlike the DKMS patches in `../amdgpu-dkms/`, these do not require rebuilding the
amdgpu kernel module. They are applied at runtime to the running kernel using the
Linux kernel live patching infrastructure (`CONFIG_LIVEPATCH=y`).

## Build

```sh
make -C /lib/modules/$(uname -r)/build M=$(pwd) amdgpu_vfu_livepatch.ko
```

Requires:
- Kernel headers matching the running kernel
- `CONFIG_LIVEPATCH=y` in kernel config
- `CONFIG_MODULE_SIG_FORCE=n` or signing key (module is unsigned)

## Install

```sh
sudo cp amdgpu_vfu_livepatch.ko /lib/modules/$(uname -r)/kernel/extra/
sudo depmod -a
```

## Load Order

The livepatch **must** be loaded before `amdgpu`:

```sh
sudo modprobe gpu_sched
sudo modprobe amdgpu_vfu_livepatch
sudo modprobe amdgpu discovery=2 fw_load_type=0 ip_block_mask=0x1F \
             vm_update_mode=3 gpu_recovery=0
```

## Module Parameters for amdgpu

| Parameter | Value | Why |
|---|---|---|
| `discovery=2` | 2 | Load IP discovery from VRAM (where rocjitsu writes it) |
| `fw_load_type=0` | 0 | Direct firmware load (no PSP/SMU needed) |
| `ip_block_mask=0x1F` | 0x1F | Enable common+GMC+IH+GFX+SDMA; skip VCN/JPEG |
| `vm_update_mode=3` | 3 | CPU-based VM page table updates (no SDMA required for PT) |
| `gpu_recovery=0` | 0 | Disable GPU recovery after job timeout (no real GPU to reset) |

## Patched Functions

### `sdma_v4_4_2_ring_test_ring` → return 0
SDMA ring test writes a packet and polls for completion. Without real SDMA
hardware, this always times out. Stub to succeed immediately.

### `sdma_v4_4_2_ring_test_ib` → return 0
SDMA IB test via DRM scheduler. Same issue. Stub to succeed so
`ring->sched.ready = true` is set (enabling SDMA DRM jobs).

### `gfx_v9_4_3_ring_test_ib` → return 0
GFX compute ring IB test. Required for compute ring scheduler readiness.

### `amdgpu_gmc_allocate_vm_inv_eng` → return 0
Assigns VM TLB invalidation engines to rings. With 5 SDMA instances on MI350P
and a limited MMHUB engine bitmap, this exhausts available engines and returns
-EINVAL during `gmc_v9_0_late_init`, aborting probe. In emulation, TLB invalidation
is a no-op so all engines can share.

### `amdgpu_fence_wait_polling` → return 1
Polled fence wait for KIQ ring (TLB invalidation, MEC setup). Returns immediately
with "success" (positive timeout remaining), bypassing the actual wait for GPU
hardware to write the fence value.

### `amdkfd_fence_wait_timeout` → return 0
DQM fence wait for HWS SET_RESOURCES/MAP_PROCESS packet acknowledgement.
Without GPU firmware, these packets never execute. Return 0 immediately.

## What Still Needs Real GPU Execution

The following operations still trigger DRM job timeouts (10s) and fail:
- `hipMemset` / `hipMemcpy` (uses SDMA for DMA)
- `hipDeviceSynchronize` after kernel launch (waits for compute fence)
- Any actual GPU kernel execution

With `gpu_recovery=0`, timeouts fail with -ETIME instead of triggering GPU reset.
The HIP runtime returns an error rather than hanging indefinitely.

## HIP API Status

| API | Status |
|---|---|
| `hipGetDeviceCount` | ✓ returns 1 |
| `hipGetDeviceProperties` | ✓ gfx942, 64 CU, 512MB |
| `hipMalloc` / `hipFree` | ✓ |
| `hipMallocManaged` | ✓ allocates |
| `hipLaunchKernelGGL` | ✓ dispatches |
| `hipDeviceSynchronize` | ✗ DRM job timeout after 10s |
| `hipMemcpy` (D2H) | ✗ SDMA job timeout |
| `hipMemset` (GPU) | ✗ SDMA job timeout |

## Current Blocker: hipStreamCreate

`hipStreamCreate` requires:
1. KFD queue creation (now works via `kfd_queue_acquire_buffers` stub)
2. ROCr sends MAP_PROCESS / SET_RESOURCES PM4 packets via KIQ ring
3. MEC firmware processes these packets and acknowledges
4. ROCr user-space spins waiting for the acknowledgment in shared memory

Step 3 blocks: without MEC firmware executing PM4 packets via the KIQ ring,
the acknowledgment is never written. The ROCr main thread spins at 100% CPU
waiting for the shared memory to change.

**Next step**: implement KIQ ring PM4 packet emulation in rocjitsu-vfu.
When KIQ wptr advances (BAR2 doorbell index 0x000), parse the PM4 packets
(MAP_PROCESS, SET_RESOURCES) and write the expected acknowledgment values.

KIQ doorbell: `AMDGPU_DOORBELL_LAYOUT1_KIQ_START = 0x000`
BAR2 byte offset = 0x000 (same as uint64_t index 0x000)
Already intercepted by fence_service_loop — just need to add handling.

## HIP API Status (latest)

| API | Status | Notes |
|---|---|---|
| `hipGetDeviceCount` | ✓ returns 1 | |
| `hipGetDeviceProperties` | ✓ gfx942, 64 CU, 512MB | |
| `hipMalloc` / `hipFree` | ✓ | |
| `hipStreamCreate` | ✗ segfault | CREATE_QUEUE stub works but kernel DQM state not set up |
| `hipLaunchKernelGGL` | not reached | blocked by hipStreamCreate crash |
| `hipDeviceSynchronize` | not reached | |

## Stall Analysis (hipStreamCreate)

The HIP initialization sequence (before `hipStreamCreate` is called):

1. `GET_VERSION, ACQUIRE_VM, SET_MEMORY_POLICY` — succeed
2. `ALLOC_MEMORY_OF_GPU, MAP_MEMORY_TO_GPU` (×3) — succeed
3. `RUNTIME_ENABLE, SET_XNACK_MODE` — succeed
4. `CREATE_EVENT` (×2), `SET_TRAP_HANDLER` — succeed
5. `AMDKFD_IOC_SVM` (SET_ATTR for queue ring buffer) — stubbed via `svm_range_set_attr`
6. `CREATE_QUEUE` — stubbed, returns fake queue_id=0, doorbell_offset=0x40
7. Post-queue alloc/map → SIGSEGV

The crash in step 7 occurs because our `CREATE_QUEUE` stub returns success
without actually creating the kernel-side queue data structures (MQD, HQD,
DQM queue state). When ROCr subsequently calls `MAP_MEMORY_TO_GPU` to map
the queue context, the kernel's queue pointer is null, causing the segfault.

## Next Steps for Full HIP Dispatch

1. **Implement kernel-side queue creation** in the stub:
   - Initialize `pdd->qpd` queue structures
   - Or stub `pqm_create_queue` at the PQM level instead of the ioctl level

2. **Signal GPU completion events**: ROCr's worker threads spin in
   `WAIT_EVENTS` waiting for GPU dispatch completion. Without GPU interrupts
   to signal KFD events, these never complete. Need to signal events after
   each queue submission.

3. **User-mode SDMA queue completion**: `hipMemcpy(H→D)` uses a user-mode
   SDMA queue. Completion signal requires the rocjitsu-vfu SDMA WRITE_LINEAR
   packet executor to find and write to the IOVA-addressed completion signal.

## rocminfo Output (confirmed working)

```
Agent 2
  Name:                    gfx942
  Device Type:             GPU
  Compute Unit:            64
  Shader Engines:          8
  SDMA engine uCode::      24
  Pool 1: GLOBAL COARSE GRAINED 512MB
```
