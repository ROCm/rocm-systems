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

## rocminfo Output

```
Agent 2
  Name:                    gfx942
  Device Type:             GPU
  Compute Unit:            64
  Shader Engines:          8
  SDMA engine uCode::      24
  Pool 1: GLOBAL COARSE GRAINED 512MB
```
