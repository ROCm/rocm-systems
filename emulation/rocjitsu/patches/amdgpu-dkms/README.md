# amdgpu patches for rocjitsu vfio-user

These patches apply to the ROCm amdgpu kernel driver source tree. They are
required to make `modprobe amdgpu` succeed when the GPU is a rocjitsu
vfio-user emulated device rather than real hardware.

Patches are applied via the `build-amdgpu-dkms` script in `kernel-tools`,
which accepts a `PATCH_DIRS` argument pointing at this directory.

## Patches

### 0002-vfu-mm-access-skip-drm-dev-enter.patch

**Problem:** During IP discovery, `amdgpu_discovery_read_binary_from_mem()`
calls `amdgpu_device_vram_access()` to read the discovery binary from VRAM.
The BAR0 aperture path (`aper_base_kaddr`) is not yet set up at this point
(it requires GMC init, which runs after discovery). The fallback is
`amdgpu_device_mm_access()`, which reads VRAM dword-by-dword via
MM_INDEX/MM_INDEX_HI/MM_DATA register writes through BAR5. However,
`amdgpu_device_mm_access()` gates all access behind `drm_dev_enter()`, which
returns false before `drm_dev_register()` is called — and discovery runs
before registration. The function silently returns without reading anything,
leaving the discovery buffer as all-zeros. The signature check then fails
with "get invalid ip discovery binary signature".

**Fix:** Remove the `drm_dev_enter()`/`drm_dev_exit()` guard from
`amdgpu_device_mm_access()`. The guard exists to prevent access after device
teardown (e.g. hotunplug), not during early initialization. At probe time the
device cannot be in teardown, so skipping the guard is safe.

**Applies to:** ROCm amdgpu `roc-7.2.x` and later.

**Note:** Not appropriate for upstream submission as-is. The correct long-term
fix is to set up a minimal BAR0 ioremap before IP discovery so the aperture
path is available.

### 0003-vfu-skip-compute-ib-tests-without-mec-fw.patch

**Problem:** When the amdgpu driver loads with `fw_load_type=0` (direct
firmware load, no PSP), `adev->gfx.mec_fw_version` stays 0. The delayed
init work (`delayed_init_work`) then runs `amdgpu_ib_ring_tests` on compute
rings whose DRM schedulers are initialized but whose compute queue mapping
(via KIQ MAP_QUEUES PM4 packets) never completed. Without CP firmware the PM4
packets never execute, so the compute queues are never enabled. When the IB
test calls `drm_sched_entity_init` for a compute ring's scheduler, `entity->rq`
is NULL (scheduler not ready), and `drm_sched_job_init` crashes with a NULL
dereference (`CR2: 0x120`).

**Fix:** Skip IB tests on `AMDGPU_RING_TYPE_COMPUTE` rings when
`adev->gfx.mec_fw_version == 0`.

**Applies to:** ROCm amdgpu `roc-7.2.x` and later.

### 0004-vfu-stub-amdkfd-drm-client-create.patch

**Problem:** `amdgpu_amdkfd_drm_client_create` is called from `amdgpu_pci_probe`
during device initialization. It triggers `amdgpu_vm_init` which crashes with
a NULL pointer dereference at offset 0x1c8 because `adev->vm_manager` fields
are not fully initialized at probe time in vfio-user emulation.

The crash call stack:
  `amdgpu_vm_init+0x359 ← amdgpu_driver_open_kms ← drm_client_init ←`
  `amdgpu_amdkfd_drm_client_create ← amdgpu_pci_probe+0x325`

**Fix:** Return 0 immediately from `amdgpu_amdkfd_drm_client_create`. KFD
topology registration still works; the DRM client is only needed for KFD
internal GPU memory operations not yet supported in vfio-user emulation.

**Applies to:** ROCm amdgpu `roc-7.2.x` and later.
