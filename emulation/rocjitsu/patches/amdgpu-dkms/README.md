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
