# PC Sampling Kernel Backend (draft)

This folder implements the **KFD / KGD split** for PC sampling ioctls in rocprofiler-sdk.

## What changed

Previously, `ioctl_adapter.cpp` opened `/dev/kfd` directly. Now:

```
ioctl_adapter.cpp          (thin public API — unchanged symbols)
    └── kernel/selector    picks backend from ROCPROFILER_KERNEL_IFACE
            ├── kfd_backend   /dev/kfd + AMDKFD_IOC_PC_SAMPLE
            └── kgd_backend   /dev/dri/renderD* + DRM_AMDGPU_IOCTL_PC_SAMPLE
```

**Start / stop / destroy** still go through ROCr (`hsa_ven_amd_pcs_*` → libhsakmt). This PR only refactors **query** and **create** in rocprofiler-sdk.

## Environment variable

| `ROCPROFILER_KERNEL_IFACE` | Behavior |
|----------------------------|----------|
| `kfd` (default) | Use `/dev/kfd` — same as before |
| `kgd` | Use render node for the agent (`agent->drm_render_minor`) |
| `auto` | Try KGD probe first, fall back to KFD |

Example:

```bash
export ROCPROFILER_KERNEL_IFACE=kgd
rocprofv3 --pc-sampling-beta-enabled ...
```

## How the driver is probed

We use **ioctl capability probes**, not `/sys/module/amdgpu/version`:

### KFD backend

1. `AMDKFD_IOC_GET_VERSION` → require **major ≥ 1, minor ≥ 16**
2. `AMDKFD_IOC_PC_SAMPLE` + `QUERY_CAPABILITIES` → read **PCS implementation version** in `args.version`
3. Per-gfx/method gates in `pcs_common.cpp` (same table as before)

### KGD backend

1. Open `/dev/dri/renderD{128 + drm_render_minor}`
2. `DRM_AMDGPU_IOCTL_PC_SAMPLE` + `QUERY_CAPABILITIES` → must succeed (or `-ENOSPC` for empty buffer)
3. Same **PCS implementation version** + gfx/method matrix

### What to ask the kernel team

Please confirm:

1. Final **DRM ioctl number** and struct name in `amdgpu_drm.h` (placeholder in `details/amdgpu_drm_pcs.h`)
2. Whether KGD needs a **KFD-style interface version** check or only the **PCS impl version** from `QUERY_CAPABILITIES`
3. Minimum **PCS impl version** values per gfx for the render-node path (we reuse the KFD matrix until told otherwise)
4. When **ROCr/libhsakmt** will route start/stop to the same backend (required for end-to-end KGD)

## Files

| File | Purpose |
|------|---------|
| `backend.hpp` | `PcSamplingBackend` interface |
| `kfd_backend.*` | KFD path + `get_kfd_fd()` |
| `kgd_backend.*` | Render-node path |
| `selector.*` | Env parsing + backend cache |
| `pcs_common.*` | Shared ioctl helpers + version matrix |
| `render_node_fd.*` | Cached render-node fds |

## Related planning

See `pcs-refactor-ioctls/planning/03-pc-sampling-migration-plan.md`.
