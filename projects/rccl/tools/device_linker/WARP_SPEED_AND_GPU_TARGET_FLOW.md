# WARP_SPEED and GPU target flow (device linker builds)

## What you asked to check

Whether the WARP_SPEED toggle and the **GPU target string** (with feature flags) are properly passed to:

1. Specialized kernel builds (`specialized_kernels_device`)
2. Dispatcher build (CMake `dispatcher_device.elf` + `build_with_device_linker.sh`)

---

## 1. WARP_SPEED (the macro) — **passed correctly**

- **rccl target**: Gets `ENABLE_WARP_SPEED` when `ENABLE_WARP_SPEED` is ON (CMakeLists.txt ~1084–1085). That is set from the WARP_SPEED toggle (install.sh passes `-DENABLE_WARP_SPEED=ON` when `warp_speed_enabled` is true).
- **specialized_kernels_device**: Gets rccl’s compile definitions via `get_target_property(RCCL_DEFS rccl COMPILE_DEFINITIONS)` and `target_compile_definitions(specialized_kernels_device PRIVATE ${RCCL_DEFS})` (~1330–1332). So **ENABLE_WARP_SPEED is passed** to specialized kernels.
- **Dispatcher (CMake)**: `DISPATCHER_DEFS` includes `-DENABLE_WARP_SPEED` when `ENABLE_WARP_SPEED` is ON (~1418–1419). So **ENABLE_WARP_SPEED is passed** to the dispatcher device ELF compile.
- **build_with_device_linker.sh**: Hardcodes `-DENABLE_WARP_SPEED` in `DEFINES` for the host object that embeds the fatbin (~124). So the dispatcher host side also has the macro.

So the **WARP_SPEED toggle is passed** to specialized kernels and to the dispatcher (device and host).

---

## 2. GPU target string (e.g. `gfx942:sramecc+:xnack-`) — **only from `GPU_TARGETS`**

The **architecture string** (with or without `:sramecc+:xnack-`) is **not** derived from WARP_SPEED. It comes only from **`GPU_TARGETS`**:

- **specialized_kernels_device**: Uses **only** `FIRST_GPU_TARGET` for `--offload-arch`:
  - `list(GET GPU_TARGETS 0 FIRST_GPU_TARGET)` (~1381)
  - `target_compile_options(..., "--offload-arch=${FIRST_GPU_TARGET}", ...)` (~1386)
  - Any `--offload-arch=...` from the rccl target is **intentionally skipped** (~1370–1371) and replaced by this one.
- **Dispatcher device ELF (CMake)**: Same:
  - `list(GET GPU_TARGETS 0 FIRST_GPU_TARGET)` (~1443)
  - `--offload-arch=${FIRST_GPU_TARGET}` in the custom command (~1454).
- **build_with_device_linker.sh**: Gets the same value as the second argument:
  - CMake calls: `build_with_device_linker.sh ${CMAKE_CURRENT_BINARY_DIR} ${FIRST_GPU_TARGET}` (~1479–1481).
  - Script uses it as `GPU_TARGET_FULL`; if it has no `:`, it **defaults** `GPU_TARGET_FEATURES="sramecc+:xnack-"` and uses that for the **bundle** target and device linker `--target`. So the **fatbin/bundle** can have the full target even when `FIRST_GPU_TARGET` is base-only. But the **device code** (specialized .o and dispatcher_device.elf) was already compiled by CMake with `--offload-arch=${FIRST_GPU_TARGET}`. So if `FIRST_GPU_TARGET` is `gfx942` (no features), that code is built **without** the feature string; only the bundle metadata has the full target.

So:

- **If `GPU_TARGETS` (and thus `FIRST_GPU_TARGET`) already contains the full string** (e.g. `gfx942:sramecc+:xnack-`), then specialized kernels and the dispatcher device ELF are built with the full target and things are consistent.
- **If `GPU_TARGETS` is base-only** (e.g. `gfx942`), then:
  - Specialized kernels and dispatcher device ELF are built with **base arch only**.
  - The script still puts the full target in the **bundle** and device linker `--target`, but the **object code** was compiled for base arch only → possible **INVALID_ISA** / “invalid kernel file” if the runtime expects the full target.

So the critical question is: **what is in `GPU_TARGETS` at configure time?**

- From **install.sh**: `-DGPU_TARGETS=${build_amdgpu_targets}` only if the user passes `--amdgpu_targets`.
- Otherwise `GPU_TARGETS` comes from **DEFAULT_GPUS** (~89). That is overridden only when **BUILD_LOCAL_GPU_TARGET_ONLY** is ON (install.sh `-l`): then `rocm_local_targets(DEFAULT_GPUS)` is used (~81–82). So with `./install.sh -l --device-linker`, `GPU_TARGETS` is whatever **rocm_agent_enumerator** returns (in `cmake/Dependencies.cmake`). If that returns only `gfx942`, then `FIRST_GPU_TARGET` is base-only and specialized + dispatcher are **not** getting the feature string.

---

## 3. Summary

| Item | Specialized kernels | Dispatcher (device ELF) | build_with_device_linker.sh |
|------|----------------------|--------------------------|-----------------------------|
| **ENABLE_WARP_SPEED** | Yes (via RCCL_DEFS) | Yes (DISPATCHER_DEFS) | Yes (hardcoded in DEFINES) |
| **GPU target string** | `FIRST_GPU_TARGET` only | `FIRST_GPU_TARGET` only | Arg 2 = `FIRST_GPU_TARGET`; script adds default features for **bundle** only |

So:

1. **WARP_SPEED toggle** is passed correctly to specialized kernel builds and to the dispatcher build.
2. **GPU target string** (with features) is passed to specialized and dispatcher **only if** `GPU_TARGETS` (and thus `FIRST_GPU_TARGET`) already contains it. That depends on:
   - `-DGPU_TARGETS=...` from install.sh, or
   - `rocm_local_targets()` (when using `-l`) returning the full target.

If the release build normally gets the full feature string into the library (e.g. via rocm or local GPU detection), then for device-linker builds we need to ensure **the same** full target is in `GPU_TARGETS` so that `FIRST_GPU_TARGET` is the full string. Otherwise, the device-linker path should be updated so that the **compilation** of specialized kernels and dispatcher device ELF uses the full target (e.g. by having the script or CMake add the default features when `FIRST_GPU_TARGET` has no colon), not only the bundle.
