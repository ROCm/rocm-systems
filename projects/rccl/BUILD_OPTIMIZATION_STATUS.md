# Split-Device Build Optimization Status

## What was done

### 1. One translation unit per ncclDevFunc (generate.py)

Modified `src/device/generate.py` so that `impl_filename()` produces a unique
filename for every ncclDevFunc variant (keyed on coll, algo, proto, redop, ty,
acc, pipeline, unroll). Previously, multiple functions shared a single .cpp file
grouped by (coll, redop, ty), giving ~180 TUs. Now each function gets its own
TU, yielding ~859 TUs for a gfx942 local-arch build (~2575 with all unroll
factors).

**File changed:** `src/device/generate.py` — `impl_filename()` function.

### 2. Host compilation overlaps device pipeline (CMakeLists.txt)

Replaced `add_dependencies(rccl rccl_device_objects)` with a `LINK_DEPENDS`
property on the `rccl` target. This allows all 105 host .cc/.cpp TUs to compile
in parallel with the device pipeline instead of waiting for it to finish. Only
the final link step (`librccl.so`) waits for `combined.fat.o`.

**File changed:** `src/CMakeLists.txt` — removed `add_dependencies(rccl
rccl_device_objects)`, added `set_property(TARGET rccl APPEND PROPERTY
LINK_DEPENDS "${DEVICE_FAT_OBJECTS}")`.

## Build timing (gfx942, local-arch, clean build)

| Phase | Wall span | CPU total | Count |
|---|---|---|---|
| A. Device bc compile | 0.5s – 44.3s (43.8s) | 5637s | 1718 |
| B. Device obj compile | 2.6s – 49.7s (47.2s) | 3871s | 1720 |
| C. Device ld.lld link | 49.7s – 49.8s (0.1s) | — | 1 |
| D. Offload bundler | 49.8s – 49.9s (0.1s) | — | 1 |
| E. Fat object compile | 49.9s – 50.7s (0.8s) | — | 1 |
| F. Host compile | 0.5s – 10.0s (9.5s) | 528s | 105 |
| G. Final link (librccl.so) | 50.7s – 50.8s (0.1s) | — | 1 |
| **Total** | **50.8s wall** | **10126s CPU** | **2127 steps** |

All 105 host TUs start before the device pipeline finishes (zero serialization).

## What was not done

**Ninja scheduling order for long-pole TUs:** CMake's Ninja generator
alphabetizes custom commands by output path, so controlling execution order from
CMake/generate.py is not practical without filename prefixes. The heaviest
device TUs (`premulsum_bf16_pipe1` variants, up to 15s each) start around
position ~185/859 alphabetically rather than first. This costs ~4s of drain
time at the end of the device phase but is not easily fixable.

## Other findings

- **I/O is not a bottleneck:** Building to `/dev/shm` (tmpfs) vs disk produced
  identical wall times (51.0s vs 50.9s). The build is entirely CPU-bound.
- **`BUILD_LOCAL_GPU_TARGET_ONLY`** prunes unroll factors only (859 vs 2575 TUs).
- **CMake configure time:** ~2s.

## Branch

`fast-build-plus-lds-v2`
