# Why specialized-kernel flags match non–device-linker, and what DEVICE_LINKER changes

## 1. Why the flags are the same as the non–device-linker build (for arch and everything else we pass through)

**Non–device-linker build:** The `rccl` target compiles all HIP sources (host + device) with `rccl`’s compile options. So every `.o` (including device code) is built with whatever `rccl` has: `--offload-arch`, defines, includes, warnings, etc. Arch comes from rocm-cmake / `GPU_TARGETS` (or the commented block that adds `--offload-arch=${target}` to `rccl`).

**Device-linker build (our change):** For the specialized kernel object library we:

- Take **rccl’s COMPILE_OPTIONS** and filter only the flags that are incompatible with device-only compilation (`-fno-gpu-rdc`, `-Xclang`/`-triple=`, `-mcpu=`, `-MF`/`-MD`). We do **not** invent or replace arch.
- Use **only the first `--offload-arch`** from `rccl` (so we still build one arch per file and avoid a 10x slowdown). So the arch string (including features) is exactly what the normal build would use for its first target.
- **If `rccl` has no `--offload-arch`** (e.g. device-linker build where that block is commented out and rocm-cmake doesn’t add it), we add **`--offload-arch=${FIRST_GPU_TARGET}`** so we still use the same `GPU_TARGETS` as the rest of the project (so **-l** = local arch = e.g. gfx942).

So for **arch and every other flag we pass through**, the specialized kernel build uses the same source of truth as the normal build: `rccl`’s options, with a single-arch restriction and a fallback from `GPU_TARGETS` when `rccl` has no arch.

The **only** extra flags we add are the ones needed to produce **device-only** `.o` files:

- `-x hip`
- `--cuda-device-only`
- `--no-gpu-bundle-output`

The normal build does **not** use these because it compiles full HIP (host + device) into each object. So “the same as non–device-linker” is true for **arch and all other non–device-only flags**; the three above are the minimal device-only tweaks.

---

## 2. What else DEVICE_LINKER changes (beyond specialized-kernel flags)

**-l (BUILD_LOCAL_GPU_TARGET_ONLY)** is independent of DEVICE_LINKER. It controls:

- **GPU_TARGETS:** With `-l`, `rocm_local_targets(DEFAULT_GPUS)` runs and `GPU_TARGETS` becomes the local arch(s) (e.g. `gfx942`).
- **Generation:** Both `generate.py` and `generate_specialized.py` are called with `BUILD_LOCAL_GPU_TARGET_ONLY` (ON with `-l`). That drives **which unrolls/pipelines** are generated:
  - **Local arch only (ON):** `calc_unroll_and_pipeline_for_local_arch()` uses rocminfo and generates only the unroll/pipeline variants needed for the local GPU → fewer `.cpp` files (e.g. not all unrolls 1,2,4).
  - **All archs (OFF):** generate all unrolls (1, 2, 4) and pipelines.

So **-l** already “means other things, like not generating all the unrolls”; that comes from `BUILD_LOCAL_GPU_TARGET_ONLY` passed into the Python generators, not from DEVICE_LINKER.

**What DEVICE_LINKER changes:**

| Area | Effect |
|------|--------|
| **Generation** | When DEVICE_LINKER, **generate_specialized.py** is run (in addition to generate.py), with `BUILD_LOCAL_GPU_TARGET_ONLY` and `GPU_TARGETS_STR`. Same **-l** behavior for unrolls. |
| **HIP_SOURCES for rccl** | DEVICE_LINKER: only `host_table.cpp`, `device_table.cpp` (+ sym_kernels_stub if needed); **common.cu.cpp** and **onerank.cu.cpp** excluded. Normal: all generated files under GEN_DIR added. |
| **Specialized kernel .cpp** | Not added to HIP_SOURCES; built via **specialized_kernels_device** OBJECT library with the flags above. |
| **rccl target** | Gets **DEVICE_LINKER** and **USE_INDIRECT_FUNCTION_CALL** defines; links **${HIP_LIBRARY}** only (no **hip::device**); no **-fgpu-rdc**; no **--offload-compress** when DEVICE_LINKER. |
| **Dispatcher** | Custom command builds **dispatcher_device.elf**; custom command runs **build_with_device_linker.sh** → **dispatcher_final.o**; **rccl** depends on **device_linker_dispatcher**. |

So DEVICE_LINKER changes **what gets compiled where** (skeleton rccl + separate device OBJECT library + dispatcher pipeline), **not** how **-l** or unroll generation works; those are driven by BUILD_LOCAL_GPU_TARGET_ONLY and the same generators.
