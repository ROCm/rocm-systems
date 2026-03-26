# Plan: Multi-Architecture Parallel Build + install.sh Integration

## 1. Multi-Architecture Parallel Split-Device Compile

### Current state

`SplitDeviceCompile.cmake` accepts a single `GPU_ARCH`. Line 1046 of
`src/CMakeLists.txt` takes only the first arch:

```cmake
list(GET GPU_TARGETS 0 _split_gpu_arch)
```

A production build with e.g. `GPU_TARGETS="gfx908;gfx90a;gfx942;gfx950"` only
compiles for the first architecture.

### Goal

All architectures compile in one flat pool of custom commands so ninja sees all
`N_arches × N_TUs` bc/obj compiles as independent work. No sequential arch loop
at build time.

### Pipeline shape (example: 4 arches × 859 TUs)

```
Phase 1 — fully parallel (3436 bc + 3436 obj compiles):
  src.cpp → fname.gfx908.bc → fname.gfx908.o
  src.cpp → fname.gfx90a.bc → fname.gfx90a.o
  src.cpp → fname.gfx942.bc → fname.gfx942.o
  src.cpp → fname.gfx950.bc → fname.gfx950.o

Phase 2 — 4-way parallel:
  ld.lld:  all *.gfx908.o → combined.gfx908.so
  ld.lld:  all *.gfx90a.o → combined.gfx90a.so
  ld.lld:  all *.gfx942.o → combined.gfx942.so
  ld.lld:  all *.gfx950.o → combined.gfx950.so

Phase 3 — single step:
  bundler:  all combined.ARCH.so → combined.hipfb

Phase 4 — per-arch host stubs (4-way parallel):
  common.cu + combined.hipfb → combined.fat.gfx908.o
  common.cu + combined.hipfb → combined.fat.gfx90a.o
  common.cu + combined.hipfb → combined.fat.gfx942.o
  common.cu + combined.hipfb → combined.fat.gfx950.o

Phase 5 — final link:
  all host .o + all combined.fat.ARCH.o → librccl.so
```

### Changes to `SplitDeviceCompile.cmake`

- Change `GPU_ARCH` parameter to `GPU_ARCHS` (list).
- Outer loop over arches, inner loop over sources for bc/obj compilation.
  Output files are already namespaced: `${fname}.${arch}.bc`, `${fname}.${arch}.o`.
- Accumulate `ALL_DEV_OBJECTS_${arch}` per architecture.
- Kernel TU (common.cu) asm-patch path: runs once per arch (each arch has its
  own `.set` directives). Produces `common.cu.${arch}.o` per arch.
- Per-arch `ld.lld` link: one `add_custom_command` per arch producing
  `combined.${arch}.so`. Independent across arches → ninja runs them in parallel.
- Per-arch response file with object compression flag for `ld.lld`.
- Single multi-arch bundler step:
  - `--targets=host-x86_64-...,hip-amdgcn-...--gfx908,hip-amdgcn-...--gfx90a,...`
  - `--input=/dev/null,combined.gfx908.so,combined.gfx90a.so,...`
  - Depends on ALL `combined.${arch}.so` files.
- Per-arch host stub: compile `common.cu` host-only once per arch, each with a
  single `--offload-arch=ARCH` and the shared `combined.hipfb`. Produces
  `combined.fat.${arch}.o`. This is safer than a single multi-arch
  `--offload-host-only` invocation.
- Export `DEVICE_FAT_OBJECTS` as a list of all per-arch fat objects.

### Changes to `src/CMakeLists.txt`

- Replace `list(GET GPU_TARGETS 0 ...)` with passing the full `GPU_TARGETS`
  list.
- Build bundler target strings for each arch.
- Pass `GPU_ARCHS ${GPU_TARGETS}` to `setup_split_device_compile`.
- The `gfx950`-specific `RCCL_ARGS_IN_SCRATCH` define: handled inside the
  per-arch loop in `SplitDeviceCompile.cmake`.
- `LINK_DEPENDS` and `target_link_options` updated to reference all fat objects.

### What stays the same

- `generate.py` — unchanged. Generated `.cpp` files are arch-independent.
- `hipify` — unchanged. Hipified sources are arch-independent.
- Host compilation — unchanged. Still fully overlapped via `LINK_DEPENDS`.

---

## 2. install.sh Integration

### 2a. Add `--split-compile` flag

**New variable** (with defaults near line 47):

```bash
build_split_compile=false
```

**Add to `display_help()`:**

```
       --split-compile        Enable split device compilation (requires ninja-build)
```

**Add to `getopt` longopts** (line 125):

Add `split-compile` to the longoptions string.

**Add to `case` block** (after line 177):

```bash
     --split-compile)        build_split_compile=true;  shift ;;
```

**When `build_split_compile=true`:**

- Force ninja (same pattern as `--time-trace` on lines 391-401):

```bash
if [[ "${build_split_compile}" == true ]]; then
    if ! hash ninja &>/dev/null ; then
        echo "ninja could not be found"
        echo "Use \"${time_trace_ninja_msg}\" to install ninja"
        exit 1
    fi
    build_system="ninja"
    enable_ninja="-GNinja"
    cmake_common_options="${cmake_common_options} -DENABLE_SPLIT_DEVICE_COMPILE=ON"
fi
```

This should be placed before the existing `--time-trace` ninja block (lines
391-401) or merged with it, since both require ninja.

### 2b. Fix hardcoded `make` references

- **Line 427** (`VERBOSE=1`): This is make syntax. For ninja, verbose is `-v`.
  Fix:

```bash
if [[ "${build_verbose}" == true ]]; then
    if [[ "${build_system}" == "ninja" ]]; then
        build_system="${build_system} -v"
    else
        build_system="${build_system} VERBOSE=1"
    fi
fi
```

- **Line 440** (`make package`): Hardcoded to make. Fix:

```bash
if [[ "${build_package}" == true ]]; then
    ${build_system} package
    check_exit_code "$?"
fi
```

### 2c. All existing flags should work with `--split-compile`

The following `install.sh` flags affect CMake options that are already
propagated into the split pipeline via `get_target_property` on the `rccl`
target (include dirs, compile defs, compile opts):

- `--address-sanitizer` → `BUILD_ADDRESS_SANITIZER`
- `--debug` / `--debug-fast` → `CMAKE_BUILD_TYPE`
- `--enable_backtrace` → `BUILD_BFD`
- `--disable-colltrace` → `COLLTRACE`
- `--local_gpu_only` / `--amdgpu_targets` → `GPU_TARGETS`
- `--enable-msccl-kernel` → `ENABLE_MSCCL_KERNEL`
- `--npkit-enable` → `ENABLE_NPKIT`
- `--generate-sym-kernels` → `GENERATE_SYM_KERNELS`
- `--force-reduce-pipeline` → `FORCE_REDUCE_PIPELINING`
- `-t` / `--tests_build` → `BUILD_TESTS`
- `--rocshmem` → `ENABLE_ROCSHMEM`
- `--cmake-options` → passthrough

No special handling needed for any of these. The split pipeline reads from the
`rccl` target properties, so any definitions/options set on `rccl` are
automatically forwarded.

---

## 3. Verify Default (Non-Split) Build Still Works

The `generate.py` change to `impl_filename()` produces one function per `.cpp`
file regardless of whether split compilation is enabled. In the non-split path,
these files are compiled as normal HIP sources via `hipcc` inside
`add_library(rccl ...)`. Having 859 source files instead of ~180 is standard
CMake and should work. However, this needs a verification build:

```bash
./install.sh --debug -f    # fast debug, no split compile
```

This confirms the non-split path still compiles and links correctly with the
finer-grained generated files.

---

## Summary of files to change

| File | Change |
|---|---|
| `cmake/SplitDeviceCompile.cmake` | `GPU_ARCH` → `GPU_ARCHS` list; nested arch×TU loop; per-arch ld.lld; multi-arch bundler; per-arch host stub |
| `src/CMakeLists.txt` | Pass full `GPU_TARGETS` list; build per-arch bundler targets; update fat object list |
| `install.sh` | Add `--split-compile` flag; force ninja; fix hardcoded `make` references |
