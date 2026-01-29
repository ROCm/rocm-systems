# Device Linker Build Process - Detailed Description

This document describes the current build process when `DEVICE_LINKER=ON`.

## Phase 1: CMake Configuration (runs once)

**Command:**
```bash
cmake -DDEVICE_LINKER=ON -DBUILD_LOCAL_GPU_TARGET_ONLY=ON ...
```

**Steps during configuration:**

### 1. Generate specialized kernel sources (~1s, sequential)
```bash
python3 src/device/generate_specialized.py \
    build/hipify/gensrc/specialized \
    True \
    "" \
    "gfx942:xnack+:sramecc+"
```

| | |
|---|---|
| **Input** | Template in `generate_specialized.py`, GPU targets |
| **Output** | 831 `.cpp` files in `build/hipify/gensrc/specialized/` |

Each file contains one `ncclDevKernel_*_Specialized` kernel calling one `ncclDevFunc_*`.

### 2. Glob specialized files and add to HIP_SOURCES

- All 831 `.cpp` files added to the `rccl` target's source list
- Each file marked with `-fno-gpu-rdc` compile option

---

## Phase 2: Parallel Compilation (ninja/make -jN)

**~900 source files compiled in parallel** using:

```bash
/opt/rocm/bin/amdclang++ -O3 -fPIC -x hip --offload-arch=gfx942:xnack+:sramecc+ \
    --offload-compress -fno-gpu-rdc \
    -I... (include paths) \
    -c <source>.cpp -o <source>.cpp.o
```

**Key files:**

| Category | Count | Compile Flags | Output Location |
|----------|-------|--------------|-----------------|
| Core RCCL sources | ~60 | default (no -fno-gpu-rdc) | `CMakeFiles/rccl.dir/hipify/src/*.o` |
| Specialized kernels | 831 | `-fno-gpu-rdc` | `CMakeFiles/rccl.dir/hipify/gensrc/specialized/*.o` |

**Each specialized `.o` file contains:**
- Host code (empty stub functions)
- `.hip_fatbin` section with compressed device code (~20KB each)

---

## Phase 3: PRE_LINK Device Linker Pipeline (sequential)

Runs **after** all compilation, **before** final link. All steps are sequential.

### Step 0: Compile dispatcher kernel
```bash
/opt/rocm/bin/hipcc -c -fPIC -fno-gpu-rdc \
    --offload-arch=gfx942:xnack+:sramecc+ --offload-compress \
    -I... tools/device_linker/merged_minimal.hip \
    -o tools/device_linker/merged_minimal.o
```

| | |
|---|---|
| **Input** | `merged_minimal.hip` (defines `ncclDevKernel_Merged_1/2/4` with function pointer tables) |
| **Output** | `merged_minimal.o` (11KB) - bundled host+device object |

### Step 0b: Extract host-only object
```bash
/opt/rocm/llvm/bin/clang-offload-bundler --type=o \
    --targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack+ \
    --input=merged_minimal.o \
    --output=host_only.o --output=/dev/null \
    --unbundle --allow-missing-bundles
```

| | |
|---|---|
| **Input** | `merged_minimal.o` |
| **Output** | `host_only.o` (11KB) - contains `__hip_module_ctor` registration code |

### Step 0c: Extract device ELF from compressed fatbin
```bash
python3 tools/device_linker/extract_device_from_fatbin.py \
    merged_minimal.o minimal_device.o gfx942:xnack+:sramecc+
```

| | |
|---|---|
| **Input** | `merged_minimal.o` |
| **Output** | `minimal_device.o` (13KB) - raw device ELF with dispatcher kernels + empty function tables |

### Step 1: Run C++ device linker
```bash
tools/device_linker/device_linker \
    -o merged_device_final.o \
    --dispatcher minimal_device.o \
    --host-table build/hipify/gensrc/host_table.cpp \
    --target gfx942:sramecc+:xnack+ \
    --input-dir CMakeFiles/rccl.dir/hipify/gensrc/specialized/
```

| | |
|---|---|
| **Inputs** | `minimal_device.o` - dispatcher template with empty tables |
| | `host_table.cpp` - funcId → function name mapping |
| | 831 `.o` files in input-dir - each with device code in `.hip_fatbin` |
| **Output** | `merged_device_final.o` (1.7MB) - merged device ELF |

**Processing steps:**
1. Parse `host_table.cpp` to get funcId → function name mapping
2. For each specialized `.o`: extract device ELF from compressed `.hip_fatbin`
3. Extract `.text` (code) and `.rodata` (constants) from each device ELF
4. Merge all code into single `.text` section
5. Build symbol table mapping function names to offsets
6. Populate `func_table_1[]`, `func_table_2[]`, `func_table_4[]` with function pointers
7. Generate merged device ELF with proper headers

### Step 2: Strip device code from specialized .o files (CURRENTLY FAILS)
```bash
find CMakeFiles/rccl.dir/hipify/gensrc/specialized/ -name "*.o" \
    -exec /opt/rocm/llvm/bin/llvm-objcopy --remove-section=.hip_fatbin {} \;
```

| | |
|---|---|
| **Intent** | Remove redundant device code so only merged code is linked |
| **Status** | **FAILS** - `.hipFatBinSegment` has relocations pointing to `.hip_fatbin` |

### Step 3: Bundle host + merged device
```bash
/opt/rocm/llvm/bin/llvm-objcopy --remove-section=.hip_fatbin host_only.o
/opt/rocm/llvm/bin/clang-offload-bundler --type=o \
    --targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack+ \
    --input=host_only.o \
    --input=merged_device_final.o \
    --output=merged_final_host.o
```

| | |
|---|---|
| **Input** | `host_only.o`, `merged_device_final.o` |
| **Output** | `merged_final_host.o` - bundled object ready for linking |

---

## Phase 4: Final Link

```bash
/opt/rocm/bin/amdclang++ -fPIC -O3 \
    merged_final_host.o \        # Our merged dispatcher + device code
    -parallel-jobs=16 \
    --offload-compress \
    --hip-link --offload-arch=gfx942:xnack+:sramecc+ \
    -shared -Wl,-soname,librccl.so.1 \
    -o librccl.so.1.0 \
    @CMakeFiles/rccl.dir/objects1.rsp \  # 831 specialized .o + 60 core .o
    -L/opt/rocm/lib -lamdhip64 ...
```

**Current Problem:** `objects1.rsp` contains 831 specialized `.o` files, each with `.hip_fatbin`. The linker:
1. Creates `.hip_fatbin` section from all 831 specialized `.o` files (~19MB)
2. Also includes `__CLANG_OFFLOAD_BUNDLE__` sections from `merged_final_host.o` (~1.7MB)
3. Runtime loads the wrong (individual) device code instead of merged code

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONFIGURATION                                                               │
│   generate_specialized.py → 831 .cpp files                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PARALLEL COMPILATION (make -j)                                              │
│                                                                             │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│   │ specialized_1.cpp│    │ specialized_2.cpp│    │ specialized_N.cpp│     │
│   │        │         │    │        │         │    │        │         │     │
│   │   amdclang++     │    │   amdclang++     │    │   amdclang++     │     │
│   │   -fno-gpu-rdc   │    │   -fno-gpu-rdc   │    │   -fno-gpu-rdc   │     │
│   │        │         │    │        │         │    │        │         │     │
│   │        ▼         │    │        ▼         │    │        ▼         │     │
│   │ specialized_1.o  │    │ specialized_2.o  │    │ specialized_N.o  │     │
│   │ (host+.hip_fatbin│    │ (host+.hip_fatbin│    │ (host+.hip_fatbin│     │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘     │
│            │                       │                       │               │
└────────────┼───────────────────────┼───────────────────────┼───────────────┘
             │                       │                       │
             ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRE_LINK: DEVICE LINKER PIPELINE (sequential)                               │
│                                                                             │
│   merged_minimal.hip ──hipcc──▶ merged_minimal.o                            │
│                                      │                                      │
│                          ┌───────────┴───────────┐                          │
│                          ▼                       ▼                          │
│                    host_only.o            minimal_device.o                  │
│                    (11KB host)            (13KB device ELF)                 │
│                          │                       │                          │
│                          │                       ▼                          │
│   specialized_*.o ───────┼────────────▶   device_linker                     │
│   (831 files)            │               (merges device code)               │
│                          │                       │                          │
│                          │                       ▼                          │
│                          │            merged_device_final.o                 │
│                          │                  (1.7MB)                         │
│                          │                       │                          │
│                          └───────────┬───────────┘                          │
│                                      ▼                                      │
│                          clang-offload-bundler                              │
│                                      │                                      │
│                                      ▼                                      │
│                           merged_final_host.o                               │
│                    (host + merged device bundle)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FINAL LINK                                                                  │
│                                                                             │
│   amdclang++ -shared                                                        │
│       merged_final_host.o           ◀── Our merged code (GOOD)              │
│       specialized_1.o ... _N.o      ◀── Individual device code (BAD)        │
│       core_rccl_*.o                                                         │
│           │                                                                 │
│           ▼                                                                 │
│       librccl.so                                                            │
│       ├── .hip_fatbin (19MB) ◀── From 831 specialized .o (WRONG)            │
│       └── __CLANG_OFFLOAD_BUNDLE__ (1.7MB) ◀── Our merged code (IGNORED)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Core Problem

The linker combines `.hip_fatbin` from all input `.o` files. We **cannot remove** `.hip_fatbin` from the specialized `.o` files because `.hipFatBinSegment` (the HIP module registration code) has relocations pointing to it.

### Potential Solutions

1. **Compile specialized files separately** - Use custom `hipcc` commands outside CMake's target, don't link them
2. **Use linker script** - Exclude `.hip_fatbin` sections from specific object files
3. **Modify HIP runtime registration** - Create our own `__hip_module_ctor` that only registers merged code
4. **Post-process the final .so** - Strip unwanted fatbin entries after linking
