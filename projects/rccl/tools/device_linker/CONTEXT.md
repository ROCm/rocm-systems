# Device Linker Project Context

## Goal
Implement a "pure specialized kernel approach" for RCCL to reduce kernel launch latency by avoiding the slow `-fgpu-rdc` LTO linking. Instead, compile specialized kernels in parallel and merge them with a custom device linker.

## Current State
- CMake configured with `-DDEVICE_LINKER=ON` which:
  - Compiles specialized kernels with `-fno-gpu-rdc` (produces device ELFs, not bitcode)
  - Creates `specialized_objs/` directory with the compiled objects
  - Excludes `common.cu` from main build (handled by device linker pipeline)
- Device linker tool (`device_linker.cpp`) restored to working version from git commit `6a0adab5d4`
- Single-GPU test passes, multi-GPU still broken

## Key Files
```
/work/lmeadows/rocm-systems/projects/rccl/tools/device_linker/
├── device_linker.cpp          # Main tool - merges dispatcher + specialized kernels
├── build_with_device_linker.sh # 5-step build pipeline script
├── DEVICE_LINKER_DESIGN.md    # Design documentation
├── DEVICE_LINKER_REDESIGN.md  # Two-pass linker design (latest)
└── CONTEXT.md                 # This file
```

## Build Commands
```bash
# Configure (use correct GPU target for your system)
cd /work/lmeadows/rocm-systems/projects/rccl
cmake -B build/release -DCMAKE_BUILD_TYPE=Release -DDEVICE_LINKER=ON -DGPU_TARGETS=gfx950 -DBUILD_TESTS=ON

# Build specialized kernels (fast, parallel)
cd build/release && make -j specialized_kernels

# Run device linker pipeline
cd /work/lmeadows/rocm-systems/projects/rccl/tools/device_linker
./build_with_device_linker.sh /work/lmeadows/rocm-systems/projects/rccl/build/release gfx950

# Build full RCCL library
cd /work/lmeadows/rocm-systems/projects/rccl/build/release && make -j
```

## How It Works
1. **Specialized kernels** compiled with `-fno-gpu-rdc` → real device ELFs (not bitcode)
2. **Dispatcher** (`common.cu`) compiled separately with `DEVICE_LINKER_DISPATCH` → indirect function calls via tables
3. **Device linker** merges dispatcher + specialized kernels:
   - Copies dispatcher .text, .rodata (kernel descriptors)
   - Appends specialized kernel code to .text
   - Populates function pointer tables in .data
   - Patches PC-relative references
   - Generates relocations for runtime fixup
4. **Bundle** merged device ELF into `.hipfb`
5. **Host compilation** embeds fatbin via `-fcuda-include-gpubinary`

## Key Technical Details
- Function ID mapping: `host_table.cpp` uses packed 64-bit keys, parsed from comments
- Host table format: `{key, funcId}, // Coll Algo Proto Redop Type Acc Pipeline Unroll`
- Kernel descriptors (KDs): 64 bytes each in .rodata, contain entry point offsets
- PC-relative refs: `s_getpc_b64` + `s_add_u32` patterns patched for new addresses
- Relocations: `R_AMDGPU_RELATIVE64` for function table entries

## Known Issues
- Multi-GPU crashes with illegal memory access (debugging incomplete)
- GPU target defaults to gfx942, should auto-detect from system

## Ground Rules (from user)
- Don't use sudo without asking
- Don't make changes that can't be easily undone without asking
- Don't make substantial code edits (>few lines) without asking
- If a command isn't there, stop and ask
- 5 iteration max on build-test-debug loops before asking for help

## Transcript Location
Full conversation history: `/home/lmeadows/.cursor/projects/work-lmeadows-rccl/agent-transcripts/506f28b4-a8ef-41b5-b914-3f9a242b7057.txt`
