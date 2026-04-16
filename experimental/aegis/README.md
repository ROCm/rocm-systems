# AegisBit

Runtime memory profiler for AMD GPUs. Intercepts HIP/ROCm kernel dispatches, instruments CDNA ISA with profiling payloads, and reports VMEM coalescing efficiency and LDS bank conflicts — all without recompilation.

## Status

The full intercept → instrument → dispatch → analyze pipeline works end-to-end on gfx950 (CDNA4 / MI350X). Tested on:

- Standard Triton kernels (matmul, softmax, flash attention, MoE GEMM)
- Gluon kernels with explicit shared memory layouts (MFMA + async copy)
- HIP kernels (vector add, multi-kernel dispatches)
- rocBLAS/Tensile GEMM kernels (FP16, extracted from CCOB fat binaries)
- High register pressure kernels (250+ VGPRs, 100+ SGPRs via zero-SGPR trampoline with bidirectional island chaining for full coverage)

### What It Reports

| Metric | Description |
|--------|-------------|
| **VMEM coalescing** | Per-instruction cache-line efficiency (%), access pattern (coalesced / scattered) |
| **LDS bank conflicts** | Per-instruction conflict cycles, conflict-free percentage |

Reports include source-level attribution (file, line, source text) via DWARF debug info.

### Known Limitations

- **gfx950 only** — instruction encoding is architecture-specific; other targets need validation
- **Multi-GPU / HIP graph** — untested
- **Multi-word LDS instructions** (`ds_read_b64`, `ds_read_b128`) — only the first phase's bank mapping is analyzed

## Installation

### Binary (recommended)

AegisBit ships as a self-contained shared library with LLVM statically linked. No LLVM build required.

```
aegisbit-<version>-gfx950/
├── lib/libaegisbit.so    # ~114 MB, all LLVM components built-in
└── tools/
    ├── aegisbit           # Python CLI wrapper
    └── smoke-test.sh      # Verify the binary works on this machine
```

**Requirements:** ROCm 6.x+ (provides `libhsa-runtime64` and `libamdhip64`), Python 3.8+, glibc 2.38+ (Ubuntu 24.04+ / RHEL 9.4+)

```bash
# Download and extract a release
tar xf aegisbit-<version>-gfx950.tar.gz
cd aegisbit-<version>-gfx950

# Verify the binary works (compiles a test kernel, profiles it, checks output)
./tools/smoke-test.sh

# Profile a kernel
./tools/aegisbit --triton -- python3 my_script.py
```

### Building from Source

**Prerequisites:**
- LLVM 23+ with AMDGPU backend (built from source or system package; validated against upstream LLVM 23.0.0git)
- ROCm/HIP runtime (typically `/opt/rocm`)
- CMake 3.20+, C++17 compiler

```bash
# 1. Build LLVM (if not using system package)
cd llvm-project && mkdir -p build && cd build
cmake ../llvm -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON -G Ninja
ninja

# 2. Build AegisBit
cd aegisbit
./build.sh build       # Build
./build.sh test        # Build + run C++ unit/integration tests
./build.sh e2e         # Build + run E2E Python tests (GPU + torch + triton required)
./build.sh test-all    # Run both C++ and E2E tests
./build.sh clean       # Clean build directory
./build.sh configure   # Run cmake configuration only
./build.sh all         # Build + run C++ tests (default when no argument given)
```

CMake will search for LLVM in `../llvm-project-amd-staging/build`, then `../llvm-project/build`, then `/usr/lib/llvm-23`. Override with `-DLLVM_DIR=<path>`.

### Building a Binary Release

To produce a self-contained tarball for distribution (no LLVM needed on target):

```bash
./release.sh               # -> build/aegisbit-0.1.0-gfx950.tar.gz
./release.sh --arch gfx942 # Override target architecture label
```

The script builds, verifies LLVM is statically linked, reports the minimum glibc/libstdc++ versions, and packages `lib/libaegisbit.so` + `tools/aegisbit` into a tarball.

## Profiling

The `tools/aegisbit` CLI wraps `LD_PRELOAD` setup with a user-friendly interface. It produces human-readable VMEM coalescing and LDS bank-conflict reports on stderr, with optional structured JSON output.

```bash
aegisbit [OPTIONS] -- COMMAND [ARGS...]
```

### Quick Start

```bash
# Profile any HIP program
./tools/aegisbit -- python3 my_script.py

# Filter to specific kernels (glob patterns)
./tools/aegisbit --filter="matmul*" -- python3 my_script.py

# Auto-discover Triton kernels + JSON output
./tools/aegisbit --triton --json -- python3 my_triton_script.py

# Write JSON report to a file
./tools/aegisbit --triton -o report.json -- python3 my_triton_script.py

# Preview what would run without executing
./tools/aegisbit --dry-run -- python3 my_script.py
```

You can also use `LD_PRELOAD` directly:

```bash
LD_PRELOAD=build/src/libaegisbit.so \
  AEGISBIT_ENABLED=1 \
  AEGISBIT_KERNELS="my_kernel*" \
  python3 my_script.py
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--filter=PATTERNS` | Comma-separated kernel name globs |
| `--triton` | Auto-discover `@triton.jit` kernels from the target `.py` file |
| `--json` | Emit structured JSON to stdout after the run |
| `-o`, `--output FILE` | Write JSON report to FILE (implies `--json`) |
| `-v`, `--verbose` | Enable library debug logging |
| `--max-sites N` | Cap instrumentation sites per kernel |
| `--dry-run` | Print environment and command without executing |

### Example Output

**Stderr** (live during execution):

```
=== VMEM Coalescing: matmul_kernel.kd ===
Overall efficiency: 100.0%  (6 sites, 393216 samples)

  a = tl.load(a_ptr + offs_am[:, None] * K + offs_k[None, :]) 2×load  eff=100%  cachelines=2  coalesced
  b = tl.load(b_ptr + offs_k[:, None] * N + offs_bn[None, :]) 2×load  eff=100%  cachelines=2  coalesced
  tl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :])   2×store eff=100%  cachelines=2  coalesced

=== LDS Bank Conflicts: matmul_kernel.kd ===
8 sites, 393216 samples

  tl.dot(a, b)    4×load  conflict_cycles=8  conflict_free=0%
  tl.dot(a, b)    4×load  conflict_cycles=4  conflict_free=0%
```

**JSON** (`--json` or `-o`):

```json
{
  "version": 1,
  "kernels": [{
    "name": "matmul_kernel.kd",
    "vmem_coalescing": {
      "overall_efficiency_pct": 100.0,
      "num_sites": 6,
      "sites": [{ "source_text": "a = tl.load(...)", "avg_efficiency_pct": 100.0, "pattern": "coalesced" }]
    },
    "lds_bank_conflicts": {
      "num_sites": 8,
      "sites": [{ "source_text": "tl.dot(a, b)", "avg_conflict_cycles": 8.0, "conflict_free_pct": 0.0 }]
    }
  }]
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGISBIT_ENABLED` | `1` | Enable/disable instrumentation |
| `AEGISBIT_MODE` | `MEMORY_ONLY` | Instrumentation mode (currently only `MEMORY_ONLY`) |
| `AEGISBIT_STRATEGY` | `on_gpu_reduce` | Profiling strategy (`on_gpu_reduce` or `full_capture`) |
| `AEGISBIT_KERNELS` | `*` | Comma-separated kernel name patterns (glob) |
| `AEGISBIT_JSON_OUTPUT` | *(disabled)* | Path for structured JSON report |
| `AEGISBIT_MAX_SITES` | *(unlimited)* | Cap on instrumentation sites per kernel |
| `AEGISBIT_LOG` | `0` | Enable debug logging (`1` = info, `2` = verbose) |
| `AEGISBIT_LIB` | *(auto-detect)* | Explicit path to `libaegisbit.so` (CLI only) |
| `AEGISBIT_BUFFER_MB` | `64` | Trace buffer size in MB |
| `AEGISBIT_OUTPUT` | `./aegisbit_traces/` | Directory for trace output files |

## How It Works

1. **Intercept**: `LD_PRELOAD=libaegisbit.so` hooks HSA queue creation and executable freeze points directly
2. **Disassemble**: LLVM MC layer decodes the kernel's `.text` section into individual AMDGPU instructions
3. **Analyze**: Build CFG, compute register liveness, locate dead registers for instrumentation scratch space
4. **Instrument**: For each VMEM/LDS instruction, inject a trampoline that:
   - Saves architectural state (EXEC, SCC, VCC, scratch registers)
   - Captures the memory address from the instruction's address VGPR
   - Runs a compiler-generated payload (LLVM IR → native AMDGPU machine code):
     - **VMEM**: counts unique cache lines via wave-intrinsic `v_cmp`/`v_readfirstlane` loop
     - **LDS**: counts max lanes per bank via `v_cmp`/`s_bcnt1` popcount loop
   - Accumulates the result into a per-site counter via `global_atomic_add`
   - Restores all state and resumes the original instruction
5. **Fixup**: Recalculate branch targets (including long-branch trampolines), update kernel descriptor (VGPR/SGPR counts, kernarg size)
6. **Rebuild**: Construct new ELF code object with patched `.text` and updated `.note` metadata
7. **Dispatch**: Load patched kernel into HSA, redirect dispatch with extended kernarg (trace buffer pointer)
8. **Report**: After kernel completion, read per-site counters and generate coalescing/bank-conflict analysis with source-level attribution via DWARF debug info

### Key Design Decisions

- **VMEM stores only**: SMEM `s_store_dword` goes through scalar cache which isn't CPU-coherent on CDNA. All trace buffer writes use `global_store_dword` (VMEM pipeline through L2) for host visibility.
- **`s_mov_b32` is SCC-safe**: EXEC save/restore uses `s_mov_b32` which does NOT clobber SCC (unlike `s_or_b64`). Critical for kernels with SCC-dependent control flow.
- **Compiler-generated payloads**: Counting loops are built programmatically as LLVM IR, compiled to native AMDGPU machine code, and spliced into the trampoline as raw bytes. This avoids hand-assembled payloads and makes the logic auditable.
- **Zero-SGPR trampoline mode**: For kernels that exhaust SGPR budget (100+ SGPRs), instrumentation falls back to a VCC-temp mode that uses zero additional SGPRs by spilling through VGPRs and scratch memory. All jumps use `s_branch` (±128 KB range).
- **Bidirectional island chaining**: In zero-SGPR mode, trampoline islands can exceed the `s_branch` ±128 KB range for large kernels. Islands are placed both after *and* before the kernel in the `.text` section, splitting instrumentation sites across forward and backward islands. This achieves full coverage (e.g., 288/288 sites on rocBLAS Tensile kernels) where forward-only placement would cap at ~83%.
- **Above-the-count registers**: Instrumentation VGPRs are allocated above the kernel descriptor's declared count. The hardware allocates in granules of 4-8, so there is padding available without increasing actual occupancy in most cases.

## Project Structure

```
aegisbit/
├── include/aegisbit/      # Public headers
├── src/
│   ├── disasm/            # LLVM disassembler + instruction builder (encoder)
│   ├── analysis/          # CFG builder, coalescing analyzer, source mapper
│   ├── codegen/           # Payload compiler (LLVM IR → AMDGPU machine code)
│   ├── transform/         # Trampoline bridge (binary patching + instrumentation)
│   ├── fixup/             # Kernel descriptor updates (VGPR/SGPR counts, branches)
│   ├── codeobj/           # ELF code object parsing / building / note metadata
│   ├── intercept/         # Dispatch interception (LD_PRELOAD + HSA API tables)
│   ├── runtime/           # KernelPatcher, TracingEngine, RuntimeConfig
│   └── launch/            # HSA kernel launcher
├── tools/
│   ├── aegisbit           # Python CLI wrapper
│   └── smoke-test.sh      # Binary release verification
├── test/
│   ├── run_e2e.py         # E2E test runner
│   ├── unit/              # GoogleTest suites (no GPU required)
│   ├── integration/       # GPU integration tests (require AMD GPU)
│   ├── triton/            # E2E Python tests (Triton/Gluon kernels)
│   └── fuzz/              # libFuzzer targets (instruction decoder, ELF parser)
├── build.sh               # Build script
└── release.sh             # Binary release tarball builder
```

## Testing

```bash
# C++ unit + integration tests (21 unit + 4 GPU = 25 tests via CTest)
./build.sh test

# E2E profiling tests (18 tests — Triton, HIP, rocBLAS, stress; requires GPU + torch + triton)
./build.sh e2e

# All tests (C++ + E2E)
./build.sh test-all
```

The E2E suite (`test/run_e2e.py`) covers:

| Category | Tests |
|----------|-------|
| **Triton** | simple correctness, coalescing correctness, coalescing profiling, diverse kernels, LDS bank conflicts, flash attention, MoE GEMM, profiler validation |
| **HIP** | vector add, multi-kernel |
| **rocBLAS** | Tensile GEMM (zero-SGPR mode, full 288-site coverage) |
| **Stress** | max VGPR, LDS heavy, huge mem, big GEMM, multi mixed, wide loads, stencil |

Individual E2E tests can be run with `-k`:

```bash
python3 test/run_e2e.py -v -k rocblas       # Just rocBLAS tests
python3 test/run_e2e.py -v -k coalescing    # Just coalescing tests
python3 test/run_e2e.py --list               # List all available tests
python3 test/run_e2e.py --triton-only        # Only Triton kernel tests
python3 test/run_e2e.py --hip-only           # Only HIP kernel tests
python3 test/run_e2e.py --rocblas-only       # Only rocBLAS tests
python3 test/run_e2e.py --stress-only        # Only stress tests
```

## License

To be determined.
