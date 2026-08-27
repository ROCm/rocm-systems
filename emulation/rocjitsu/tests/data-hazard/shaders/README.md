# Test Shaders & Mutation Testing

This directory contains HIP GPU kernel shaders and a mutation testing framework
that systematically removes `s_wait_*` synchronization instructions from compiled
assembly to detect data hazards.

## Running

From `tests/data-hazard`, with rocJitsu already built:

```bash
python -m mutate_and_test --hazard-detection --arch gfx950 shaders/*.hip
```

`--hazard-detection` runs every kernel under the rocJitsu launcher with the
`data_hazard` plugin enabled. Use `--rocjitsu-launcher`, `--rocjitsu-config`
and `--rocjitsu-build-dir` (or the matching `ROCJITSU_*` environment
variables) when the build is not in the default location.

## Overview

AMD GPUs execute memory operations asynchronously. The compiler inserts `s_wait_*`
instructions to ensure data is ready before it is used. Removing these waits can
expose **data hazards** — situations where a shader reads stale, uninitialized, or
partially written data. This framework automates that process:

1. **Compile** each HIP shader to GCN assembly (`.s`)
2. **Parse** the assembly to find all `s_wait_*` instructions
3. **Mutate** — remove each wait one at a time (replace with `s_nop 0`)
4. **Rebuild** from modified assembly → object → offload bundle → executable
5. **Run** each mutant and compare output to the baseline (unmutated) run
6. **Report** a matrix of which wait removals cause failures

## Wait Instruction Types

| Instruction | Counter | Triggered by | Purpose |
|-|-|-|-|
| `s_wait_loadcnt` | VMEM load | `global_load_*`, `buffer_load_*` | Wait for vector memory loads |
| `s_wait_storecnt` | VMEM store | `global_store_*`, `buffer_store_*` | Wait for stores to be visible |
| `s_wait_kmcnt` | SMEM | `s_load_*` | Wait for scalar (kernel arg) loads |
| `s_wait_dscnt` | LDS | `ds_read_*`, `ds_write_*` | Wait for shared memory ops |
| `s_wait_tensorcnt` | Tensor DMA | `tensor_load_to_lds` | Wait for TDM transfers into LDS |
| `s_wait_xcnt` | Address translation | any memory op that can XNACK | Wait for translation replay, not for data |
| `s_wait_samplecnt` | Sample | texture sampling | Wait for sample operations |
| `s_wait_bvhcnt` | BVH | ray tracing | Wait for BVH traversal |
| `s_waitcnt` | Legacy | all (GFX9/10) | Legacy combined wait |

## Hazard Categories

### By kind
- **RAW** (Read-After-Write) — reading a register/address before an async write completes
- **WAR** (Write-After-Read) — overwriting data that a pending async op still needs to read
- **WAW** (Write-After-Write) — two async writes to the same destination; final value undefined

### By memory location
- **Global** — device-visible VRAM (`global_load/store`)
- **Buffer** — buffer resource descriptors (`buffer_load/store`)
- **LDS** (shared) — workgroup-local data store (`ds_read/write`)
- **Scratch** (private) — per-thread stack memory (`scratch_load/store`)
- **Flat** — generic pointer addressing (could be global, LDS, or scratch)

### By atomicity
- **Atomic increment** — `atomicAdd`, `global_atomic_add_*`
- **Atomic swap** — `atomicExch`, `atomicCAS`

### Other complications
- Loops / control flow (`s_cbranch`, divergent execution)
- Execution mask changes (`s_and_saveexec_b32`, `v_cmpx_*`)
- Scalar loads out-of-order (`s_load_*` → `s_wait_kmcnt`)
- Clause grouping (`s_clause N` — N+1 ops form an unbreakable group)
- Cross-wave / cross-workgroup races
- Within-wave permute instructions

## Shaders

| Shader | Access type | Hazard pattern | Wait types exercised |
|-|-|-|-|
| `add_checked.hip` | Global vector | RAW (load→use) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `add_buffer.hip` | Buffer resource | RAW (buffer load→use) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `double_unsafe.hip` | Global vector | RAW (no bounds check) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `collatz.hip` | Global + control flow | RAW with loops/branches | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `atomic_inc.hip` | Global atomic | Atomic histogram | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `pointer_chase.hip` | Global vector | RAW (load result addresses the next load) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `war_pattern.hip` | Global vector | WAR (read then overwrite) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `waw_pattern.hip` | Global vector | WAW (two writes, same addr) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `scratch_access.hip` | Scratch (private) | RAW via scratch spills | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `lds_reduce.hip` | LDS (shared memory) | RAW via shared memory | `s_wait_dscnt`, `s_wait_kmcnt` |
| `lds_war_pattern.hip` | LDS (shared memory) | WAR (`ds_read` then `ds_write`, same slot) | `s_wait_dscnt`, `s_wait_kmcnt` |
| `tensor_lds.hip` | Tensor DMA → LDS | RAW (TDM write→LDS read) | `s_wait_tensorcnt`, `s_wait_dscnt` |
| `tensor_lds_offset.hip` | Tensor DMA → LDS at a nonzero descriptor base | RAW (TDM write→LDS read away from offset zero) | `s_wait_tensorcnt`, `s_wait_dscnt` |
| `fa_barrier_epoch.hip` | Tensor DMA + LDS + barriers | Cross-wave LDS reuse across barrier epochs | `s_wait_tensorcnt`, `s_wait_dscnt` |
| `trans_sin_cos.hip` | Global vector | RAW (load→`v_sin`/`v_cos`) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `trans_rcp_sqrt.hip` | Global vector | RAW (load→`v_rcp`/`v_sqrt`/`v_rsq`) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `trans_chain.hip` | Global vector | RAW (`v_exp`→`v_log` chain) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `trans_lds.hip` | LDS (shared memory) | RAW (LDS read→transcendental) | `s_wait_dscnt`, `s_wait_kmcnt` |
| `wmma_exp.hip` | WMMA (builtin) | RAW (WMMA→transcendental) | `s_wait_loadcnt`, `s_wait_kmcnt` |
| `wmma_rocwmma.hip` | WMMA (rocWMMA lib) | RAW (WMMA→transcendental) | `s_wait_loadcnt`, `s_wait_kmcnt` |

`wmma_rocwmma.hip` needs the rocWMMA headers; point the harness at them with
`--cxxflags "-I/path/to/rocwmma/include"`.

A kernel that only assembles for particular targets declares them with a
`// requires: <arch>` comment, and both the CLI and the pytest suite skip a
shader whose `requires:` list does not contain the target architecture.
`tensor_lds.hip`, `tensor_lds_offset.hip` and `wmma_exp.hip` declare
`// requires: gfx1250` because their builtins need the `gfx1250-insts` target
feature. A shader without that comment
is treated as portable and runs everywhere, so a kernel that names its target
only in prose is compiled for every architecture and fails there instead of
being skipped.

## Build Pipeline

The mutation test rebuilds from assembly following the
[ROCm assembly_to_executable](https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/assembly_to_executable) approach:

```text
 HIP source (.hip)
     │
     ├──► hipcc -S --cuda-device-only ──► device assembly (.s)
     │                                        │
     │                                    [MUTATE: replace s_wait with s_nop]
     │                                        │
     │                                        ▼
     │                                    clang -target amdgcn-amd-amdhsa
     │                                        │
     │                                        ▼
     │                                    device object (.o)
     │                                        │
     │                                    clang-offload-bundler
     │                                        │
     │                                        ▼
     │                                    offload bundle (.hipfb)
     │                                        │
     │                                    llvm-mc (.incbin embed)
     │                                        │
     │                                        ▼
     │                                    fatbin object (.o)
     │                                        │
     ├──► hipcc -c --cuda-host-only ──► host object (.o)
     │                                        │
     └────────────── hipcc (link) ◄───────────┘
                        │
                        ▼
                   executable
```

### Manual steps (for reference)

```bash
export TARGET_ARCH=gfx1250
export ROCM_PATH=$(rocm-sdk path --root)

# 1. Compile device assembly
hipcc -S --cuda-device-only --offload-arch=$TARGET_ARCH shader.hip -o shader.s

# 2. Compile host object
hipcc -c --cuda-host-only shader.hip -o shader_host.o

# 3. Assemble device code
$ROCM_PATH/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=$TARGET_ARCH shader.s -o shader_dev.o

# 4. Create offload bundle
$ROCM_PATH/llvm/bin/clang-offload-bundler -type=o -bundle-align=4096 \
  -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--$TARGET_ARCH \
  -input=/dev/null -input=shader_dev.o \
  -output=shader.hipfb

# 5. Embed bundle into host object
cat > hip_obj_gen.mcin << 'EOF'
    .type __hip_fatbin,@object
    .section .hip_fatbin,"a",@progbits
    .globl __hip_fatbin
    .p2align 12
__hip_fatbin:
    .incbin "shader.hipfb"
EOF
$ROCM_PATH/llvm/bin/llvm-mc -triple x86_64-unknown-linux-gnu \
  -o shader_fatbin.o hip_obj_gen.mcin --filetype=obj

# 6. Link
hipcc -o shader shader_host.o shader_fatbin.o
```

## Running the Mutation Test

### Prerequisites

- ROCm installation with `hipcc`, `clang`, `clang-offload-bundler`, `llvm-mc`
- Python 3.8+
- Either an AMD GPU matching the target architecture (default: `gfx1250`), or a
  rocJitsu build simulating it — with `--hazard-detection` the kernels run on the
  simulator and no physical GPU is needed

### Basic usage

The harness is a package. Run it from `tests/data-hazard`, not from this
directory:

```bash
# Every shader, with hazard detection
python -m mutate_and_test --hazard-detection --arch gfx950 shaders/*.hip

# Specific shaders
python -m mutate_and_test --hazard-detection shaders/collatz.hip shaders/double_unsafe.hip

# Keep the build artifacts and reports somewhere you can inspect them
python -m mutate_and_test --hazard-detection --keep-artifacts --output-dir ./build_mutants shaders/*.hip

# Verbose progress with a shorter timeout
python -m mutate_and_test --hazard-detection -v --timeout 60 shaders/*.hip
```

Without `--hazard-detection` the mutants run natively and no hazards are
collected at all: the run only tells you whether removing a wait changed the
output, which on real hardware it frequently does not.

### Options

| Flag | Default | Description |
|-|-|-|
| `shaders` | all `.hip` in `shaders/` | HIP source files to test |
| `--arch` | `gfx1250` / `$TARGET_ARCH` | GPU target architecture |
| `--rocm-path` | auto-detected / `$ROCM_PATH` | ROCm installation root |
| `--output-dir` | temp directory | Where to store artifacts & reports |
| `--timeout` | 240 | Per-execution timeout (seconds) |
| `--keep-artifacts` | off | Preserve build directory after run |
| `-v` / `--verbose` | off | Detailed progress output |
| `--hazard-detection` | off | Run under rocJitsu with the `data_hazard` plugin and collect its report |
| `--rocjitsu-launcher` | `$ROCJITSU_LAUNCHER`, else `<build-dir>/tools/rocjitsu/rocjitsu` | Launcher binary |
| `--rocjitsu-config` | `$ROCJITSU_CONFIG`, else the `configs/` entry matching `--arch` | Simulator config; must match `--arch` |
| `--rocjitsu-build-dir` | `$ROCJITSU_BUILD_DIR`, else `<rocjitsu>/build` | Where to find the launcher |
| `--target-features` | none | Comma-separated AMDGPU target features |
| `--cxxflags` | none | Extra `hipcc` flags for the assembly compile |
| `--include-xcnt` | off | Mutate `s_wait_xcnt` too (excluded by default: it tracks translation, not data) |
| `--subprocess-output` | off | Echo stdout/stderr of every subprocess; useful in CI |
| `--benchmark` | off | Record wall-clock time per kernel run |
| `--perf` | off | Profile each run with `perf record` and emit a flamegraph |
| `--flamegraph-dir` | auto-discovered | Where `flamegraph.pl` lives |

### Output

The harness produces:

1. **Terminal matrix** — human-readable table printed to stdout
2. **`mutation_report.json`** — machine-readable full report
3. **`mutation_report.csv`** — flat CSV for spreadsheet analysis
4. **`mutation_report.md`** — human-readable Markdown summary report
5. **`data_hazard_report_*.json`** — the plugin's own report for the baseline and
   each mutant, only under `--hazard-detection`
6. **`benchmark_report.json`** and **`flamegraph.svg`** — only under
   `--benchmark` and `--perf` respectively

All of these are written into the working directory and copied to the directory
you ran from.

### Example output

```
================================================================================
MUTATION TESTING RESULTS
================================================================================

--- double_unsafe ---
    Assembly: /tmp/hazard_mutate_xyz/double_unsafe.s
    Baseline: PASS (exit 0)
    Baseline hazards: 0
    Wait instructions found: 3
      #   Line  Instruction                   Status                Detection  Notes
    ---------------------------------------------------------------------------
      0     14  s_wait_dscnt 0x0              MATCH                    FN (0)
      1     21  s_wait_kmcnt 0x0              WRONG_OUTPUT       DETECTED (2)
      2     26  s_wait_loadcnt 0x0            WRONG_OUTPUT       DETECTED (1)

================================================================================
SUMMARY: 3 mutants | 2 killed | 1 survived
Mutation score: 66.7% of 3 that ran
False positives (baseline hazards): 0
False negatives (missed hazards):   1
================================================================================
```

### Interpreting results

The **Status** column is about the kernel's output, and says nothing about the
plugin:

| Status | Meaning |
|-|-|
| `MATCH` | Mutant ran and its stdout matched the baseline — the wait may be redundant |
| `WRONG_OUTPUT` | Mutant ran but its stdout differed from the baseline |
| `EXIT_N` | Mutant ran, output matched, but it exited non-zero |
| `TIMEOUT` | Execution exceeded `--timeout` — possible hang |
| `BUILD_FAIL` | Modified assembly failed to assemble or link |
| `RUN_FAIL` | Executable never started or never finished |

`BUILD_FAIL`, `TIMEOUT` and `RUN_FAIL` mutants never produced a comparable run,
so they are reported as **inconclusive** and excluded from the mutation score
rather than counted as killed. If every mutant is inconclusive the harness says
so and exits non-zero, because such a run measured nothing.

The **Detection** column, present only under `--hazard-detection`, is about the
plugin: `DETECTED (n)` means the mutant produced `n` hazards beyond the
baseline, and `FN (n)` means it produced no new hazards even though the wait was
removed. Hazards reported on an unmutated baseline are counted as false
positives.

**Important**: `MATCH` does not prove the wait is unnecessary. A hazard can be
benign on one run and not on the next, so the status column is a weaker signal
than the detection column.

## Adding New Test Shaders

To add a new shader:

1. Create a `.hip` file in this directory
2. Include `<hip/hip_runtime.h>` and `"utils.hpp"`
3. Write a `__global__` kernel exercising the hazard pattern
4. In `main()`:
   - Allocate and initialize host/device memory
   - Launch kernel, synchronize, copy back results
   - Print results as JSON array to **stdout**: `[val1,val2,...]`
   - Print `PASS` or `FAIL` to **stderr**
   - Return `EXIT_SUCCESS` on pass, `EXIT_FAILURE` on fail
   - Make the check reject the zero-initialized output buffer, so a kernel that
     stores nothing reports `FAIL` rather than `PASS`
5. From `tests/data-hazard`, run
   `python -m mutate_and_test --hazard-detection shaders/your_shader.hip` to verify
6. If the kernel only assembles for particular targets, add a
   `// requires: <arch>[, <arch>...]` comment so it is skipped elsewhere. Only
   that exact form is read, and everything after the colon is taken as the
   architecture list, so keep any explanation on its own line

### Template

```cpp
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "utils.hpp"

__global__ void my_kernel(const int* in, int* out, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = in[i] * 2;  // your hazard pattern here
    }
}

int main()
{
    constexpr unsigned int N = 256 * 256;
    std::vector<int> input(N), output(N);
    int *d_in, *d_out;

    for (unsigned int i = 0; i < N; ++i) input[i] = static_cast<int>(i);

    HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), N * sizeof(int), hipMemcpyHostToDevice));

    my_kernel<<<N / 256, 256>>>(d_in, d_out, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(), d_out, N * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    // JSON output to stdout
    std::cout << "[";
    for (unsigned int i = 0; i < N; ++i) {
        if (i > 0) std::cout << ",";
        std::cout << output[i];
    }
    std::cout << "]" << std::endl;

    // Validation to stderr
    bool pass = true;
    for (unsigned int i = 0; i < N; ++i) {
        if (output[i] != static_cast<int>(i) * 2) { pass = false; break; }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

## Dependencies

| Tool | Package | Purpose |
|-|-|-|
| `hipcc` | `hip-dev` / ROCm | HIP compiler (host + device) |
| `clang` | `rocm-llvm` | Assembles device code from `.s` |
| `clang-offload-bundler` | `rocm-llvm` | Creates offload bundles |
| `llvm-mc` | `rocm-llvm` | Embeds bundles into host objects |
| Python 3.8+ | system | Runs mutation test script |
