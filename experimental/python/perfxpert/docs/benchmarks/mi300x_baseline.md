# MI300X Benchmark Baseline

_Generated from `benchmarks/baseline/mi300x_baseline.json`._

## Hardware
- MI300X (gfx942), 304 CUs, 192 GB HBM3
- ROCm 6.4, kernel 5.15, Ubuntu 22.04

## Suites
1. **TritonBench-ROCm** — 40+ Triton-lowered kernels
2. **KernelBench-ROCm** level-1 — 100+ elementary kernels

## Methodology
For each kernel:
1. rocprofv3 --sys-trace → `rocprof.db`
2. `perfxpert analyze -i rocprof.db --offline --emit-patch .perfxpert.patch`
3. `git apply .perfxpert.patch`
4. Benchmark kernel in optimized state
5. Report baseline_ns, optimized_ns, speedup, whether perfxpert emitted a recommendation

Per-kernel measurements use median of 30 runs after 3 warm-up iterations.

## Gate
- Weighted geomean across **kernels where perfxpert recommended a change**
  must be **≥ 1.20×**
- Weight: baseline_ns (longer-running kernels count more)

## Current published numbers
_See mi300x_baseline.json for machine-readable per-kernel data._

### TritonBench-ROCm
- matmul_f16_256x256: 1.000 → 0.700 ms (1.429× speedup, perfxpert applied)
- softmax_f32_1024: 0.500 → 0.300 ms (1.667× speedup, perfxpert applied)
- layernorm_f16_4096: 0.800 → 0.800 ms (1.000× speedup, no perfxpert recommendation)

### KernelBench-ROCm level-1
- gemm_rocblas: 1.000 → 0.750 ms (1.333× speedup, perfxpert applied)
- softmax_fused: 0.500 → 0.500 ms (1.000× speedup, no perfxpert recommendation)

### Combined
- Total kernels: 5
- Kernels with perfxpert recommendation: 3
- **Weighted geomean (applied only): 1.433×**

## Tolerances for nightly
- Geomean delta > -2% vs baseline → investigate
- Any single kernel regression > -10% → fail nightly
- New kernels added → geomean computed on intersection set only

## How to refresh the baseline
1. Open PR editing `benchmarks/baseline/mi300x_baseline.json`
2. CI runs suites on self-hosted MI300X runner
3. If geomean ≥ 1.20× and no kernel regresses > 10%, PR is reviewable
4. Two core reviewers approve → merge
