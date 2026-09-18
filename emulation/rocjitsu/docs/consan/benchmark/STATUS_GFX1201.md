# ConSan `gfx1201` benchmark status

For each mode, **Startup** sums instrumentation and the measured first run;
**Run** is the absolute second-run latency followed by its ratio to the
matching uninstrumented second run. PyTorch/Gluon use synchronized host
timing. hipBLASLt uses GPU event timing, so its Startup excludes host client
setup and is not an end-to-end cold-start measurement.

| Workload | Uninstrumented Startup | Uninstrumented Run | Default Mode Startup | Default Mode Run | Default Mode (high) Startup | Default Mode (high) Run | SuperCollider Startup | SuperCollider Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 0.77 s | 0.00369 s (1×) | 190 s | 0.00386 s (1.05×) | 189 s | 0.00655 s (1.78×) | 153 s | 0.00326 s (0.885×) | [native-validation: accepted](</tmp/consan-benchmark-gfx1201-refresh-fixed/pytorch-dense-prefill--native-validation.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | 1.11 s | 0.00267 s (1×) | 198 s | 0.00704 s (2.64×) | 198 s | 0.0102 s (3.84×) | 155 s | 0.00387 s (1.45×) | [native-validation: accepted](</tmp/consan-benchmark-gfx1201-refresh-fixed/pytorch-synthetic-decode--native-validation.log>) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 1.07 s | 0.00883 s (1×) | 361 s | 0.0141 s (1.59×) | 362 s | 0.023 s (2.61×) | 280 s | 0.0106 s (1.2×) | [native-validation: accepted](</tmp/consan-benchmark-gfx1201-refresh-fixed/pytorch-top1-moe-prefill--native-validation.log>) |
| Gluon verified shared-memory round trip (1024 elements) | 0.429 s | 0.00018 s (1×) | 0.433 s | 0.000217 s (1.2×) | 0.43 s | 0.000198 s (1.1×) | 0.43 s | 0.000175 s (0.973×) | [native-validation: accepted](</tmp/consan-benchmark-gfx1201-refresh-fixed/gluon-shared-roundtrip--native-validation.log>) |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | 0.0000125 s | 0.00000948 s (1×) | 0.897 s | 0.0000895 s (9.44×) | 0.886 s | 0.000733 s (77.3×) | 0.57 s | 0.000026 s (2.74×) | [native-validation: accepted](</tmp/consan-benchmark-gfx1201-refresh-fixed/hipblaslt-tensile-gemm--native-validation.log>) |

## Run notes

- Completed 2026-09-17 (America/Toronto) on the physical gfx1201 Radeon RX 9070,
  with one GPU workload process at a time. All 15 instrumented configurations
  passed numerical validation and complete selected-site coverage checks.
- Measured code: `4d73e1ce876`; Release Clang/lld build against TheRock
  `10.1.0a20260822`, matching PyTorch `2.15.0a0+rocm10.1.0a20260822` and Triton.
  The hipBLASLt client comes from the matching official TheRock tests archive.
- Default Mode uses `RJ_CONSAN_PRESET=default` (workgroup/cell strides 256/256);
  Default Mode (high) uses `RJ_CONSAN_PRESET=high` (16/16). SuperCollider has no
  preset. Instrumentation coverage does not imply exhaustive runtime sampling.
- Final native second-run checks differed from the initial references by
  −6.6% (prefill), −0.5% (decode), −4.3% (MoE), −9.9% (Gluon), and −1.0%
  (hipBLASLt). Small ratios near 1×, including apparent speedups, are noisy.
- The full measured campaign took 36.5 minutes, chiefly cold instrumentation of
  PyTorch library code. The initial failed campaign is excluded. A WMMA spill
  ordering fix was validated before this campaign, including a physical
  regression that fails with the old hook and passes with the fix.
- [Summary and per-cell provenance](/tmp/consan-benchmark-gfx1201-refresh-fixed/summary.json),
  [final audit](/tmp/consan-benchmark-gfx1201-refresh-fixed/audit.json), and
  [environment setup and reproduction notes](/tmp/consan-benchmark-refresh/README.md)
  are retained locally. Later fixes should rerun only affected cells; a mixed-commit
  matrix is acceptable when each cell retains its provenance.
