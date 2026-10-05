# gfx1201 benchmark reevaluation — October 5, 2026

Columns compare the same workload under these configurations:

- **Workload:** the benchmark being measured.
- **Native:** execution without ConSan instrumentation.
- **Default:** ConSan's default analysis mode with `RJ_CONSAN_PRESET=default`.
- **High:** the same analysis mode with `RJ_CONSAN_PRESET=high`, which samples more workgroups and LDS cells, increasing coverage and usually overhead. It is a sampling preset, not a separate detector.
- **SuperCollider:** ConSan's alternative detector, which perturbs memory-access timing to expose races.
- **Native Run drift:** the percentage change between the final native Run sample and the initial native median; large drift makes overhead ratios unreliable.

Parenthesized multipliers are instrumented Run latency divided by native Run latency. See [presets](../USAGE.md#presets) for the sampling controls.

Times are milliseconds. Startup is instrumentation plus the first run; Run is the second-run latency. PyTorch/Gluon use synchronized host timing; hipBLASLt uses GPU event timing, so its Startup excludes host client setup. Every completed row uses fresh native profiling, numerical oracles, and static coverage checks. Drift refers to the final native Run sample versus the initial native median.

Measured source: `952b36716c2` (`shared/rocjitsu/sanitizers`), freshly rebuilt with GCC 15 (`-O2 -g`, assertions enabled). Physical device: AMD Radeon RX 9070 (`gfx1201`). Raw checkpoints, numerical results, coverage, runtime identities, and the qualification audit are in the [campaign directory](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20261005).

| Workload | Native startup / run | Default startup / run | High startup / run | SuperCollider startup / run | Native Run drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| gluon-shared-roundtrip | 474 / 0.0869 | 483 / 0.127 (1.46×) | 481 / 0.128 (1.48×) | 484 / 0.098 (1.13×) | -1.2% |
| hipblaslt-tensile-gemm | 0.0168 / 0.0133 | 572 / 0.112 (8.42×) | 592 / 0.819 (61.4×) | 502 / 0.0386 (2.9×) | -5.9% |
| pytorch-dense-prefill | 813 / 2.52 | 1.1e+04 / 3 (1.19×) | 1.08e+04 / 6.84 (2.71×) | 2.37e+04 / 2.74 (1.09×) | +7.8% |
| pytorch-synthetic-decode | 1.17e+03 / 2.72 | 1.17e+04 / 6.25 (2.3×) | 1.17e+04 / 12 (4.42×) | 2.5e+04 / 6.29 (2.31×) | -0.3% |
| pytorch-top1-moe-prefill | 1.05e+03 / 8.43 | 2.49e+04 / 14.9 (1.77×) | 2.5e+04 / 24 (2.85×) | 5.14e+04 / 10 (1.19×) | -4.6% |

All five workloads passed fresh native profiling, numerical oracles, and all three instrumented static-coverage gates: 15 mode cells and 60 per-run numerical checks. No builds, tests, or validation GPU work overlapped these timings. Native Run drift stayed within the campaign review bound of 10%. These are single-matrix observations, not confidence intervals.

Relative to the September 30 matrix, Default startup measured 11.0 / 11.7 / 24.9 seconds for dense prefill, synthetic decode, and MoE prefill, versus 12.1 / 12.8 / 26.1 seconds previously. Synthetic-decode SuperCollider Run increased from 3.96 to 6.29 ms; this observation is retained rather than replaced with a faster repeat. The two campaigns are not a controlled attribution of individual source changes.

The [September 30 profiling report](STARTUP_GFX1201_20260930.md) preserves the earlier startup investigation. The previous full table is retained in `STATUS_GFX1201.baseline.md` in the new campaign directory. Historical ASan/UBSan results are not fresh qualification of this revision. Benchmark acceptance does not establish fault-detection qualification; see [RDNA4 validation](../validation/STATUS_RDNA4.md).
