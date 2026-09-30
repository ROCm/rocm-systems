# gfx1201 benchmark reevaluation — September 30, 2026

Columns compare the same workload under these configurations:

- **Workload:** the benchmark being measured.
- **Native:** execution without ConSan instrumentation.
- **Default:** ConSan's default analysis mode with `RJ_CONSAN_PRESET=default`.
- **High:** the same analysis mode with `RJ_CONSAN_PRESET=high`, which samples more workgroups and LDS cells, increasing coverage and usually overhead. It is a sampling preset, not a separate detector.
- **SuperCollider:** ConSan's alternative detector, which perturbs memory-access timing to expose races.
- **Native Run drift:** the percentage change between the final native Run sample and the initial native median; large drift makes overhead ratios unreliable.

Parenthesized multipliers are instrumented Run latency divided by native Run latency. See [presets](../USAGE.md#presets) for the sampling controls.

Times are milliseconds. Startup is instrumentation plus the first run; Run is the second-run latency. PyTorch/Gluon use synchronized host timing; hipBLASLt uses GPU event timing, so its Startup excludes host client setup. Every completed row uses fresh native profiling, numerical oracles, and static coverage checks. Drift refers to the final native Run sample versus the initial native median.

Measured source: `72a843d3b61` (`shared/rocjitsu/sanitizers`). Raw checkpoints, numerical results, coverage records, runtime identities, and the timing review are in the [campaign directory](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260930).

| Workload | Native startup / run | Default startup / run | High startup / run | SuperCollider startup / run | Native Run drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| gluon-shared-roundtrip | 435 / 0.09 | 444 / 0.12 (1.34×) | 441 / 0.127 (1.41×) | 437 / 0.0958 (1.06×) | +4.3% |
| hipblaslt-tensile-gemm | 0.0196 / 0.013 | 691 / 0.12 (9.29×) | 672 / 0.791 (61.1×) | 587 / 0.0391 (3.02×) | +1.1% |
| pytorch-dense-prefill | 736 / 3.34 | 1.21e+04 / 3.86 (1.16×) | 1.24e+04 / 7.3 (2.19×) | 2.46e+04 / 3.26 (0.976×) | -7.5% |
| pytorch-synthetic-decode | 1.07e+03 / 2.72 | 1.28e+04 / 7.48 (2.75×) | 1.29e+04 / 12 (4.43×) | 2.47e+04 / 3.96 (1.46×) | -0.8% |
| pytorch-top1-moe-prefill | 994 / 9.37 | 2.61e+04 / 15.6 (1.67×) | 2.59e+04 / 27.4 (2.93×) | 5.06e+04 / 10.1 (1.08×) | -7.8% |

All five workloads passed fresh native profiling, numerical oracles, and all three instrumented static-coverage gates: 15 mode cells and 60 per-run numerical checks. No compilation, tests, or other campaign GPU work overlapped these timings. Native Run drift stayed within the campaign review bound of 10%; the exact values above remain relevant when comparing short Run latencies. These are single-matrix observations, not confidence intervals. hipBLASLt’s tiny native first-operation sample drifted −13.9%; its native Run drift was +1.1%.

Compared with the September 26 table, Default startup fell from 78.5 / 90.1 / 167 seconds to 12.1 / 12.8 / 26.1 seconds for dense prefill, synthetic decode, and MoE prefill (about 6.5× / 7.0× / 6.4× faster). The [September 30 profiling report](STARTUP_GFX1201_20260930.md) explains the successive startup fixes; this fresh full matrix also measures High and SuperCollider.

hipBLASLt is an exception: startup increased from 593 / 577 / 409 ms to 691 / 672 / 587 ms. Fresh old-hook versus current-hook controls reproduce the increase. The old hook stopped checking hazards after retaining 32 diagnostics; current waitcheck correctly checks and counts all 261, retaining 32 details. Selected-object wait checking therefore increased from roughly 81 to 265 ms, while inventory and patching became faster. This comparison includes additional correctness work; reverting to the old early stop would restore inaccurate diagnostic totals. The [control and profiling evidence](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260930/hipblaslt-controls) is retained separately from the table’s main campaign.

The historical synthetic-decode Run increase did not recur consistently in a matched-hook comparison. Fresh old/current hook latencies were 7.08 / 5.89 ms (Default), 12.67 / 12.83 ms (High), and 4.176 / 4.178 ms (SuperCollider); Default startup was 91.09 / 12.74 seconds. Both controls passed all numerical and coverage gates. The old run's native Run reference drifted −28.1%, so its overhead ratios are rejected and these direct-latency comparisons remain exploratory; the current repeat drifted −0.9%. The table above retains the original complete matrix rather than substituting the faster repeat. See [control evidence](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260930/decode-controls-audit.json).

The [Clang sanitizer report](../validation/CLANG_SANITIZERS_20260930.md) covers the complete UBSan and ASan suites on the same executable sources. Benchmark acceptance does not establish fault-detection qualification; see [RDNA4 validation](../validation/STATUS_RDNA4.md).
