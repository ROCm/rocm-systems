# gfx1201 benchmark reevaluation — September 26, 2026

Columns compare the same workload under these configurations:

- **Workload:** the benchmark being measured.
- **Native:** execution without ConSan instrumentation.
- **Default:** ConSan's default analysis mode with `RJ_CONSAN_PRESET=default`.
- **High:** the same analysis mode with `RJ_CONSAN_PRESET=high`, which samples more workgroups and LDS cells, increasing coverage and usually overhead. It is a sampling preset, not a separate detector.
- **SuperCollider:** ConSan's alternative detector, which perturbs memory-access timing to expose races.
- **Native Run drift:** the percentage change between the final native Run sample and the initial native median; large drift makes overhead ratios unreliable.

Parenthesized multipliers are instrumented Run latency divided by native Run latency. See [presets](../USAGE.md#presets) for the sampling controls.

Times are milliseconds. Startup is instrumentation plus the first run; Run is the second-run latency. PyTorch/Gluon use synchronized host timing; hipBLASLt uses GPU event timing, so its Startup excludes host client setup. Every completed row uses fresh native profiling, numerical oracles, and static coverage checks. Drift refers to the final native Run sample versus the initial native median.

| Workload | Native startup / run | Default startup / run | High startup / run | SuperCollider startup / run | Native Run drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| gluon-shared-roundtrip | 425 / 0.0872 | 440 / 0.107 (1.22×) | 434 / 0.106 (1.21×) | 434 / 0.0977 (1.12×) | -3.1% |
| hipblaslt-tensile-gemm | 0.0126 / 0.00888 | 593 / 0.0712 (8.02×) | 577 / 0.496 (55.8×) | 409 / 0.0252 (2.84×) | -0.4% |
| pytorch-dense-prefill | 724 / 2.94 | 7.85e+04 / 3.46 (1.18×) | 7.93e+04 / 6.06 (2.06×) | 5.6e+04 / 3.42 (1.16×) | +0.5% |
| pytorch-synthetic-decode | 1.05e+03 / 2.66 | 9.01e+04 / 5.68 (2.13×) | 9.23e+04 / 8.98 (3.37×) | 6.08e+04 / 3.52 (1.32×) | -1.1% |
| pytorch-top1-moe-prefill | 1.11e+03 / 7.56 | 1.67e+05 / 13.1 (1.73×) | 1.73e+05 / 20.8 (2.75×) | 1.05e+05 / 9.09 (1.2×) | +15.3%; timing provisional |

September 30 follow-up: [startup profiling and no-op preparation fix](STARTUP_GFX1201_20260930.md)
reproduces the default-preset overhead on current code and reduces dense-prefill /
synthetic-decode startup from 85.0 / 97.7 s to 29.0 / 32.9 s. This focused result
does not replace the historical full-matrix table above.
Further iterations in that report reduce startup to 20.8–21.1 / 22.5–22.8 s
in two endpoint runs, by sharing block-position indices and invariant kernel
call facts. Selected-site coverage and numerical checks remain unchanged.
