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
