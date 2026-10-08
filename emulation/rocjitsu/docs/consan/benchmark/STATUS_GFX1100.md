# ConSan `gfx1100` benchmark status

For each mode, **Startup** sums instrumentation and the measured first run;
**Run** is the absolute second-run latency followed by its ratio to the
matching uninstrumented second run. PyTorch/Gluon use synchronized host
timing. hipBLASLt uses GPU event timing, so its Startup excludes host client
setup and is not an end-to-end cold-start measurement.

| Workload | Uninstrumented Startup | Uninstrumented Run | Default Mode Startup | Default Mode Run | Default Mode (high) Startup | Default Mode (high) Run | SuperCollider Startup | SuperCollider Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| PyTorch synthetic dense decode (one continuous-batch tick) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| Gluon verified shared-memory round trip (1024 elements) | 0.442 s | 0.000107 s (1×) | 0.446 s | 0.000139 s (1.29×) | 0.448 s | 0.000134 s (1.25×) | 0.447 s | 0.00011 s (1.02×) | native validation accepted; drift reviewed below |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |

This probe used a physical Radeon PRO W7900 with native `gfx1100` code.
All three modes passed the numeric oracle and patched 32/32 applicable sites.
The kernel uses one wave32 wave; it does not qualify
cross-wave fault detection or Aorta model performance.

The table uses an independent repeat measurement. Native validation drift was
+0.75% for Run1 and +0.31% for Run2, within the prospectively recorded 5% review
limit. An earlier measurement was excluded from performance admission because
Run2 drift was -12.96%, despite the runner accepting each cell.
The approximately 0.1 ms second-run times are
small-probe measurements; the current runner records drift without enforcing
an automatic threshold, so final admission still requires this review.

Use the maintained [benchmark workflow](BENCHMARK.md) for reproduction and
retain exact runtime versions, commands, and raw measurements in execution
records. These measurements do not establish performance across other workload
shapes, toolchains, or devices.
