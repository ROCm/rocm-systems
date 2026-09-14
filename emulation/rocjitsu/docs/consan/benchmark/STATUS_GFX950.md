# ConSan `gfx950` benchmark status

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

Previous measurements are marked **previous** until replaced by this run; their ratios use their original baseline. [Previous table](</home/benjacob/consan-gfx950-benchmark-full/status-before-scalar-barrier-fix.md>).
| Workload | Uninstrumented Startup | Uninstrumented Run | SuperCollider Startup | SuperCollider Run | RecordReplay Startup | RecordReplay Run | Sampled Startup | Sampled Run | InlineShadow Startup | InlineShadow Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 4.3 s | 0.00351 s (1×) | 173 s **previous** | 0.00386 s (1.1×) **previous** | 278 s **previous** | 5.37 s (1,530×) **previous** | 226 s **previous** | 0.00774 s (2.2×) **previous** | pending | pending | [supercollider--audit-on: running](</home/benjacob/consan-gfx950-benchmark-full/pytorch-dense-prefill--supercollider--audit-on.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| Gluon verified shared-memory round trip (1024 elements) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon attention prefill | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon attention decode | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Qwen3-0.6B prefill | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Qwen3-0.6B real cached decode | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Triton FP8 block-scaled GEMM | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 MoE | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
