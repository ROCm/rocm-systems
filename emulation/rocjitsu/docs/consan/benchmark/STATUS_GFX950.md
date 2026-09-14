# ConSan `gfx950` benchmark status

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

| Workload | Uninstrumented Startup | Uninstrumented Run | SuperCollider Startup | SuperCollider Run | RecordReplay Startup | RecordReplay Run | Sampled Startup | Sampled Run | InlineShadow Startup | InlineShadow Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 4.3 s | 0.00351 s (1×) | 185 s | 0.00401 s (1.14×) | 278 s | 5.37 s (1,530×) | 226 s | 0.00774 s (2.2×) | 292 s | 50.5 s (14,400×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/pytorch-dense-prefill--native-validation.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | running | running | pending | pending | pending | pending | pending | pending | pending | pending | [native-reference-1: running](</home/benjacob/consan-gfx950-benchmark-full/pytorch-synthetic-decode--native-reference-1.log>) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| Gluon verified shared-memory round trip (1024 elements) | 0.608 s | 0.0000899 s (1×) | 0.612 s | 0.000127 s (1.41×) | 0.807 s | 0.0102 s (113×) | 0.617 s | 0.000109 s (1.21×) | 0.72 s | 0.0133 s (148×) | complete (retained) |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | 0.0000133 s | 0.0000115 s (1×) | 4.8 s | 0.0000165 s (1.44×) | 8.85 s | 0.00498 s (435×) | 7.8 s | 0.000382 s (33.4×) | 14.9 s | 7.52 s (656,000×) | complete (retained) |
| TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon attention prefill | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon attention decode | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Qwen3-0.6B prefill | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Qwen3-0.6B real cached decode | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Triton FP8 block-scaled GEMM | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 MoE | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
