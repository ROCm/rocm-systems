# ConSan `gfx950` benchmark status

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

| Workload | Uninstrumented Startup | Uninstrumented Run | SuperCollider Startup | SuperCollider Run | RecordReplay Startup | RecordReplay Run | Sampled Startup | Sampled Run | InlineShadow Startup | InlineShadow Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 4.3 s | 0.00351 s (1×) | 185 s | 0.00401 s (1.14×) | 278 s | 5.37 s (1,530×) | 226 s | 0.00774 s (2.2×) | 292 s | 50.5 s (14,400×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/pytorch-dense-prefill--native-validation.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | 3.73 s | 0.00361 s (1×) | 190 s | 0.00346 s (0.958×) | 269 s | 5.1 s (1,410×) | 252 s | 0.0077 s (2.13×) | 250 s | 28.2 s (7,800×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/pytorch-synthetic-decode--native-validation.log>) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 4.64 s | 0.0112 s (1×) | 326 s | 0.0137 s (1.22×) | 603 s | 21 s (1,870×) | 451 s | 0.021 s (1.87×) | 524 s | 72.1 s (6,410×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/pytorch-top1-moe-prefill--native-validation.log>) |
| Gluon verified shared-memory round trip (1024 elements) | 0.608 s | 0.0000899 s (1×) | 0.612 s | 0.000127 s (1.41×) | 0.807 s | 0.0102 s (113×) | 0.617 s | 0.000109 s (1.21×) | 0.72 s | 0.0133 s (148×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/gluon-shared-roundtrip--native-validation.log>) |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | 0.0000133 s | 0.0000115 s (1×) | 4.8 s | 0.0000165 s (1.44×) | 8.85 s | 0.00498 s (435×) | 7.8 s | 0.000382 s (33.4×) | 14.9 s | 7.52 s (656,000×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/hipblaslt-tensile-gemm--native-validation.log>) |
| TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096) | 0.124 s | 0.000628 s (1×) | 1.28 s | 0.00122 s (1.94×) | 6.22 s | 0.293 s (467×) | 5.38 s | 0.00536 s (8.54×) | 102 s | 96 s (153,000×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-bf16-gemm-mediumm--native-validation.log>) |
| TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096) | 0.121 s | 0.000649 s (1×) | 1.32 s | 0.00177 s (2.73×) | 8.58 s | 1.72 s (2,660×) | 5.39 s | 0.0213 s (32.8×) | timeout (600 s) | timeout (600 s) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-bf16-gemm-largem--native-validation.log>) |
| TokenSpeed Gluon attention prefill | 0.126 s | 0.000624 s (1×) | 1.37 s | 0.000818 s (1.31×) | 28.5 s | 7.05 s (11,300×) | 5.4 s | 0.00282 s (4.52×) | failed | failed | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-attention-prefill--native-validation.log>) |
| TokenSpeed Gluon attention decode | 0.128 s | 0.000664 s (1×) | 1.33 s | 0.000687 s (1.04×) | 5.71 s | 0.0178 s (26.8×) | 5.35 s | 0.000687 s (1.04×) | 5.62 s | 0.105 s (158×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-attention-decode--native-validation.log>) |
| TokenSpeed Qwen3-0.6B prefill | 3.77 s | 0.0255 s (1×) | running | running | pending | pending | pending | pending | skipped by request | skipped by request | [supercollider--audit-on: running](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-qwen-prefill--supercollider--audit-on.log>) |
| TokenSpeed Qwen3-0.6B real cached decode | pending | pending | pending | pending | pending | pending | pending | pending | skipped by request | skipped by request | not started: pending |
| TokenSpeed Triton FP8 block-scaled GEMM | pending | pending | pending | pending | pending | pending | pending | pending | skipped by request | skipped by request | not started: pending |
| TokenSpeed Gluon BF16 MoE | pending | pending | pending | pending | pending | pending | pending | pending | skipped by request | skipped by request | not started: pending |
